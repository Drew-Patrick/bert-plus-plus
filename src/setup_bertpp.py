"""
setup_bertpp.py

The tokenizer, the streaming datasets, the MLM collator, and the model all
live here. The notebook and train.py both import from this file so there is
only one copy of everything.

A few decisions worth writing down:
  * BERT-Large size. 24 layers, 16 heads, 1024 dim, with parallel
    attention/FFN blocks like PaLM, SwiGLU feed forwards, and XPos rotary
    embeddings.
  * ff_dim uses the usual SwiGLU sizing of about (8/3)*d rounded to a
    multiple of 256, which is 2816 for d=1024. That keeps the parameter
    count in line with a normal 4*d FFN. With tied weights the model comes
    out to 340,925,242 parameters, about 340.9M.
  * Attention is one fused QKV projection with precomputed RoPE and XPos
    tables. FlashAttention-2 kicks in when it is installed and there is no
    padding mask, otherwise torch scaled_dot_product_attention.
  * Gradient checkpointing uses use_reentrant=False since that is the
    version that plays nice with FSDP.
"""

import math
import os
import warnings

import torch
import torch.nn.functional as F
from datasets import interleave_datasets, load_dataset
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    from flash_attn.flash_attn_interface import flash_attn_func
    _flash_available = True
except ImportError:
    _flash_available = False

torch.set_float32_matmul_precision("high")

MAX_SEQ_LEN = 512


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def get_tokenizer():
    """WordPiece tokenizer matching the BERT vocabulary (30,522 tokens)."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


# ---------------------------------------------------------------------------
# Streaming datasets: The Pile (uncopyrighted) interleaved 50/50 with C4
# ---------------------------------------------------------------------------
def _stream_pile(use_streaming: bool):
    """
    Try to obtain The Pile stream in this order:
      1. local 'pile.jsonl' file
      2. public 'monology/pile-uncopyrighted' on Hugging Face
      3. None  (caller falls back to C4 only)
    """
    if os.path.exists("pile.jsonl"):
        return load_dataset(
            "json", data_files="pile.jsonl", split="train", streaming=use_streaming
        )
    try:
        return load_dataset(
            "monology/pile-uncopyrighted",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:  # network / dataset availability
        warnings.warn(f"Could not stream The Pile - {e}. Falling back to C4 only.")
        return None


def get_datasets(tokenizer, use_streaming: bool = True):
    """
    Build the iterable pre-training dataset.

      * Streams The Pile when it is available, and always streams AllenAI C4.
      * Interleaves the two 50/50 if both are present.
      * Shuffles with a fixed-size buffer, tokenizes on the fly, and keeps
        only the columns the collator needs.
    """
    pile_stream = _stream_pile(use_streaming)
    c4_stream = load_dataset("allenai/c4", "en", split="train", streaming=use_streaming)

    if pile_stream is None:
        dataset = c4_stream
    else:
        dataset = interleave_datasets(
            [pile_stream, c4_stream], probabilities=[0.5, 0.5], seed=42
        )

    dataset = dataset.shuffle(buffer_size=10_000, seed=42)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_special_tokens_mask=True,
        )

    dataset = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])

    cols_to_keep = {"input_ids", "special_tokens_mask"}
    cols_to_remove = [c for c in dataset.column_names if c not in cols_to_keep]
    return dataset.remove_columns(cols_to_remove)


# ---------------------------------------------------------------------------
# MLM collation (hand-written)
# ---------------------------------------------------------------------------
def make_mlm_collate(tokenizer, max_seq_len: int = MAX_SEQ_LEN, mlm_prob: float = 0.15):
    """
    Builds the collate_fn for BERT style masking. 15% of the non special
    tokens get picked, then 80/10/10 mask/random/keep. The vocab size, the
    mask id, and the special ids all get grabbed from the tokenizer here.
    Special tokens like [CLS] and [SEP] never get masked.
    """
    vocab_size = tokenizer.vocab_size
    mask_token_id = tokenizer.mask_token_id
    special_ids = set(tokenizer.all_special_ids)

    def collate_fn(batch):
        batch_input_ids = [s["input_ids"][:max_seq_len] for s in batch]
        batch_size = len(batch_input_ids)
        seq_lengths = [len(ids) for ids in batch_input_ids]
        max_len = max(1, min(max(seq_lengths), max_seq_len))

        input_ids_tensor = torch.full((batch_size, max_len), 0, dtype=torch.long)
        attention_mask_tensor = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, ids in enumerate(batch_input_ids):
            seq = ids[:max_len]
            input_ids_tensor[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask_tensor[i, : len(seq)] = 1

            # only real tokens can get masked, never the special ones
            candidates = [j for j, t in enumerate(seq) if t not in special_ids]
            if not candidates:
                continue

            num_mask = max(1, int(mlm_prob * len(candidates)))
            perm = torch.randperm(len(candidates))[:num_mask]

            for p in perm:
                idx = candidates[int(p)]
                original_token = input_ids_tensor[i, idx].item()
                labels_tensor[i, idx] = original_token

                rand = torch.rand(1).item()
                if rand < 0.8:                       # 80%: [MASK]
                    input_ids_tensor[i, idx] = mask_token_id
                elif rand < 0.9:                     # 10%: random vocab token
                    input_ids_tensor[i, idx] = int(
                        torch.randint(0, vocab_size, (1,)).item()
                    )
                # else 10%: keep original

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels_tensor,
        }

    return collate_fn


def get_dataloader(dataset, tokenizer, batch_size: int = 8, num_workers: int = 0):
    """
    DataLoader over the streaming dataset with the MLM collator.
    num_workers stays 0 on purpose. with a streaming dataset every worker
    opens its own copy from the top and the data silently gets duplicated.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=make_mlm_collate(tokenizer),
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Architecture: fused-QKV attention with RoPE + XPos, SwiGLU FFN,
# PaLM-style parallel blocks
# ---------------------------------------------------------------------------
class SelfAttention(nn.Module):
    """
    Multi head self attention with XPos rotary embeddings. Uses
    FlashAttention-2 when it is installed and there is no padding mask,
    since the basic flash api does not take padding masks. Otherwise it
    falls back to torch scaled_dot_product_attention.
    """

    def __init__(self, embed_dim, num_heads, dropout_rate=0.1,
                 max_seq_len=MAX_SEQ_LEN, xpos_scale_base=512):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must divide num_heads"
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.proj_dropout = nn.Dropout(dropout_rate)

        # precomputed RoPE tables
        theta = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float()
                                 / self.head_dim))
        pos = torch.arange(max_seq_len).float()[:, None]
        angles = pos * theta                                  # [L, D/2]
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

        # XPos decay
        scale = (torch.arange(0, self.head_dim, 2).float()
                 + 0.4 * self.head_dim) / (1.4 * self.head_dim)
        self.register_buffer("xpos_scale", scale, persistent=False)
        self.scale_base = xpos_scale_base

    def _apply_rope_xpos(self, q, k):
        # q, k: [B, H, L, D]
        L = q.shape[2]
        cos = self.cos[:L, :].to(q)
        sin = self.sin[:L, :].to(q)

        pos = torch.arange(L, device=q.device)[:, None]
        scale = (self.xpos_scale.to(q.device) ** (pos / self.scale_base)).to(q)
        scale = scale.unsqueeze(0).unsqueeze(0)               # [1,1,L,D/2]
        inv = 1.0 / scale

        def rotate(t, c, s):
            t_even, t_odd = t[..., 0::2], t[..., 1::2]
            rot_even = t_even * c - t_odd * s
            rot_odd = t_even * s + t_odd * c
            return torch.stack((rot_even, rot_odd), dim=-1).reshape_as(t)

        q = rotate(q, cos, sin) * scale
        k = rotate(k, cos, sin) * inv
        return q, k

    def forward(self, x, attention_mask=None):
        """
        x is [B, L, E]. attention_mask is [B, L] with 1 for valid tokens.
        """
        B, L, _ = x.size()
        qkv = self.qkv_proj(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                           # each [B, L, H, D]
        q, k = self._apply_rope_xpos(q.transpose(1, 2), k.transpose(1, 2))
        v = v.transpose(1, 2)                                 # [B, H, L, D]

        use_flash = _flash_available and attention_mask is None
        if use_flash:
            # FlashAttention-2 wants [B, L, H, D] fp16/bf16 tensors
            context = flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                softmax_scale=None,
                causal=False,
            )                                                 # [B, L, H, D]
        else:
            # SDPA bool masks work as True = this position can be attended to
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.to(torch.bool).view(B, 1, 1, L)
            context = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )                                                 # [B, H, L, D]
            context = context.transpose(1, 2)                 # [B, L, H, D]

        context = context.reshape(B, L, self.embed_dim)
        return self.proj_dropout(self.out_proj(context))


class FeedForward(nn.Module):
    """SwiGLU feed forward. The input projection is fused so the value and
    gate branches come out of one matmul."""

    def __init__(self, embed_dim: int, ff_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.fc_in = nn.Linear(embed_dim, ff_dim * 2)
        self.fc_out = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        value, gate = self.fc_in(x).chunk(2, dim=-1)
        x = value * F.silu(gate)                              # SwiGLU
        return self.dropout(self.fc_out(x))


class ParallelTransformerBlock(nn.Module):
    """PaLM style block. Attention and the FFN both read the same pre norm
    input and their outputs get summed into one residual."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads, dropout_rate)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, attention_mask=None):
        x_norm = self.ln(x)
        merged = self.attn(x_norm, attention_mask) + self.ffn(x_norm)
        return x + self.dropout(merged)


class TransformerModel(nn.Module):
    """
    The BERT++ encoder. BERT-Large depth and width with parallel blocks,
    SwiGLU feed forwards, XPos rotary embeddings, and a tied weight MLM
    head. The defaults are the real thing, 24 layers by 16 heads by 1024
    dim. ff_dim=2816 follows the SwiGLU 8/3 sizing so the tied model lands
    at about 340.9M parameters.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        ff_dim: int = 2816,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList(
            ParallelTransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        )

        # BERT style MLM head, the decoder is tied to the input embedding
        self.mlm_dense = nn.Linear(embed_dim, embed_dim)
        self.mlm_act = nn.GELU()
        self.mlm_ln = nn.LayerNorm(embed_dim)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))

        # train.py flips this on. non reentrant so it works with FSDP
        self.use_checkpoint = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.emb_dropout(self.token_emb(input_ids))       # [B, L, E]

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, attention_mask, use_reentrant=False)
            else:
                x = layer(x, attention_mask=attention_mask)

        h = self.mlm_ln(self.mlm_act(self.mlm_dense(x)))
        logits = torch.matmul(h, self.token_emb.weight.T) + self.mlm_bias

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return loss, logits
        return logits


def count_parameters(model: nn.Module) -> int:
    """Counts unique trainable parameters, tied weights only once."""
    seen, total = set(), 0
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


if __name__ == "__main__":
    tok = get_tokenizer()
    model = TransformerModel(vocab_size=tok.vocab_size)
    print(f"BERT++ built: {count_parameters(model)/1e6:.1f}M parameters")
