# BERT++: BERT-Large-Scale Pretraining on Constrained Hardware

**A 341M-parameter encoder — BERT-Large depth and width with PaLM-style parallel blocks, SwiGLU feed-forwards, and XPos rotary embeddings — trained under a tight memory budget: FSDP parameter sharding, activation checkpointing, mixed precision, and a FlashAttention-ready attention path, over streamed C4 + The Pile.**

---

## The problem

A BERT-Large-scale encoder (24 layers, 1024 hidden) is ~340M parameters. A naive training loop replicates optimizer state on every GPU and holds every activation for the backward pass, which puts a model that size out of reach of modest hardware. This project is a systems exercise: **how far can memory-optimization techniques stretch limited GPUs before you need a cluster?**

This is not a BERT reimplementation. The encoder uses PaLM-style parallel attention/FFN blocks, SwiGLU feed-forwards, and XPos rotary position embeddings, trained with a hand-written MLM objective and a BERT-style transform head tied to the input embedding.

## Architecture

```
Streamed C4 ⟷ The Pile (50/50 interleave, on-the-fly WordPiece tokenization)
                              │
                 hand-written MLM collator
              (15% / 80-10-10, specials excluded)
                              │
                    ┌─────────▼─────────┐
                    │  Encoder          │  24 layers · 16 heads · 1024 dim
                    │  parallel blocks  │  SwiGLU ffn 2816 (≈8/3·d convention)
                    │  fused QKV + XPos │  340,925,242 params (tied, counted once)
                    └─────────┬─────────┘
                              │
              BERT-style MLM head, decoder tied to embedding
                              │
        FSDP (FULL_SHARD, per-block wrap) · activation ckpt · bf16/fp16 AMP
```

| Technique | What it does here | Why it matters |
|---|---|---|
| **FSDP** with `transformer_auto_wrap_policy` | Shards params, grads, and optimizer state; gathers/releases **per block** | The per-layer cycle is the memory win — wrapping the whole model as one flat group (the naive call) forfeits it |
| **Activation checkpointing** (`use_reentrant=False`) | Recomputes activations in backward | Trades compute for memory; the non-reentrant variant composes cleanly with FSDP |
| **Mixed precision** (`torch.amp`) | bf16 where supported, fp16 + GradScaler otherwise | Halves activation memory, engages tensor cores |
| **FlashAttention-2 (maskless path)** | IO-aware attention kernels when no padding mask is present | Caveat: MLM batches nearly always carry padding masks, so the SDPA path does the day-to-day work; the flash path exists for packed/fixed-length regimes |
| **Streaming data** | `interleave_datasets` over C4 + uncopyrighted Pile, sharded per rank via `split_dataset_by_node` | No corpus materialized on disk |
| **Warmup + decay, gradient clipping** | 10k-step linear warmup → linear decay; clip at 1.0 | Cold-starting 341M params at 1e-4 in fp16 without either is a loss-spike recipe |

## Status — read before citing numbers

**No throughput or memory benchmark has been run yet.** The write-up in `docs/` cites *expected* gains from published work (≈1.5× step-time from AMP; ~15% further from FlashAttention on BERT-Large at 512 tokens, Dao et al.). Those are citations, not measurements from this repo. The benchmark plan (four configs × 1/3 GPUs, step time + peak memory) is spelled out in the notebook; results will land here when measured.

What *is* verified today, by running the notebook top to bottom on CPU: the model builds at the stated size (the parameter count is printed, 340,925,242), a padded batch flows through attention with the mask honored, the MLM loss at random init sits at ~ln(vocab) as it should, and the collator's masking invariants hold (≈15% rate, specials never masked, labels carry original ids).

## Repository layout

```
├── src/
│   ├── setup_bertpp.py   # single source of truth: tokenizer, streaming data,
│   │                     #   MLM collator, model (SelfAttention/FFN/blocks)
│   └── train.py          # FSDP entry point: torchrun --nproc_per_node=N src/train.py
├── notebooks/
│   └── bert_pp_fsdp_training.ipynb   # driver: imports src, smoke-tests collator + model on CPU
├── docs/                 # write-up (see PLACE_FILES_HERE.md)
└── requirements.txt
```

All model and data code lives in `src/`; the notebook imports what it trains, so the smoke tests exercise exactly the code `torchrun` runs.

## Running it

```bash
git clone https://github.com/Drew-Patrick/bert-plus-plus.git
cd bert-plus-plus
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Smoke-test without a GPU** — open `notebooks/bert_pp_fsdp_training.ipynb` and run it: the collator demo and the 341M-parameter forward/loss check run offline on CPU in seconds.

**Train:**

```bash
wandb login          # or export WANDB_API_KEY=...
torchrun --nproc_per_node=3 src/train.py
```

FlashAttention-2 is optional and CUDA-toolchain-specific: `pip install flash-attn --no-build-isolation`. The code falls back to `scaled_dot_product_attention` automatically.

## Next steps

1. **Run and publish the benchmark** — baseline / +AMP / +checkpointing / +FlashAttention across 1 and 3 GPUs; chart to `docs/`, numbers to this Status section.
2. Add resumable optimizer state to checkpoints (`FSDP.optim_state_dict`) — current checkpoints are model-only.
3. Downstream GLUE evaluation to confirm the memory optimizations don't cost quality.
4. Compare `FULL_SHARD` vs `SHARD_GRAD_OP` at this model size.

## Reference

Dao, T., et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.

## Author

**Drew Patrick** — M.S. Artificial Intelligence, Kennesaw State University.

## License

MIT — see [LICENSE](LICENSE).
