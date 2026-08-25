"""
train.py

FSDP training entry point for BERT++. Launch with torchrun:

    torchrun --nproc_per_node=N src/train.py

How it is set up:
  * FSDP wraps each ParallelTransformerBlock through
    transformer_auto_wrap_policy with FULL_SHARD. The per layer gather and
    release is where the memory savings actually come from.
  * bf16 autocast when the GPU supports it, which needs no GradScaler.
    Otherwise fp16 with a GradScaler.
  * 10k steps of linear warmup then linear decay, gradient clipping at 1.0.
  * Each rank gets its own slice of the stream through
    datasets.distributed.split_dataset_by_node.
  * Checkpoints are the full model state dict gathered on rank 0 and saved
    locally. Optimizer state is not saved yet, that is on the todo list.
"""

import functools
import os

import torch
import torch.distributed as dist
import wandb
from datasets.distributed import split_dataset_by_node
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from setup_bertpp import (
    ParallelTransformerBlock,
    TransformerModel,
    count_parameters,
    get_dataloader,
    get_datasets,
    get_tokenizer,
)

# ----------------------------- configuration -------------------------------
MAX_STEPS = 1_000_000
WARMUP_STEPS = 10_000
LR = 1e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 8            # per GPU
GRAD_CLIP = 1.0
LOG_INTERVAL = 50
CKPT_INTERVAL = 10_000
CKPT_DIR = "checkpoints"
EMA_DECAY = 0.99


def lr_lambda(step: int) -> float:
    """Linear warmup to LR over WARMUP_STEPS, then linear decay to zero."""
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    return max(0.0, (MAX_STEPS - step) / max(1, MAX_STEPS - WARMUP_STEPS))


def main() -> None:
    # ------------------------- distributed setup ---------------------------
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # bf16 does not need loss scaling, fp16 does
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    # ------------------------------- data ----------------------------------
    tokenizer = get_tokenizer()
    dataset = get_datasets(tokenizer)
    if distributed:
        # the supported way to shard a streaming dataset across ranks
        dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    dataloader = get_dataloader(dataset, tokenizer, batch_size=BATCH_SIZE)

    # ------------------------------- model ---------------------------------
    model = TransformerModel(vocab_size=tokenizer.vocab_size)
    model.use_checkpoint = True
    n_params = count_parameters(model)
    model.to(device)

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={ParallelTransformerBlock},
    )
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
    )

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = LambdaLR(optimizer, lr_lambda)

    # ------------------------------ logging --------------------------------
    if rank == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)
        wandb.init(
            project="bertpp_pretrain",
            name="bertpp_fsdp",
            config=dict(
                max_steps=MAX_STEPS,
                warmup_steps=WARMUP_STEPS,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                per_gpu_batch_size=BATCH_SIZE,
                world_size=world_size,
                amp_dtype=str(amp_dtype),
                grad_clip=GRAD_CLIP,
                parameters=n_params,
                model="BERT++ (24L/16H/1024d, SwiGLU 2816, XPos)",
            ),
        )
        print(f"BERT++: {n_params/1e6:.1f}M parameters | "
              f"world_size={world_size} | amp={amp_dtype}")
    else:
        wandb.init(mode="disabled")

    # ---------------------------- training loop ----------------------------
    model.train()
    data_iter = iter(dataloader)
    ema_loss = None

    for step in range(1, MAX_STEPS + 1):
        try:
            batch = next(data_iter)
        except StopIteration:            # stream exhausted -> restart it
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            loss, _ = model(
                input_ids, attention_mask=attention_mask, labels=labels
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_val = loss.item()
        ema_loss = loss_val if ema_loss is None else (
            ema_loss * EMA_DECAY + loss_val * (1 - EMA_DECAY)
        )

        if step % LOG_INTERVAL == 0 and rank == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"[{step:>7}/{MAX_STEPS}] loss={loss_val:.4f} "
                  f"ema={ema_loss:.4f} lr={lr_now:.2e}")
            wandb.log(
                {"loss": loss_val, "loss_ema": ema_loss, "lr": lr_now},
                step=step,
            )

        if step % CKPT_INTERVAL == 0 or step == MAX_STEPS:
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = model.state_dict()
            if rank == 0:
                path = os.path.join(CKPT_DIR, f"checkpoint_step_{step}.pt")
                torch.save({"step": step, "model_state": state_dict}, path)
                wandb.log({"checkpoint_step": step}, step=step)
                print(f"saved checkpoint -> {path}")

    if distributed:
        dist.barrier()
    if rank == 0:
        wandb.finish()
    print(f"Training completed on rank {rank}.")


if __name__ == "__main__":
    main()
