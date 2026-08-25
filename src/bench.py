"""
bench.py

Single GPU benchmark for the four configurations the README promises:
fp32 baseline, then AMP, then AMP with activation checkpointing, then
AMP with checkpointing and FlashAttention on the maskless path.

Run it from the repo root on a GPU box or Colab:

    python src/bench.py

It picks the biggest batch that fits the fp32 baseline and uses that same
batch for every config so the step times are comparable. It also probes
the max batch each config can hold, since raising the ceiling is half the
point of these techniques. Results print as a markdown table and get
saved to docs/benchmark.json.

This is a single device benchmark. FSDP multi GPU scaling is a separate
measurement and is not covered here.
"""

import json
import os
import statistics
import sys
import time
from contextlib import nullcontext

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup_bertpp import TransformerModel, get_tokenizer, _flash_available

SEQ_LEN = 512
WARMUP_STEPS = 5
TIMED_STEPS = 20
CANDIDATE_BATCHES = [16, 12, 8, 6, 4, 2, 1]


def make_batch(batch_size, vocab_size, device, use_mask):
    torch.manual_seed(0)
    ids = torch.randint(0, vocab_size, (batch_size, SEQ_LEN), device=device)
    labels = torch.full_like(ids, -100)
    picked = torch.rand(ids.shape, device=device) < 0.15
    labels[picked] = ids[picked]
    mask = torch.ones_like(ids) if use_mask else None
    return ids, mask, labels


def one_step(model, optimizer, scaler, amp_dtype, ids, mask, labels):
    optimizer.zero_grad(set_to_none=True)
    ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
           if amp_dtype is not None else nullcontext())
    with ctx:
        loss, _ = model(ids, attention_mask=mask, labels=labels)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return loss


def try_step(model, batch_size, vocab_size, device, amp_dtype, use_mask):
    # one forward and backward at this batch size, True if it fits
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = (torch.amp.GradScaler("cuda")
                  if amp_dtype == torch.float16 else None)
        ids, mask, labels = make_batch(batch_size, vocab_size, device, use_mask)
        one_step(model, optimizer, scaler, amp_dtype, ids, mask, labels)
        torch.cuda.synchronize()
        del optimizer, scaler
        torch.cuda.empty_cache()
        return True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False


def bench_config(model, name, batch_size, vocab_size, device, amp_dtype, use_ckpt, use_mask):
    model.use_checkpoint = use_ckpt
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    ids, mask, labels = make_batch(batch_size, vocab_size, device, use_mask)

    for _ in range(WARMUP_STEPS):
        one_step(model, optimizer, scaler, amp_dtype, ids, mask, labels)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(TIMED_STEPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_step(model, optimizer, scaler, amp_dtype, ids, mask, labels)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    med = statistics.median(times)
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    del optimizer, scaler
    torch.cuda.empty_cache()
    return {
        "config": name,
        "batch": batch_size,
        "median_step_s": round(med, 4),
        "tokens_per_s": int(batch_size * SEQ_LEN / med),
        "peak_mem_gb": round(peak_gb, 2),
    }


def main():
    if not torch.cuda.is_available():
        print("needs a GPU, nothing to measure on CPU")
        return
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16

    tokenizer = get_tokenizer()
    model = TransformerModel(vocab_size=tokenizer.vocab_size).to(device)

    # amp_dtype None means plain fp32. use_mask False routes the flash path.
    configs = [
        ("fp32 baseline",        None,      False, True),
        ("+ AMP",                amp_dtype, False, True),
        ("+ AMP + ckpt",         amp_dtype, True,  True),
    ]
    if _flash_available:
        configs.append(("+ AMP + ckpt + flash", amp_dtype, True, False))
    else:
        print("flash_attn not installed, skipping the flash config")

    # biggest batch the fp32 baseline can hold, shared by every config
    model.use_checkpoint = False
    shared_batch = None
    for b in CANDIDATE_BATCHES:
        if try_step(model, b, tokenizer.vocab_size, device, None, True):
            shared_batch = b
            break
    if shared_batch is None:
        print("even batch 1 does not fit in fp32 on this GPU")
        return

    print(f"GPU: {gpu} | amp dtype: {amp_dtype} | shared batch: {shared_batch}\n")

    results = []
    for name, dtype, ckpt, mask in configs:
        results.append(bench_config(model, name, shared_batch,
                                    tokenizer.vocab_size, device, dtype, ckpt, mask))
        r = results[-1]
        print(f"done: {r['config']:<22} {r['median_step_s']}s/step  "
              f"{r['tokens_per_s']} tok/s  peak {r['peak_mem_gb']} GB")

    # ceiling probe, the other half of the story
    print("\nmax batch that fits per config:")
    ceilings = {}
    for name, dtype, ckpt, mask in configs:
        model.use_checkpoint = ckpt
        fit = 0
        for b in CANDIDATE_BATCHES:
            if try_step(model, b, tokenizer.vocab_size, device, dtype, mask):
                fit = b
                break
        ceilings[name] = fit
        print(f"  {name:<22} batch {fit}")

    base = results[0]["median_step_s"]
    print("\n| Config | Batch | Median step (s) | Tokens/s | Peak mem (GB) | Speedup vs fp32 |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['config']} | {r['batch']} | {r['median_step_s']} | "
              f"{r['tokens_per_s']} | {r['peak_mem_gb']} | "
              f"{base / r['median_step_s']:.2f}x |")

    os.makedirs("docs", exist_ok=True)
    with open("docs/benchmark.json", "w") as fh:
        json.dump({"gpu": gpu, "amp_dtype": str(amp_dtype),
                   "shared_batch": shared_batch, "results": results,
                   "max_batch_per_config": ceilings}, fh, indent=2)
    print("\nsaved docs/benchmark.json")


if __name__ == "__main__":
    main()
