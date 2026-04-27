#!/usr/bin/env python3
"""Minimal FSDP2 + Context Parallel trainer for gemma-4-26B-A4B-it.

Skips swift entirely because swift's Ulysses SP has GQA shape mismatch bugs
on gemma4's `num_global_key_value_heads=2`. Uses HF accelerate's native
`ParallelismConfig(cp_size=N, dp_shard_size=M)` which is wired to
`torch.distributed.tensor.experimental.context_parallel` (accelerate >= 1.13).

Reference: HF accelerate `examples/torch_native_parallelism/nd_parallel.py`.

Constraints (per accelerate docs):
- SDPA only (FA incompatible with gemma4 global_head_dim=512 anyway)
- No mask or causal mask — we train plain causal LM, OK
- `shift_labels` must be passed; model uses those (CP doesn't handle label shift)
- Dataset uses random tokens for bench purposes (measuring throughput, not
  convergence). Real training should wire the sft-data jsonl.

Usage:
    accelerate launch --num-processes 8 --mixed-precision bf16 \\
        --use-fsdp --fsdp-version 2 \\
        scripts/gemma4_opt/train_fsdp2_cp.py \\
            --cp-size 2 --dp-shard-size 4 --num-steps 40 --seq-len 16384
"""
from __future__ import annotations
import argparse
import json
import os
import time
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from accelerate.parallelism_config import ParallelismConfig
from accelerate.utils import FullyShardedDataParallelPlugin, set_seed
from accelerate.utils.dataclasses import TorchContextParallelConfig

from transformers import AutoModelForImageTextToText


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it")
    p.add_argument("--dp-replicate-size", type=int, default=1)
    p.add_argument("--dp-shard-size", type=int, default=4,
                   help="FSDP2 shard dim; with 8 ranks + cp=2 -> dp-shard=4")
    p.add_argument("--cp-size", type=int, default=2,
                   help="Context parallel (sequence shard). Replaces swift Ulysses SP.")
    p.add_argument("--cp-comm", default="allgather", choices=["allgather", "alltoall"])
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--mbs", type=int, default=1, help="per-device micro batch size")
    p.add_argument("--num-steps", type=int, default=40)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--activation-checkpointing", action="store_true", default=True)
    p.add_argument("--no-activation-checkpointing", dest="activation_checkpointing", action="store_false")
    p.add_argument("--output-dir", default="/home/ubuntu/fyh/megatron_output/gemma4_opt/p0_baseline_fsdp2_cp/run")
    p.add_argument("--metrics-out", default=None, help="Path to dump per-step metrics JSONL")
    return p.parse_args()


class RandomLMDataset(Dataset):
    """Random causal-LM tokens. Good enough for throughput bench."""
    def __init__(self, n_samples: int, seq_len: int, vocab_size: int):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.rng = torch.Generator().manual_seed(42)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        ids = torch.randint(0, self.vocab_size, (self.seq_len,), generator=self.rng)
        return {"input_ids": ids}


def collate_fn(batch):
    ids = torch.stack([b["input_ids"] for b in batch], dim=0)  # [B, S]
    labels = ids.clone()
    # shift_labels: labels[t+1] at position t, last position = -100 (ignore)
    shift = torch.full_like(labels, -100)
    shift[:, :-1] = labels[:, 1:]
    position_ids = torch.arange(ids.size(1)).unsqueeze(0).expand(ids.size(0), -1)
    return {
        "input_ids": ids,
        "labels": labels,
        "shift_labels": shift,
        "position_ids": position_ids,
    }


def freeze_vision(model):
    """Freeze visual tower + aligner (matches --freeze_vit true --freeze_aligner true)."""
    n_frozen = 0
    for name, p in model.named_parameters():
        if (".visual." in name or "multi_modal_projector" in name or
                "vision_tower" in name or ".embedder." in name):
            p.requires_grad = False
            n_frozen += 1
    return n_frozen


def main():
    args = parse_args()
    set_seed(42)

    # Build parallelism config
    cp_handler = TorchContextParallelConfig(cp_comm_strategy=args.cp_comm) if args.cp_size > 1 else None
    parallelism_config = ParallelismConfig(
        dp_replicate_size=args.dp_replicate_size,
        dp_shard_size=args.dp_shard_size,
        tp_size=args.tp_size,
        cp_size=args.cp_size,
        cp_handler=cp_handler,
    )

    # FSDP2 plugin: wrap by gemma4 decoder layer class; skip Gemma4AudioLayer
    # which doesn't exist on 26B A4B (only E2B/E4B have audio).
    fsdp2_plugin = None
    if parallelism_config.dp_shard_enabled or parallelism_config.cp_enabled:
        fsdp2_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer"],
            state_dict_type="SHARDED_STATE_DICT",
            cpu_ram_efficient_loading=False,  # avoid mixed DTensor/Tensor bug
            activation_checkpointing=args.activation_checkpointing,
        )

    accelerator = Accelerator(
        mixed_precision="bf16",
        parallelism_config=parallelism_config,
        fsdp_plugin=fsdp2_plugin,
    )

    accelerator.print(f"ParallelismConfig: {parallelism_config}")
    accelerator.print(f"FSDP2 plugin activation_checkpointing={args.activation_checkpointing}, "
                      f"cpu_ram_efficient_loading=False")

    # Load gemma4 with SDPA (FA doesn't support head_dim=512 on gemma4 globals)
    accelerator.print(f"Loading model from {args.model_path} with sdpa + bf16 ...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_cache=False,
    )
    n_frozen = freeze_vision(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Frozen {n_frozen} vision/aligner params; trainable={trainable/1e9:.2f}B / total={total/1e9:.2f}B")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5, betas=(0.9, 0.95), weight_decay=0.1,
    )

    # Dataset: random tokens (text-only, no pixel_values)
    tc = model.config.text_config if hasattr(model.config, "text_config") else model.config
    vocab_size = tc.vocab_size
    dataset = RandomLMDataset(n_samples=args.num_steps * args.mbs * 10,  # plenty
                              seq_len=args.seq_len, vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=args.mbs, collate_fn=collate_fn, num_workers=0)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    accelerator.print("Starting training loop ...")
    metrics_path = Path(args.metrics_out) if args.metrics_out else Path(args.output_dir) / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    step_times = []
    t_prev = None
    peak_mem_gb = 0.0
    loss_reduce_grp = (
        accelerator.torch_device_mesh["dp_cp"].get_group()
        if accelerator.parallelism_config.dp_cp_dim_names else None
    )

    torch.cuda.reset_peak_memory_stats()

    for step, batch in enumerate(dataloader):
        if step >= args.num_steps:
            break

        buffers = [batch["input_ids"], batch["shift_labels"], batch["labels"], batch["position_ids"]]
        dist.barrier()
        t0 = time.time()
        with accelerator.maybe_context_parallel(
                buffers=buffers, buffer_seq_dims=[1, 1, 1, 1], no_restore_buffers=set(buffers)):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if loss_reduce_grp is not None:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG, group=loss_reduce_grp)
        dist.barrier()
        t1 = time.time()
        step_time = t1 - t0
        step_times.append(step_time)

        cur_peak_mem_bytes = torch.cuda.max_memory_allocated()
        cur_peak_mem_gb = cur_peak_mem_bytes / (1024 ** 3)
        peak_mem_gb = max(peak_mem_gb, cur_peak_mem_gb)

        rec = {
            "step": step,
            "loss": float(loss.item()),
            "step_time_s": step_time,
            "peak_mem_gb": cur_peak_mem_gb,
        }
        if accelerator.is_main_process:
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        accelerator.print(f"step {step:3d}  loss {loss.item():.4f}  step_time {step_time:.3f}s  "
                          f"peak_mem {cur_peak_mem_gb:.2f} GiB")

    # Summary: steady-state after warmup
    if len(step_times) > args.warmup_steps + 5:
        steady = step_times[args.warmup_steps:]
        steady_mean = sum(steady) / len(steady)
        accelerator.print(f"\n=== summary ===")
        accelerator.print(f"steady_step_s (mean after {args.warmup_steps} warmup): {steady_mean:.3f}")
        accelerator.print(f"peak_mem_gb: {peak_mem_gb:.2f}")
        accelerator.print(f"tokens_per_step: {args.mbs * args.seq_len * parallelism_config.dp_size}")
        if accelerator.is_main_process:
            with open(metrics_path.parent / "summary.json", "w") as f:
                json.dump({
                    "steady_step_s": steady_mean,
                    "peak_mem_gb": peak_mem_gb,
                    "config": vars(args),
                    "tokens_per_step": args.mbs * args.seq_len * parallelism_config.dp_size,
                }, f, indent=2)

    accelerator.wait_for_everyone()
    accelerator.print("Done.")


if __name__ == "__main__":
    main()
