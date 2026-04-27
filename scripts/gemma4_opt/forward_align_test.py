#!/usr/bin/env python3
"""Forward-pass alignment test for gemma4-26B-A4B-it.

Goal: verify framework numerical equivalence by comparing logits / loss
on identical input + identical (init) weights between:
  - Single-GPU baseline (deterministic SDPA)        -- "ground truth"
  - FSDP2 distributed                                -- gather + compare
  - DS ZeRO-3 distributed                            -- gather + compare

This script is launched in 3 modes:
  python forward_align_test.py --mode single       # 1 GPU, no dist
  torchrun --nproc_per_node 8 forward_align_test.py --mode fsdp2
  torchrun --nproc_per_node 8 forward_align_test.py --mode ds

Output: <out_dir>/<mode>_run<N>.npz  with keys {logits_first10, logits_last10, loss}

Determinism settings:
  - torch.manual_seed + torch.cuda.manual_seed_all
  - torch.use_deterministic_algorithms(True, warn_only=True)
  - CUBLAS_WORKSPACE_CONFIG=:4096:8
  - attn_impl=sdpa (PyTorch native, deterministic backend)
  - eval mode (no dropout)

Expected (per HF docs + community):
  - bf16 single-GPU rerun:     |Δ| < 1e-3 relative (atomics noise)
  - FSDP2 vs single-GPU:       |Δ| < 5e-3 relative (DTensor reductions add noise)
  - DS vs single-GPU:          |Δ| < 5e-3 relative
  - FSDP2 vs DS:               |Δ| < 1e-2 relative (worst case, both deviating from gt)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "fsdp2", "ds"], required=True)
    p.add_argument("--model-path", default="/home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="short seq to fit single-gpu baseline; 2048 is plenty for fwd alignment")
    p.add_argument("--out-dir", default="/home/ubuntu/fyh/megatron_output/gemma4_opt/p0_forward_align")
    p.add_argument("--run-id", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_input(seq_len, vocab_size, seed, device):
    """Fixed input given seed -- input_ids will be byte-identical across runs."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(0, min(vocab_size, 50000), (1, seq_len), generator=g)
    labels = input_ids.clone()
    position_ids = torch.arange(seq_len).unsqueeze(0)
    return (input_ids.to(device), labels.to(device), position_ids.to(device))


def run_single(args):
    """Single-GPU deterministic forward."""
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    from transformers import AutoModelForImageTextToText
    print(f"[single] loading model from {args.model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    print(f"[single] model loaded; {sum(p.numel() for p in model.parameters()) / 1e9:.2f} B params")

    text_cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    vocab = text_cfg.vocab_size
    input_ids, labels, position_ids = make_input(args.seq_len, vocab, args.seed, device)

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels, position_ids=position_ids)

    logits = out.logits.detach().float().cpu().numpy()  # cast for save
    loss = float(out.loss.item()) if out.loss is not None else None
    save_outputs(args, logits, loss, rank=0)


def run_distributed(args):
    """FSDP2 or DS distributed forward. Each rank loads model, gathers logits to rank0."""
    import torch.distributed as dist
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    dist.init_process_group(backend="nccl")

    from transformers import AutoModelForImageTextToText
    if rank == 0:
        print(f"[{args.mode}/{rank}] loading model from {args.model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()

    if args.mode == "fsdp2":
        # Wrap with FSDP2 (per-parameter)
        from torch.distributed.fsdp import fully_shard, FSDPModule
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh("cuda", (world_size,))
        # Wrap each Gemma4TextDecoderLayer
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
        for module in model.modules():
            if isinstance(module, Gemma4TextDecoderLayer):
                fully_shard(module, mesh=mesh)
        fully_shard(model, mesh=mesh)
        if rank == 0:
            print(f"[fsdp2] wrapped with FSDP2 mesh={mesh}")

    elif args.mode == "ds":
        # Initialize DeepSpeed inference engine (no opt, no offload — simpler)
        import deepspeed
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "bf16": {"enabled": True},
            "zero_optimization": {"stage": 3},
        }
        # Snapshot a sample weight before DS init for debug
        sample_weight_before = None
        for name, p in model.named_parameters():
            if "lm_head" in name:
                sample_weight_before = (name, p.detach().float().mean().item(),
                                         p.detach().float().std().item(), p.dtype)
                break
        model_engine, _, _, _ = deepspeed.initialize(
            model=model, model_parameters=model.parameters(), config=ds_config,
        )
        model = model_engine
        if rank == 0:
            print(f"[ds] initialized with ZeRO-3")
            # Snapshot same weight after DS init
            inner = model.module if hasattr(model, "module") else model
            for name, p in inner.named_parameters():
                if "lm_head" in name:
                    # Need to gather sharded param
                    if hasattr(p, "ds_id"):
                        from deepspeed.runtime.zero.partition_parameters import GatheredParameters
                        with GatheredParameters([p], modifier_rank=0):
                            print(f"[ds] BEFORE: {sample_weight_before}")
                            print(f"[ds] AFTER : ({name}, mean={p.detach().float().mean().item()}, "
                                  f"std={p.detach().float().std().item()}, dtype={p.dtype})")
                    else:
                        print(f"[ds] BEFORE: {sample_weight_before}")
                        print(f"[ds] AFTER : ({name}, mean={p.detach().float().mean().item()}, "
                              f"std={p.detach().float().std().item()}, dtype={p.dtype})")
                    break

    text_cfg = (model.module if hasattr(model, "module") else model).config
    text_cfg = text_cfg.text_config if hasattr(text_cfg, "text_config") else text_cfg
    vocab = text_cfg.vocab_size
    input_ids, labels, position_ids = make_input(args.seq_len, vocab, args.seed, device)

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels, position_ids=position_ids)

    # logits should be on every rank (replicated across DP). Just rank 0 saves.
    if rank == 0:
        logits = out.logits.detach().float().cpu().numpy()
        loss = float(out.loss.item()) if out.loss is not None else None
        save_outputs(args, logits, loss, rank=0)

    dist.barrier()
    dist.destroy_process_group()


def save_outputs(args, logits, loss, rank):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.mode}_run{args.run_id}.npz"
    # Save just slices to avoid huge files; full 1×2048×262144 = 1 GB
    summary = {
        "logits_first10": logits[0, :10, :].astype(np.float32),  # first 10 positions
        "logits_last10": logits[0, -10:, :].astype(np.float32),  # last 10 positions
        "logits_mean": logits.mean(axis=(0, 1)).astype(np.float32),  # vocab-mean per-token
        "logits_std": logits.std(axis=(0, 1)).astype(np.float32),
        "loss": np.array([loss if loss is not None else float("nan")], dtype=np.float64),
        "config": np.array([
            f"mode={args.mode}",
            f"seq_len={args.seq_len}",
            f"seed={args.seed}",
            f"run_id={args.run_id}",
        ]),
    }
    np.savez(out_file, **summary)
    print(f"[{args.mode}/{rank}] saved {out_file}  loss={loss}  logits.shape={logits.shape}")


def main():
    args = parse_args()
    if args.mode == "single":
        run_single(args)
    else:
        run_distributed(args)


if __name__ == "__main__":
    main()
