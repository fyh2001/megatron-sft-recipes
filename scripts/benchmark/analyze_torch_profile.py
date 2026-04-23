#!/usr/bin/env python3
"""Aggregate one rank's torch.profiler Chrome-trace JSON into kernel buckets.

Reads `trace_rank*_step*.json` and groups CUDA kernel events by category
(GEMM / FlashAttn / GDN scan / NCCL / Elementwise-Norm-Softmax / Other),
reporting per-category wall-time sum, share of GPU time, and top kernels.

Usage:
    python analyze_torch_profile.py /path/to/trace_rank0_stepN.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def classify(name: str) -> str:
    n = name.lower()
    # NCCL comms
    if "nccldevkernel" in n or "ncclkernel" in n or "allreduce" in n \
            or "reduce_scatter" in n or "allgather" in n or "all_gather" in n \
            or "alltoall" in n or "all_to_all" in n:
        return "NCCL"
    # Optimizer (Adam / foreach fused) — check BEFORE elementwise pattern
    if "fusedoptimizer" in n or "adam_" in n or "fused_adam" in n or "fused_sgd" in n \
            or "_foreach_" in n or "lamb_" in n:
        return "Optimizer"
    # CUTLASS / cuBLAS / cuBLASLt JIT (nvjet_*) GEMM kernels — tensor-core matmul
    if any(t in n for t in ("cutlass", "gemm", "cublas", "hmma", "wgmma", "mma_",
                            "sm90_xmma", "sm90_gemm", "sm_90a_gemm", "nvjet_",
                            "ampere_gemm", "volta_gemm", "s884gemm", "s1688gemm",
                            "h884gemm", "h1688gemm", "f16_s16")):
        return "GEMM"
    # Flash attention (multiple impls)
    if ("flash" in n and ("attn" in n or "attention" in n or "fwd" in n or "bwd" in n)) \
            or "flash_attn" in n or "fmha" in n:
        return "FlashAttn"
    # Triton-compiled fused kernels (GDN / fla / user kernels)
    if "triton" in n or "fla_" in n or "gated_delta" in n or "chunk_delta" in n \
            or "chunk_gla" in n or "recurrent_" in n or n.startswith("_kernel_"):
        return "TritonFused"
    # Reductions (sum, mean, norm)
    if "reduce_kernel" in n or "norm_kernel" in n or "softmax" in n or "logsoftmax" in n:
        return "Reduction"
    # Memory / shuffling (SP all-to-all buffers, cat, split, etc.)
    if any(t in n for t in ("memset", "memcpy", "permute", "transpose", "contiguous",
                            "split_with_sizes", "chunk_cat", "gather", "scatter",
                            "index_", "copy_kernel")):
        return "Memory"
    # Elementwise / activations
    if any(t in n for t in ("elementwise", "vectorized", "silu", "gelu", "dropout",
                            "binaryfunctor", "unaryfunctor", "where", "masked",
                            "clamp", "cast")):
        return "Elementwise"
    return "Other"


def analyze(trace_path: Path) -> dict:
    with open(trace_path) as f:
        data = json.load(f)
    events = data.get("traceEvents", data if isinstance(data, list) else [])

    cat_dur: dict[str, float] = defaultdict(float)
    cat_count: dict[str, int] = defaultdict(int)
    top_kernels: dict[str, float] = defaultdict(float)

    gpu_total_us = 0.0
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        if "kernel" not in cat and "gpu_memcpy" not in cat and "gpu_memset" not in cat:
            continue
        name = ev.get("name", "")
        dur = float(ev.get("dur", 0))
        bucket = classify(name)
        cat_dur[bucket] += dur
        cat_count[bucket] += 1
        top_kernels[name] += dur
        gpu_total_us += dur

    result = {
        "trace": str(trace_path),
        "total_gpu_us": gpu_total_us,
        "total_gpu_ms": gpu_total_us / 1000,
        "by_category": [],
        "top_kernels": [],
    }
    for cat, dur in sorted(cat_dur.items(), key=lambda x: -x[1]):
        result["by_category"].append({
            "category": cat,
            "ms": dur / 1000,
            "pct": (dur / gpu_total_us * 100) if gpu_total_us else 0,
            "kernels": cat_count[cat],
        })
    for name, dur in sorted(top_kernels.items(), key=lambda x: -x[1])[:20]:
        result["top_kernels"].append({
            "name": name[:140],
            "ms": dur / 1000,
            "pct": (dur / gpu_total_us * 100) if gpu_total_us else 0,
        })
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    r = analyze(args.trace)

    if args.json:
        print(json.dumps(r, indent=2))
        return 0

    print(f"Trace: {r['trace']}")
    print(f"Total GPU kernel time: {r['total_gpu_ms']:.1f} ms")
    print()
    print(f"{'Category':<16}{'ms':>10}{'%GPU':>8}{'#kernels':>12}")
    for row in r["by_category"]:
        print(f"{row['category']:<16}{row['ms']:>10.1f}{row['pct']:>7.1f}%{row['kernels']:>12}")
    print()
    print("Top 20 kernels by CUDA time:")
    for row in r["top_kernels"]:
        print(f"  {row['ms']:>8.1f} ms  {row['pct']:>5.1f}%  {row['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
