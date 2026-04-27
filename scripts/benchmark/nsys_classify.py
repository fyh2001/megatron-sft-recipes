#!/usr/bin/env python3
"""Aggregate nsys cuda_gpu_kern_sum CSV by kernel category and compute true MFU.

Inputs:
    csv_path  - output of `nsys stats --report cuda_gpu_kern_sum --format csv ...`
    --capture-window-s   - the nsys --duration value used (seconds). Required for
                           computing per-rank GEMM time / wall ratio.
    --num-ranks  - total ranks in the job (default 8). We profile rank 0 only,
                   so per-rank GEMM time = GEMM kernel time on this trace.
    --peak-tflops  - device peak BF16 TC TFLOPS (default 989 = H100 SXM)

True MFU upper bound = (GEMM_time_on_rank / capture_window_s)
                       (≤ 1.0 ; assumes GEMM ran at peak)

It is an UPPER bound because GEMM kernels rarely hit 989 TF in practice.
"""
from __future__ import annotations
import argparse, csv, sys
from collections import defaultdict


def classify(name: str) -> str:
    n = name.lower()
    if "ncclkernel" in n or "ncclDevKernel".lower() in n or "allreduce" in n \
            or "reduce_scatter" in n or "allgather" in n or "all_gather" in n \
            or "alltoall" in n or "all_to_all" in n or "sendrecv" in n:
        return "NCCL"
    if "fusedoptimizer" in n or "adam_" in n or "fused_adam" in n or "fused_sgd" in n \
            or "_foreach_" in n or "lamb_" in n:
        return "Optimizer"
    # FlashAttn FIRST — flash::flash_*_kernel uses cutlass::bfloat16_t in
    # template args which would otherwise collide with the GEMM rule below.
    if ("flash::flash_" in n) or "flash_attn" in n or "fmha" in n \
            or ("flash" in n and ("attn" in n or "attention" in n)):
        return "FlashAttn"
    if any(t in n for t in ("cutlass", "gemm", "cublas", "hmma", "wgmma", "mma_",
                            "sm90_xmma", "sm90_gemm", "sm_90a_gemm", "nvjet_",
                            "ampere_gemm", "volta_gemm", "s884gemm", "s1688gemm",
                            "h884gemm", "h1688gemm", "f16_s16")):
        return "GEMM"
    # FLA GDN scan + Liger Triton kernels
    if "chunk_" in n or "fused_recurrent" in n or "gated_delta" in n \
            or "_swiglu_" in n or "_rms_" in n or "_rope_" in n or "liger" in n \
            or "_kernel_" in n.split("(")[0]:
        return "TritonFused"
    if "softmax" in n or "logsoftmax" in n or "reduce_kernel" in n or "norm_kernel" in n:
        return "Reduction"
    if any(t in n for t in ("memset", "memcpy", "permute", "transpose", "contiguous",
                            "split_with_sizes", "chunk_cat", "gather", "scatter",
                            "index_", "copy_kernel")):
        return "Memory"
    if any(t in n for t in ("elementwise", "vectorized", "silu", "gelu", "dropout",
                            "binaryfunctor", "unaryfunctor", "where", "masked",
                            "clamp", "cast")):
        return "Elementwise"
    return "Other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--capture-window-s", type=float, required=True)
    ap.add_argument("--num-ranks", type=int, default=8)
    ap.add_argument("--peak-tflops", type=float, default=989.0)
    args = ap.parse_args()

    cat_ns = defaultdict(float)
    cat_count = defaultdict(int)
    top = []
    total_ns = 0.0

    with open(args.csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ns = float(row["Total Time (ns)"])
            except (KeyError, ValueError):
                continue
            name = row.get("Name", "")
            cat = classify(name)
            cat_ns[cat] += ns
            cat_count[cat] += int(row.get("Instances", 0) or 0)
            top.append((ns, cat, name))
            total_ns += ns

    capture_ns = args.capture_window_s * 1e9
    print(f"\nrank0 nsys trace: {args.csv_path}")
    print(f"capture window: {args.capture_window_s} s")
    print(f"total kernel time on rank0: {total_ns/1e6:.1f} ms "
          f"= {total_ns/capture_ns*100:.1f}% of wall (overlap with stream parallelism possible)")
    print()
    print(f"{'Category':<14}{'ms':>10}{'%kern':>8}{'%wall':>8}{'#kernels':>10}")
    for cat in sorted(cat_ns, key=lambda c: -cat_ns[c]):
        ms = cat_ns[cat] / 1e6
        print(f"{cat:<14}{ms:>10.1f}{cat_ns[cat]/total_ns*100:>7.1f}%{cat_ns[cat]/capture_ns*100:>7.1f}%{cat_count[cat]:>10}")

    gemm_ns = cat_ns.get("GEMM", 0)
    true_mfu_upper = gemm_ns / capture_ns  # already per-rank since trace is single rank
    print()
    print(f"=== TRUE MFU (per-rank GEMM time / wall, upper bound) ===")
    print(f"  GEMM time (rank0): {gemm_ns/1e6:.1f} ms")
    print(f"  Wall (capture window): {args.capture_window_s*1000:.0f} ms")
    print(f"  GEMM / wall ratio: {true_mfu_upper*100:.2f}%")
    print(f"  (== upper bound on real MFU assuming GEMM hits {args.peak_tflops} TF peak)")
    print()
    print("Top 15 kernels:")
    for ns, cat, name in sorted(top, reverse=True)[:15]:
        short = name[:120].replace("\n", " ")
        print(f"  {ns/1e6:>7.1f} ms  [{cat:<11}] {short}")


if __name__ == "__main__":
    main()
