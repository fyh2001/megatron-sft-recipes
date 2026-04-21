#!/usr/bin/env python3
"""Classify CUDA kernels from an `nsys-rep` profile and summarise per-category.

Runs ``nsys stats --report cuda_gpu_kern_sum`` under the hood (needs `nsys`
in PATH or ``NSYS_BIN`` env var) and groups kernels into:

    GEMM (cutlass/cublas)  - matmul, the work we want to see
    NCCL allgather         - FSDP param fetch
    NCCL reducescatter     - FSDP grad reduce
    NCCL allreduce         - Megatron TP sync
    flash-attention        - softmax(QK^T)V on flash-attn
    softmax / CE           - logsoftmax + nll_loss
    elementwise/norm/rope  - elementwise + RMSNorm + silu + rope + adam updates
    FSDP split/cat         - all_gather stage rearrange
    other                  - everything unclassified (should be small)

Usage::

    python nsys_analyze.py /path/to/profile.nsys-rep [--num_ranks 8] [--num_steps 5]

`--num_ranks` and `--num_steps` are used to normalise the "per step per rank"
time, so the final table is comparable between FSDP and Megatron even though
they profile different windows.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict

# Order matters: first match wins.
CATEGORIES: list[tuple[str, re.Pattern]] = [
    ("NCCL allreduce",   re.compile(r"ncclDevKernel_AllReduce|ncclAllReduce")),
    ("NCCL allgather",   re.compile(r"ncclDevKernel_AllGather|ncclAllGather")),
    ("NCCL reducescatter", re.compile(r"ncclDevKernel_ReduceScatter|ncclReduceScatter")),
    ("NCCL other",       re.compile(r"ncclDevKernel|nccl[A-Z]")),
    ("flash-attention",  re.compile(r"flash::flash_|flash_fwd|flash_bwd|fmha|FlashAttn")),
    ("GEMM (cutlass/cublas)", re.compile(
        r"nvjet_tst|ampere_\w+_gemm|sm\d+_xmma_gemm|cutlass\w*gemm|cublasLt|hgemm|"
        r"Kernel.*gemm_|gemv|gemm_[a-z]"
    )),
    ("softmax / CE",     re.compile(r"softmax|log_softmax|nll_loss|cross_entropy")),
    ("FSDP split/cat",   re.compile(
        r"split_with_sizes_copy|chunk_cat_cuda_kernel|_foreach_copy|"
        r"CatArrayBatchedCopy"
    )),
    ("elementwise/norm/rope", re.compile(
        r"triton_poi_fused|triton_red_fused(?!.*softmax)|"
        r"elementwise_kernel|vectorized_elementwise_kernel|reduce_kernel|"
        r"layer_norm|rms_norm|RmsNorm|LayerNorm|silu|Silu|"
        r"multi_tensor_apply_kernel|fused_adam|adam_kernel|"
        r"direct_copy_kernel|unrolled_elementwise"
    )),
]

NSYS_BIN = os.environ.get(
    "NSYS_BIN",
    "/opt/nvidia/nsight-compute/2025.2.1/host/target-linux-x64/nsys",
)


def classify(name: str) -> str:
    for label, pattern in CATEGORIES:
        if pattern.search(name):
            return label
    return "other"


def run_nsys_stats(rep_path: str) -> list[dict]:
    """Return a list of dicts with kernel-level stats."""
    cmd = [
        NSYS_BIN, "stats",
        "--report", "cuda_gpu_kern_sum",
        "--format", "csv",
        rep_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    # Find the CSV block (nsys prepends a banner and the sqlite path). Skip
    # anything before the header line that starts with "Time (%)".
    lines = out.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith("Time (%)")),
        None,
    )
    if start is None:
        raise RuntimeError(f"Could not find CSV header in nsys output:\n{out[:500]}")
    csv_text = "\n".join(lines[start:])
    rows: list[dict] = []
    for row in csv.DictReader(io.StringIO(csv_text)):
        row["Total Time (ns)"] = int(row["Total Time (ns)"])
        row["Instances"] = int(row["Instances"])
        row["Time (%)"] = float(row["Time (%)"])
        rows.append(row)
    return rows


def summarise(rows: list[dict], num_ranks: int, num_steps: int, label: str) -> dict:
    """Group rows by category and compute per-step-per-rank time."""
    by_cat: dict[str, dict] = defaultdict(lambda: {"total_ns": 0, "instances": 0, "pct": 0.0})
    for r in rows:
        cat = classify(r["Name"])
        by_cat[cat]["total_ns"] += r["Total Time (ns)"]
        by_cat[cat]["instances"] += r["Instances"]
        by_cat[cat]["pct"] += r["Time (%)"]

    scale = num_ranks * num_steps  # per-step per-rank normalisation
    total_ns = sum(v["total_ns"] for v in by_cat.values())
    print(f"\n=== {label}  (profile covers ~{num_steps} steps × {num_ranks} ranks) ===")
    print(f"{'Category':<28} {'% of trace':>10} {'Total (ms)':>12} {'per step (ms)':>14} {'kernels':>10} {'avg k/step':>12}")
    print("-" * 88)
    order = [
        "GEMM (cutlass/cublas)", "NCCL allreduce", "NCCL allgather",
        "NCCL reducescatter", "NCCL other", "flash-attention",
        "softmax / CE", "elementwise/norm/rope", "FSDP split/cat",
        "other",
    ]
    results = {}
    for cat in order:
        if cat not in by_cat:
            continue
        d = by_cat[cat]
        per_step_ms = d["total_ns"] / 1e6 / scale
        kernels_per_step = d["instances"] / scale
        pct = d["pct"]
        results[cat] = {
            "pct": pct,
            "total_ms": d["total_ns"] / 1e6,
            "per_step_ms": per_step_ms,
            "instances": d["instances"],
            "kernels_per_step": kernels_per_step,
        }
        print(
            f"{cat:<28} {pct:>9.1f}% {d['total_ns']/1e6:>11.1f} "
            f"{per_step_ms:>13.2f} {d['instances']:>10d} {kernels_per_step:>11.1f}"
        )
    print("-" * 88)
    total_per_step = total_ns / 1e6 / scale
    total_inst = sum(v["instances"] for v in by_cat.values())
    print(
        f"{'TOTAL':<28} {100.0:>9.1f}% {total_ns/1e6:>11.1f} "
        f"{total_per_step:>13.2f} {total_inst:>10d} {total_inst/scale:>11.1f}"
    )
    results["__total__"] = {
        "per_step_ms": total_per_step,
        "instances_per_step": total_inst / scale,
    }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rep", help="Path to .nsys-rep file")
    ap.add_argument("--num_ranks", type=int, default=8,
                    help="Number of GPU ranks in the run (default: 8)")
    ap.add_argument("--num_steps", type=int, default=5,
                    help="Number of training steps covered by the profile window")
    ap.add_argument("--label", type=str, default=None,
                    help="Report label (default: basename of rep file)")
    args = ap.parse_args()

    if not shutil.which(NSYS_BIN):
        print(f"ERROR: nsys not found at {NSYS_BIN}; set NSYS_BIN env var",
              file=sys.stderr)
        return 2

    label = args.label or os.path.basename(os.path.dirname(args.rep))
    rows = run_nsys_stats(args.rep)
    summarise(rows, args.num_ranks, args.num_steps, label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
