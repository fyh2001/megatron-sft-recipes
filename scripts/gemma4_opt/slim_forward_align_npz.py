#!/usr/bin/env python3
"""Slim p0_forward_align npz dumps for git committal.

Reads each *.npz produced by forward_align_test.py and writes a slim copy
that drops the ~22 MB logits_first10 / logits_last10 spot-check tensors.
The retained fields (logits_mean, logits_std, loss, config) are sufficient
for the FSDP2 = single-GPU vs DS bf16 1.5%-diff verdict that the
walkthrough §0d/§0e/§0g chapters cite.

Per-file size: ~22.97 MB -> ~2 MB.

Usage:
    python slim_forward_align_npz.py \
        --src /home/ubuntu/fyh/megatron-sft-recipes/experiments/gemma4_opt/p0_forward_align \
        --dst /home/ubuntu/fyh/megatron-sft-recipes/experiments/gemma4_opt/p0_forward_align_slim
"""
import argparse
from pathlib import Path

import numpy as np

KEEP = ("logits_mean", "logits_std", "loss", "config")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        default="/home/ubuntu/fyh/megatron-sft-recipes/experiments/gemma4_opt/p0_forward_align",
    )
    p.add_argument(
        "--dst",
        default="/home/ubuntu/fyh/megatron-sft-recipes/experiments/gemma4_opt/p0_forward_align_slim",
    )
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    src_files = sorted(src.glob("*.npz"))
    if not src_files:
        raise SystemExit(f"no *.npz in {src}")

    for f in src_files:
        data = np.load(f, allow_pickle=True)
        keep = {k: data[k] for k in KEEP if k in data.files}
        out = dst / f.name
        np.savez(out, **keep)
        before = f.stat().st_size / 1024 / 1024
        after = out.stat().st_size / 1024 / 1024
        kept = ",".join(keep.keys())
        print(f"{f.name}: {before:.2f} MB -> {after:.2f} MB  (kept: {kept})")


if __name__ == "__main__":
    main()
