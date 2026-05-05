#!/usr/bin/env python3
"""Compute per-(rank, micro, param) and step-aggregated cosine similarity
between two grad-dump runs (e.g. DS3 vs FSDP2).

Loads the raw .pt files written by sitecustomize.py patch (22) for selected
parameters, then computes:
  - per-cell cos_sim (one number per rank, micro, param)
  - magnitude ratio (|g_FSDP2| / |g_DS3|)
  - step-aggregated grad direction agreement (sum over rank,micro)
"""
import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import torch


def parse_filename(fname):
    # raw_<rank>_step<S>_micro<M>_<param>.pt
    m = re.match(r"raw_(\d+)_step(\d+)_micro(\d+)_(.+)\.pt$", fname)
    if not m:
        return None
    return {
        "rank": int(m.group(1)),
        "step": int(m.group(2)),
        "micro": int(m.group(3)),
        "param": m.group(4),
        "fname": fname,
    }


def load_dir(path):
    """Load all raw .pt files from a directory into a nested dict.

    Returns:
        dict: param -> (rank, micro) -> tensor (fp32 cpu, flattened)
    """
    path = Path(path)
    out = defaultdict(dict)
    for f in sorted(path.glob("raw_*.pt")):
        info = parse_filename(f.name)
        if info is None:
            continue
        t = torch.load(f, map_location="cpu", weights_only=True).float().flatten()
        out[info["param"]][(info["rank"], info["micro"])] = t
    return out


def cossim(a, b):
    a64 = a.double()
    b64 = b.double()
    na = a64.norm()
    nb = b64.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return float((a64 * b64).sum() / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds3", required=True, help="DS3 dump dir")
    ap.add_argument("--fsdp2", required=True, help="FSDP2 dump dir")
    ap.add_argument("--label", default="FSDP2", help="label for FSDP2 in output")
    args = ap.parse_args()

    print(f"Loading DS3 from {args.ds3} ...")
    ds3 = load_dir(args.ds3)
    print(f"Loading {args.label} from {args.fsdp2} ...")
    fsdp2 = load_dir(args.fsdp2)

    common_params = sorted(set(ds3) & set(fsdp2))
    print(f"\nCommon params: {len(common_params)}")
    for p in common_params:
        print(f"  - {p}")
    print()

    for param in common_params:
        ds3_p = ds3[param]
        fs_p = fsdp2[param]
        common_keys = sorted(set(ds3_p) & set(fs_p))
        if not common_keys:
            print(f"  {param}: NO common (rank,micro) keys")
            continue

        per_cell_cos = []
        per_cell_mag_ratio = []
        per_cell_norm_ds3 = []
        per_cell_norm_fs = []

        for k in common_keys:
            a = ds3_p[k]
            b = fs_p[k]
            if a.shape != b.shape:
                print(f"  {param} {k}: SHAPE MISMATCH ds3={tuple(a.shape)} {args.label}={tuple(b.shape)}")
                continue
            cs = cossim(a, b)
            per_cell_cos.append(cs)
            na = a.double().norm().item()
            nb = b.double().norm().item()
            per_cell_norm_ds3.append(na)
            per_cell_norm_fs.append(nb)
            per_cell_mag_ratio.append(nb / na if na > 0 else float("nan"))

        if not per_cell_cos:
            continue

        cos_t = torch.tensor(per_cell_cos)
        mag_t = torch.tensor(per_cell_mag_ratio)

        ds3_sum = sum(ds3_p[k] for k in common_keys)
        fs_sum = sum(fs_p[k] for k in common_keys)
        step_cos = cossim(ds3_sum, fs_sum)
        step_mag_ratio = (fs_sum.double().norm() / ds3_sum.double().norm()).item()

        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"{param}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  shape: {tuple(ds3_p[common_keys[0]].shape)}, n_cells: {len(common_keys)}")
        print(f"")
        print(f"  per-(rank, micro) cos_sim:")
        print(f"    mean   = {cos_t.mean().item():.6f}")
        print(f"    median = {cos_t.median().item():.6f}")
        print(f"    min    = {cos_t.min().item():.6f}")
        print(f"    max    = {cos_t.max().item():.6f}")
        print(f"    std    = {cos_t.std().item():.6f}")
        print(f"")
        print(f"  per-cell |{args.label}|/|DS3| ratio:")
        print(f"    mean   = {mag_t.mean().item():.6f}")
        print(f"    median = {mag_t.median().item():.6f}")
        print(f"    min    = {mag_t.min().item():.6f}")
        print(f"    max    = {mag_t.max().item():.6f}")
        print(f"")
        print(f"  step-summed (sum over rank,micro):")
        print(f"    cos_sim       = {step_cos:.6f}")
        print(f"    |{args.label}|/|DS3|  = {step_mag_ratio:.6f}")
        print(f"    |DS3|         = {ds3_sum.double().norm().item():.6e}")
        print(f"    |{args.label}|       = {fs_sum.double().norm().item():.6e}")

    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("INTERPRETATION GUIDE")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" cos_sim ≈ 1.0    → identical direction (problem is magnitude-only)")
    print(" cos_sim > 0.95   → very similar direction (max_grad_norm should rescue)")
    print(" cos_sim 0.5-0.9  → moderately different direction (training trajectory differs)")
    print(" cos_sim < 0.5    → significantly different direction (fundamental issue)")
    print(" cos_sim < 0      → opposite direction (definitely broken)")


if __name__ == "__main__":
    main()
