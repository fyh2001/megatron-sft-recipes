#!/usr/bin/env python3
"""Compare DS3 baseline vs our FSDP2 run with KV detach fix.

Reads two `logging.jsonl` files (one row per training step), aligns by
`global_step`, and produces a 2×3 figure of:
  Row 1 (absolute curves):
    [0,0] loss vs step
    [0,1] grad_norm vs step
    [0,2] step_time(s) vs step
  Row 2 (relative comparison):
    [1,0] Δloss = FSDP2 - DS3, vs step
    [1,1] grad_norm ratio = FSDP2 / DS3, vs step (log y)
    [1,2] token_acc vs step

Usage:
    python plot_compare_ds3_fsdp2.py
    python plot_compare_ds3_fsdp2.py --baseline /path/to/baseline.jsonl
    python plot_compare_ds3_fsdp2.py --fsdp2 /path/to/run/v0-*/logging.jsonl
    python plot_compare_ds3_fsdp2.py --output /tmp/compare.png

Re-run any time during training: it picks up the latest log lines.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BASELINE = "/mnt/shared/fyh/v4-20260429-144702/logging.jsonl"
DEFAULT_RUN_GLOB = os.environ.get(
    "DEFAULT_RUN_GLOB",
    "/home/ubuntu/fyh/megatron-sft-recipes/experiments/"
    "gemma4_E4B_alt_offload/run_*_fsdp2_offload_a3_pf_2ep_*/"
    "v0-*/logging.jsonl",
)


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            # Only keep training-step rows (have loss + global_step)
            if "loss" not in d or "global_step/max_steps" not in d:
                continue
            try:
                step = int(str(d["global_step/max_steps"]).split("/")[0])
            except (ValueError, IndexError):
                continue
            d["_step"] = step
            rows.append(d)
    return rows


def to_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    """Returns dict of {name: np.array}, indexed by row order."""
    keys = ["_step", "loss", "grad_norm", "learning_rate", "token_acc",
            "train_speed(s/it)", "memory(GiB)"]
    out = {}
    for k in keys:
        vals = []
        for r in rows:
            v = r.get(k)
            try:
                vals.append(float(v) if v is not None else np.nan)
            except (TypeError, ValueError):
                vals.append(np.nan)
        out[k] = np.array(vals)
    return out


def align_on_step(a: dict[str, np.ndarray], b: dict[str, np.ndarray]):
    """Return (steps, a_by_step, b_by_step) keeping only steps present in BOTH."""
    a_map = {int(s): i for i, s in enumerate(a["_step"])}
    b_map = {int(s): i for i, s in enumerate(b["_step"])}
    common = sorted(set(a_map.keys()) & set(b_map.keys()))
    if not common:
        return np.array([]), {}, {}
    steps = np.array(common)
    a_idx = np.array([a_map[s] for s in common])
    b_idx = np.array([b_map[s] for s in common])
    a_aligned = {k: v[a_idx] for k, v in a.items()}
    b_aligned = {k: v[b_idx] for k, v in b.items()}
    return steps, a_aligned, b_aligned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=DEFAULT_BASELINE,
                    help="Path to DS3 baseline logging.jsonl")
    ap.add_argument("--fsdp2", default=None,
                    help="Path to FSDP2 logging.jsonl (defaults to latest run)")
    ap.add_argument("--output", default=None,
                    help="Output PNG path (default next to fsdp2 jsonl)")
    ap.add_argument("--smooth", type=int, default=5,
                    help="Window size for moving-average smoothing on Δloss "
                         "and grad_norm ratio (set 1 to disable)")
    ap.add_argument("--label", default=None,
                    help="Legend label for FSDP2 curves (e.g. 'FSDP2 + detach' "
                         "or 'FSDP2 no-detach'). Defaults to auto-detect from "
                         "run dir name.")
    ap.add_argument("--max_step", type=int, default=None,
                    help="Truncate the data at this step (inclusive); useful for "
                         "milestone snapshots so step50.png only shows steps 1-50.")
    args = ap.parse_args()

    if args.fsdp2 is None:
        candidates = sorted(glob.glob(DEFAULT_RUN_GLOB))
        if not candidates:
            print("No FSDP2 logging.jsonl found via glob", file=sys.stderr)
            sys.exit(1)
        args.fsdp2 = candidates[-1]
        print(f"[info] using latest FSDP2 jsonl: {args.fsdp2}")

    if args.output is None:
        args.output = str(Path(args.fsdp2).parent.parent / "compare_ds3_fsdp2.png")

    if args.label is None:
        run_name = Path(args.fsdp2).parent.parent.name
        if "NO_detach" in run_name:
            args.label = "FSDP2 no-detach"
        elif "detach" in run_name or "a3_pf" in run_name:
            args.label = "FSDP2 + detach"
        else:
            args.label = "FSDP2"

    if not os.path.isfile(args.baseline):
        print(f"baseline not found: {args.baseline}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.fsdp2):
        print(f"fsdp2 not found: {args.fsdp2}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] loading baseline: {args.baseline}")
    base_rows = load_jsonl(args.baseline)
    print(f"[info]   loaded {len(base_rows)} rows")
    print(f"[info] loading fsdp2:    {args.fsdp2}")
    fsdp_rows = load_jsonl(args.fsdp2)
    print(f"[info]   loaded {len(fsdp_rows)} rows")

    base_a = to_arrays(base_rows)
    fsdp_a = to_arrays(fsdp_rows)
    steps, base, fsdp = align_on_step(base_a, fsdp_a)
    if steps.size == 0:
        print("No common steps between the two runs", file=sys.stderr)
        sys.exit(1)
    if args.max_step is not None:
        mask = steps <= args.max_step
        steps = steps[mask]
        base = {k: v[mask] for k, v in base.items()}
        fsdp = {k: v[mask] for k, v in fsdp.items()}
    print(f"[info] aligned on {steps.size} common steps "
          f"(step {steps[0]} → step {steps[-1]})")

    # Optional moving-average smoothing for noisy curves
    def smooth(x, w):
        if w is None or w <= 1:
            return x
        x = np.asarray(x, dtype=float)
        # ignore NaNs in the window
        kernel = np.ones(w) / w
        # use np.convolve in valid mode then pad-edge to keep length
        valid = np.convolve(np.where(np.isnan(x), 0, x), kernel, mode="same")
        cnt = np.convolve((~np.isnan(x)).astype(float), kernel, mode="same")
        with np.errstate(invalid="ignore"):
            res = np.where(cnt > 0, valid / cnt, np.nan)
        return res

    from matplotlib.ticker import ScalarFormatter, PercentFormatter

    def no_offset(ax):
        """Disable matplotlib's scientific / offset notation on the y-axis;
        always show plain decimal numbers."""
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    fig.suptitle(
        f"DS3 baseline vs {args.label} — Gemma-4-E4B-it text-only SFT\n"
        f"GBS=128, GAS=16, NPROC=8, max_length=16384, seed=42",
        fontsize=13,
    )

    # [0,0] loss
    ax = axes[0, 0]
    ax.plot(steps, base["loss"], label="DS3", color="#1f77b4", lw=1.2)
    ax.plot(steps, fsdp["loss"], label=args.label, color="#d62728", lw=1.2)
    ax.set_title("loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    no_offset(ax)

    # [0,1] grad_norm
    ax = axes[0, 1]
    ax.plot(steps, base["grad_norm"], label="DS3", color="#1f77b4", lw=1.2)
    ax.plot(steps, fsdp["grad_norm"], label=args.label, color="#d62728", lw=1.2)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="max_grad_norm=1.0")
    ax.set_title("grad_norm (pre-clip)")
    ax.set_xlabel("step")
    ax.set_ylabel("grad_norm")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    no_offset(ax)

    # [0,2] step time
    ax = axes[0, 2]
    ax.plot(steps, base["train_speed(s/it)"], label=f"DS3 (mean={np.nanmean(base['train_speed(s/it)']):.1f}s)",
            color="#1f77b4", lw=1.2)
    ax.plot(steps, fsdp["train_speed(s/it)"], label=f"FSDP2 (mean={np.nanmean(fsdp['train_speed(s/it)']):.1f}s)",
            color="#d62728", lw=1.2)
    ax.set_title("step time (rolling avg s/it)")
    ax.set_xlabel("step")
    ax.set_ylabel("seconds")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    no_offset(ax)

    # [1,0] Δloss
    ax = axes[1, 0]
    delta = fsdp["loss"] - base["loss"]
    ax.plot(steps, delta, color="#9467bd", lw=0.7, alpha=0.5, label="raw")
    if args.smooth > 1:
        ax.plot(steps, smooth(delta, args.smooth), color="#9467bd", lw=1.5,
                label=f"smoothed (w={args.smooth})")
    ax.axhline(0.0, color="gray", ls="--", lw=0.8)
    pct = 100 * np.nanmean(np.abs(delta[steps >= 50])) / np.nanmean(base["loss"][steps >= 50]) \
        if (steps >= 50).any() else float("nan")
    ax.set_title(f"Δloss = FSDP2 − DS3   (mean |Δ| / mean DS3 loss after step 50: {pct:.2f}%)")
    ax.set_xlabel("step")
    ax.set_ylabel("Δloss")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    no_offset(ax)

    # [1,1] grad_norm ratio
    ax = axes[1, 1]
    ratio = fsdp["grad_norm"] / np.where(base["grad_norm"] > 0, base["grad_norm"], np.nan)
    ax.plot(steps, ratio, color="#9467bd", lw=0.7, alpha=0.5, label="raw")
    if args.smooth > 1:
        ax.plot(steps, smooth(ratio, args.smooth), color="#9467bd", lw=1.5,
                label=f"smoothed (w={args.smooth})")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="ratio=1.0 (perfect match)")
    ax.set_title("grad_norm ratio = FSDP2 / DS3")
    ax.set_xlabel("step")
    ax.set_ylabel("ratio")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    no_offset(ax)

    # [1,2] token_acc
    ax = axes[1, 2]
    ax.plot(steps, base["token_acc"], label="DS3", color="#1f77b4", lw=1.2)
    ax.plot(steps, fsdp["token_acc"], label=args.label, color="#d62728", lw=1.2)
    ax.set_title("token_acc")
    ax.set_xlabel("step")
    ax.set_ylabel("token_acc")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    no_offset(ax)

    plt.savefig(args.output, dpi=130, bbox_inches="tight")
    print(f"[info] wrote: {args.output}")

    # Also print a quick summary table
    print()
    print("=== summary at key checkpoints ===")
    print(f"{'step':>5} | {'DS3 loss':>8} {'DS3 gn':>7} | {'FSDP2 loss':>10} {'FSDP2 gn':>9} "
          f"| {'Δloss':>7} {'gn ratio':>9}")
    print("-" * 80)
    targets = [1, 5, 10, 25, 50, 100, 200, 400, 600, 800]
    step_to_idx = {int(s): i for i, s in enumerate(steps)}
    for t in targets:
        if t not in step_to_idx:
            continue
        i = step_to_idx[t]
        bl, bg = base["loss"][i], base["grad_norm"][i]
        fl, fg = fsdp["loss"][i], fsdp["grad_norm"][i]
        print(f"{t:>5} | {bl:>8.4f} {bg:>7.3f} | {fl:>10.4f} {fg:>9.3f} "
              f"| {fl-bl:>+7.4f} {(fg/bg if bg>0 else float('inf')):>9.3f}")
    print(f"current latest step: {int(steps[-1])} / 806")


if __name__ == "__main__":
    main()
