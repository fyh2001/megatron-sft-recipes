#!/usr/bin/env python3
"""Compare two swift logging.jsonl curves step-by-step.

Designed for backend-vs-backend training-curve alignment (FSDP2 vs DS), where
both runs share identical micro-batch composition and you want to verify they
produce the same loss / grad_norm trajectory.

Usage:
    python compare_loss_curves.py \
        --a /path/to/A/logging.jsonl --a-label fsdp2_align \
        --b /path/to/B/logging.jsonl --b-label ds_baseline \
        --max-steps 40 \
        --out /path/to/compare_dir

Outputs (in --out):
    compare.tsv     -- step  A_loss  B_loss  loss_abs  loss_pct
                              A_grad  B_grad  grad_abs  grad_pct
                              A_lr    B_lr    A_tok     B_tok
    compare.txt     -- summary stats + side-by-side ASCII chart
                       (max/mean abs+pct delta over first N steps)

Numerical tolerance reference (gemma4 bf16):
    - Forward bitwise (single-GPU rerun):  |Δloss| / |loss| < 1e-4
    - FSDP2 vs single-GPU (DTensor):       |Δloss| / |loss| < 5e-3
    - DS vs single-GPU (autocast):         |Δloss| / |loss| < 1.5e-2
    - FSDP2 vs DS (worst-case):            |Δloss| / |loss| < 2e-2
                                            (both deviating from gt, opposite
                                             directions allowed)

Anything > 2e-2 on step 1 is suspicious and warrants debugging.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def parse_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "global_step/max_steps" not in d:
                continue
            try:
                step = int(str(d["global_step/max_steps"]).split("/")[0])
            except Exception:
                continue
            rows.append({
                "step": step,
                "loss": d.get("loss"),
                "grad_norm": d.get("grad_norm"),
                "lr": d.get("learning_rate"),
                "tokens_this_step": d.get("tokens_this_step"),
                "mem_gib": d.get("memory(GiB)"),
                "token_acc": d.get("token_acc"),
            })
    return rows


def fmt(x, width=10, prec=6):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return f"{'?':>{width}}"
    if isinstance(x, float):
        return f"{x:>{width}.{prec}f}"
    return f"{str(x):>{width}}"


def fmt_pct(x, width=8):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return f"{'?':>{width}}"
    return f"{x*100:>{width-1}.3f}%"


def safe_pct(diff, ref):
    if ref is None or diff is None:
        return None
    if not isinstance(ref, (int, float)) or abs(ref) < 1e-12:
        return None
    return diff / ref


def safe_diff(a, b):
    if a is None or b is None:
        return None
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    return a - b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="path to A logging.jsonl")
    ap.add_argument("--b", required=True, help="path to B logging.jsonl")
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--max-steps", type=int, default=40,
                    help="compare first N steps only (default 40)")
    ap.add_argument("--out", required=True, help="output directory for compare.{tsv,txt}")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_a = parse_jsonl(Path(args.a))
    rows_b = parse_jsonl(Path(args.b))
    if not rows_a:
        print(f"[ERR] no per-step records in {args.a}")
        return 2
    if not rows_b:
        print(f"[ERR] no per-step records in {args.b}")
        return 2

    map_a = {r["step"]: r for r in rows_a}
    map_b = {r["step"]: r for r in rows_b}
    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    common = [s for s in common if s <= args.max_steps]
    if not common:
        print(f"[ERR] no overlapping steps within first {args.max_steps}")
        return 2

    A, B = args.a_label, args.b_label

    tsv_path = out_dir / "compare.tsv"
    cols = ["step",
            f"{A}_loss", f"{B}_loss", "loss_abs", "loss_pct",
            f"{A}_grad", f"{B}_grad", "grad_abs", "grad_pct",
            f"{A}_lr", f"{B}_lr",
            f"{A}_tok", f"{B}_tok"]
    with open(tsv_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for s in common:
            ra, rb = map_a[s], map_b[s]
            la, lb = ra.get("loss"), rb.get("loss")
            ga, gb = ra.get("grad_norm"), rb.get("grad_norm")
            ld = safe_diff(la, lb); gd = safe_diff(ga, gb)
            lp = safe_pct(ld, lb);  gp = safe_pct(gd, gb)
            row = [
                s,
                la, lb, ld, lp,
                ga, gb, gd, gp,
                ra.get("lr"), rb.get("lr"),
                ra.get("tokens_this_step"), rb.get("tokens_this_step"),
            ]
            f.write("\t".join("" if v is None else str(v) for v in row) + "\n")

    # Summary + ASCII side-by-side
    txt_path = out_dir / "compare.txt"

    loss_abs_list = [abs(safe_diff(map_a[s].get("loss"), map_b[s].get("loss")))
                     for s in common
                     if isinstance(map_a[s].get("loss"), (int, float))
                     and isinstance(map_b[s].get("loss"), (int, float))]
    loss_pct_list = [abs(safe_pct(safe_diff(map_a[s].get("loss"), map_b[s].get("loss")),
                                   map_b[s].get("loss")))
                     for s in common
                     if isinstance(map_a[s].get("loss"), (int, float))
                     and isinstance(map_b[s].get("loss"), (int, float))
                     and abs(map_b[s].get("loss") or 0) > 1e-12]
    grad_pct_list = [abs(safe_pct(safe_diff(map_a[s].get("grad_norm"), map_b[s].get("grad_norm")),
                                   map_b[s].get("grad_norm")))
                     for s in common
                     if isinstance(map_a[s].get("grad_norm"), (int, float))
                     and isinstance(map_b[s].get("grad_norm"), (int, float))
                     and abs(map_b[s].get("grad_norm") or 0) > 1e-12]

    def stats(xs, name):
        if not xs:
            return f"{name}: (no data)"
        return (f"{name}: n={len(xs)} max={max(xs):.4g} "
                f"mean={sum(xs)/len(xs):.4g} min={min(xs):.4g}")

    with open(txt_path, "w") as f:
        f.write(f"# Compare {A} vs {B}  (first {args.max_steps} common steps)\n")
        f.write(f"# A: {args.a}\n")
        f.write(f"# B: {args.b}\n")
        f.write(f"# common steps: {len(common)}  "
                f"(A had {len(rows_a)}, B had {len(rows_b)} total)\n\n")

        f.write("== Step-1 sanity ==\n")
        if common and 1 in map_a and 1 in map_b:
            la = map_a[1].get("loss"); lb = map_b[1].get("loss")
            ga = map_a[1].get("grad_norm"); gb = map_b[1].get("grad_norm")
            f.write(f"  {A}.loss   = {la}\n")
            f.write(f"  {B}.loss   = {lb}\n")
            ld = safe_diff(la, lb); lp = safe_pct(ld, lb)
            f.write(f"  Δloss     = {ld}  ({lp*100 if lp is not None else 'n/a'}{'%' if lp is not None else ''})\n")
            f.write(f"  {A}.grad   = {ga}\n")
            f.write(f"  {B}.grad   = {gb}\n")
            gd = safe_diff(ga, gb); gp = safe_pct(gd, gb)
            f.write(f"  Δgrad     = {gd}  ({gp*100 if gp is not None else 'n/a'}{'%' if gp is not None else ''})\n")
        else:
            f.write("  (no overlapping step 1 row)\n")
        f.write("\n")

        f.write("== Aggregate over first N steps ==\n")
        f.write("  " + stats(loss_abs_list, "abs(Δloss)") + "\n")
        f.write("  " + stats(loss_pct_list, "abs(Δloss/B)") + "\n")
        f.write("  " + stats(grad_pct_list, "abs(Δgrad/B)") + "\n\n")

        f.write("== Verdict (heuristic, gemma4 bf16) ==\n")
        if loss_pct_list:
            mp = max(loss_pct_list)
            if mp < 5e-3:
                v = "EXCELLENT (<0.5%)"
            elif mp < 1.5e-2:
                v = "GOOD (<1.5%, within DS bf16 mode budget)"
            elif mp < 3e-2:
                v = "BORDERLINE (1.5-3%, double-check tokenisation/seed/sampler)"
            else:
                v = "BAD (>3%, framework alignment broken)"
            f.write(f"  loss-pct verdict: {v}  (max abs Δloss/B = {mp*100:.2f}%)\n")
        f.write("\n")

        f.write("== Side-by-side ==\n")
        f.write(f"{'step':>5}  "
                f"{A+'.loss':>11}  {B+'.loss':>11}  {'Δloss':>10}  {'Δloss%':>8}  "
                f"{A+'.grad':>10}  {B+'.grad':>10}  {'Δgrad%':>8}  "
                f"{A+'.tok':>8}  {B+'.tok':>8}\n")
        for s in common:
            ra, rb = map_a[s], map_b[s]
            la, lb = ra.get("loss"), rb.get("loss")
            ga, gb = ra.get("grad_norm"), rb.get("grad_norm")
            ld = safe_diff(la, lb); lp = safe_pct(ld, lb)
            gd = safe_diff(ga, gb); gp = safe_pct(gd, gb)
            f.write(f"{s:>5}  {fmt(la,11)}  {fmt(lb,11)}  {fmt(ld,10,5)}  {fmt_pct(lp,8)}  "
                    f"{fmt(ga,10,4)}  {fmt(gb,10,4)}  {fmt_pct(gp,8)}  "
                    f"{fmt(ra.get('tokens_this_step'),8,0)}  "
                    f"{fmt(rb.get('tokens_this_step'),8,0)}\n")

    print(f"wrote {tsv_path}")
    print(f"wrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
