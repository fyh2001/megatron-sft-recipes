#!/usr/bin/env python3
"""Compare forward-pass logits between two runs.

Usage:
    python compare_forward.py <ref.npz> <test.npz>
    
Prints relative error stats: max / median / 99th percentile,
plus loss diff.
"""
import sys
import numpy as np


def rel_err(a, b, eps=1e-9):
    """Relative error per-element: |a-b| / max(|a|, |b|, eps)"""
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), eps)


def main():
    if len(sys.argv) < 3:
        print("usage: compare_forward.py <ref.npz> <test.npz>")
        sys.exit(1)
    ref = np.load(sys.argv[1], allow_pickle=True)
    test = np.load(sys.argv[2], allow_pickle=True)

    print(f"Comparing {sys.argv[1]}  vs  {sys.argv[2]}")
    print(f"  ref  config: {dict(zip(['mode','seq_len','seed','run_id'], [c.split('=')[1] for c in ref['config']]))}")
    print(f"  test config: {dict(zip(['mode','seq_len','seed','run_id'], [c.split('=')[1] for c in test['config']]))}")
    print()

    # Loss
    rl, tl = float(ref["loss"][0]), float(test["loss"][0])
    print(f"Loss:  ref={rl:.6f}  test={tl:.6f}  diff={tl - rl:+.6f}  rel_err={abs(tl - rl)/abs(rl):.3e}")
    print()

    # Logits — first/last 10 positions are optional (slim dumps drop them);
    # mean/std are sufficient to call FSDP2/DS forward equivalence.
    for k in ["logits_first10", "logits_last10", "logits_mean", "logits_std"]:
        if k not in ref.files or k not in test.files:
            print(f"{k:>18}  skipped (missing in slim dump)")
            print()
            continue
        a = ref[k].astype(np.float64)
        b = test[k].astype(np.float64)
        if a.shape != b.shape:
            print(f"{k}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            continue
        re = rel_err(a, b)
        ae = np.abs(a - b)
        print(f"{k:>18}  shape={a.shape}")
        print(f"    abs_err: max={ae.max():.3e}  median={np.median(ae):.3e}  p99={np.percentile(ae, 99):.3e}")
        print(f"    rel_err: max={re.max():.3e}  median={np.median(re):.3e}  p99={np.percentile(re, 99):.3e}")
        # Threshold check
        if "logits" in k:
            if re.max() < 1e-3:
                verdict = "✓ tight (bf16 noise floor)"
            elif re.max() < 1e-2:
                verdict = "○ loose (kernel non-det)"
            else:
                verdict = "✗ DIVERGENT"
            print(f"    verdict: {verdict}")
        print()


if __name__ == "__main__":
    main()
