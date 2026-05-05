#!/usr/bin/env python3
"""Pinpoint the outlier (rank, micro, param) cells in layers 0-13."""
from collections import defaultdict


def normalize(name):
    return name.replace("._checkpoint_wrapped_module.", ".")


def load(path_glob, prefix):
    data = {}
    for rank in range(8):
        with open(f"{path_glob}/{prefix}_rank{rank}.tsv") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 4:
                    continue
                step, micro, param, ss = parts
                data[(rank, int(micro), normalize(param))] = float(ss)
    return data


def layer_idx(p):
    if "language_model.layers." in p:
        try:
            return int(p.split("language_model.layers.")[1].split(".")[0])
        except (ValueError, IndexError):
            return None
    return None


ds3 = load("/tmp/grad_dump_ds3", "ds3")
fsdp = load("/tmp/grad_dump_fsdp2_kv_detach", "fsdp2")

# Top abs delta in layers 0-13
deltas = []
for k in ds3:
    if k not in fsdp:
        continue
    rank, micro, p = k
    li = layer_idx(p)
    if li is None or li > 13:
        continue
    d, f = ds3[k], fsdp[k]
    if max(d, f) < 1e-3:
        continue
    delta = d - f
    deltas.append((abs(delta), rank, micro, p, d, f, f / d if d > 0 else float("inf")))

print("=" * 110)
print("Top 30 (rank, micro, param) cells with biggest |DS3 - FSDP| in layers 0-13")
print("=" * 110)
print(f"{'|Δ|':>10} {'rank':>4} {'micro':>5} {'param':<60} {'DS3':>12} {'FSDP':>12} {'ratio':>8}")
for absD, rk, mi, p, d, f, r in sorted(deltas, reverse=True)[:30]:
    short_p = p.replace("model.language_model.", "")
    print(f"{absD:>10.4e} {rk:>4} {mi:>5} {short_p:<60} {d:>12.4e} {f:>12.4e} {r:>8.4f}")

print()
print("=" * 110)
print("Same (rank=3, micro=8) — show ALL params in layers 0-13 to see the spike pattern")
print("=" * 110)
ROWS = []
for k in ds3:
    if k not in fsdp:
        continue
    rank, micro, p = k
    if rank != 3 or micro != 8:
        continue
    li = layer_idx(p)
    if li is None or li > 13:
        continue
    d, f = ds3[k], fsdp[k]
    ROWS.append((d, p, f))
ROWS.sort(reverse=True)
print(f"{'DS3':>14} {'param':<70} {'FSDP':>14} {'ratio':>8}")
for d, p, f in ROWS[:25]:
    short_p = p.replace("model.language_model.", "")
    r = f / d if d > 0 else float("inf")
    print(f"{d:>14.4e} {short_p:<70} {f:>14.4e} {r:>8.4f}")

print()
print("=" * 110)
print("Now look at SAME outlier cells but in layers 14-41 (which should match)")
print("=" * 110)
for k in ds3:
    if k not in fsdp:
        continue
    rank, micro, p = k
    if rank != 3 or micro != 8:
        continue
    li = layer_idx(p)
    if li is None or li < 14:
        continue
    d, f = ds3[k], fsdp[k]
    if max(d, f) < 1e-1:
        continue
    short_p = p.replace("model.language_model.", "")
    r = f / d if d > 0 else float("inf")
    print(f"  L{li:>2} {short_p:<60} DS3={d:>12.4e} FSDP={f:>12.4e} r={r:.4f}")

print()
print("=" * 110)
print("Verify: total per-(rank,micro) for ALL layers (not just 0-13). Loss-driven means forward outputs differ.")
print("=" * 110)
totals_d = defaultdict(float)
totals_f = defaultdict(float)
for k in ds3:
    if k not in fsdp:
        continue
    rank, micro, p = k
    totals_d[(rank, micro)] += ds3[k]
    totals_f[(rank, micro)] += fsdp[k]
print(f"  {'rank':>4} {'micro':>5} {'DS3 total':>14} {'FSDP total':>14} {'ratio':>8} {'Δ%':>8}")
extreme = sorted(totals_d.keys(), key=lambda c: abs(totals_d[c] - totals_f[c]), reverse=True)[:10]
for c in extreme:
    rk, mi = c
    d, f = totals_d[c], totals_f[c]
    r = f / d if d > 0 else float("inf")
    pct = (f - d) / d * 100 if d > 0 else 0
    print(f"  {rk:>4} {mi:>5} {d:>14.4e} {f:>14.4e} {r:>8.4f} {pct:>7.1f}%")
