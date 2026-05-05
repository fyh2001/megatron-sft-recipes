#!/usr/bin/env python3
"""Layer 0-13 deep-dive: find WHICH params are off and HOW (deterministic vs noise)."""
import os, sys, statistics
from collections import defaultdict


def normalize(name):
    return name.replace("._checkpoint_wrapped_module.", ".")


def load(path_glob, prefix):
    data = defaultdict(dict)  # (rank, micro, param) -> sum_sq
    for rank in range(8):
        fpath = f"{path_glob}/{prefix}_rank{rank}.tsv"
        with open(fpath) as f:
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


def param_type(p):
    """Categorize param by role within a layer."""
    if "self_attn" in p:
        for k in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"]:
            if k in p:
                return f"attn.{k}"
        return "attn.other"
    if "mlp" in p:
        for k in ["gate_proj", "up_proj", "down_proj"]:
            if k in p:
                return f"mlp.{k}"
        return "mlp.other"
    if "norm" in p:
        return "norm"
    return "other"


def main():
    ds3 = load("/tmp/grad_dump_ds3", "ds3")
    fsdp = load("/tmp/grad_dump_fsdp2_gemma4_template", "fsdp2")

    print(f"DS3 entries: {len(ds3):,}")
    print(f"FSDP entries: {len(fsdp):,}")
    common = set(ds3.keys()) & set(fsdp.keys())
    print(f"Common: {len(common):,}\n")

    # ============ 1. Per-layer aggregate: confirm 0-13 deficit ============
    print("=" * 90)
    print("§1. Per-layer aggregate (sum over rank×micro×param-in-layer) [layers 0-13]")
    print("=" * 90)
    print(f"{'layer':>6} {'DS3 sum_sq':>16} {'FSDP sum_sq':>16} {'ratio':>8} {'Δ':>16}")
    layer_d = defaultdict(float)
    layer_f = defaultdict(float)
    for k in common:
        rank, micro, p = k
        li = layer_idx(p)
        if li is None or li > 13:
            continue
        layer_d[li] += ds3[k]
        layer_f[li] += fsdp[k]
    for li in sorted(layer_d):
        d, f = layer_d[li], layer_f[li]
        r = f / d if d > 0 else float('inf')
        print(f"{li:>6} {d:>16.4e} {f:>16.4e} {r:>8.4f} {f - d:>16.4e}")

    # ============ 2. Per-(layer, param-type) ratio ============
    print()
    print("=" * 90)
    print("§2. Per-(layer, param-type) — layers 0-13 only — find which TYPE is off")
    print("=" * 90)
    type_per_layer_d = defaultdict(float)
    type_per_layer_f = defaultdict(float)
    for k in common:
        rank, micro, p = k
        li = layer_idx(p)
        if li is None or li > 13:
            continue
        t = param_type(p)
        type_per_layer_d[(li, t)] += ds3[k]
        type_per_layer_f[(li, t)] += fsdp[k]
    types = sorted(set(t for _, t in type_per_layer_d))
    header = f"{'layer':>5} | " + " | ".join(f"{t:>22}" for t in types)
    print(header)
    print("-" * len(header))
    for li in range(14):
        row = [f"{li:>5}"]
        for t in types:
            d = type_per_layer_d.get((li, t), 0)
            f = type_per_layer_f.get((li, t), 0)
            r = f / d if d > 0 else float('inf')
            row.append(f"{r:>10.3f}({d:>5.2e})")
        print(" | ".join(row))

    # Aggregate by type only (layers 0-13)
    print()
    print(f"{'type':<22} {'DS3 sum':>14} {'FSDP sum':>14} {'ratio':>8}")
    type_d = defaultdict(float)
    type_f = defaultdict(float)
    for (li, t), v in type_per_layer_d.items():
        type_d[t] += v
        type_f[t] += type_per_layer_f[(li, t)]
    for t in types:
        r = type_f[t] / type_d[t] if type_d[t] > 0 else float('inf')
        print(f"{t:<22} {type_d[t]:>14.4e} {type_f[t]:>14.4e} {r:>8.4f}")

    # ============ 3. Determinism check: per-(rank, micro) ratio for same layer ============
    print()
    print("=" * 90)
    print("§3. Determinism: layer 0 sum across all params, per (rank, micro) cell")
    print("=" * 90)
    print(f"  if FSDP < DS3 deterministically across all 128 cells -> structural")
    print(f"  if random with mean 1.0 -> numerical noise")
    cells_d = defaultdict(float)
    cells_f = defaultdict(float)
    for k in common:
        rank, micro, p = k
        li = layer_idx(p)
        if li != 0:
            continue
        cells_d[(rank, micro)] += ds3[k]
        cells_f[(rank, micro)] += fsdp[k]

    ratios_layer0 = []
    for cell in sorted(cells_d):
        d, f = cells_d[cell], cells_f[cell]
        if d > 0:
            ratios_layer0.append(f / d)
    print(f"  layer 0 cells: n={len(ratios_layer0)}")
    print(f"    mean ratio = {statistics.mean(ratios_layer0):.4f}")
    print(f"    median     = {statistics.median(ratios_layer0):.4f}")
    print(f"    stdev      = {statistics.stdev(ratios_layer0):.4f}")
    print(f"    min        = {min(ratios_layer0):.4f}")
    print(f"    max        = {max(ratios_layer0):.4f}")
    print(f"    fraction >1 = {sum(1 for r in ratios_layer0 if r>1)/len(ratios_layer0):.2%}")

    # also for layer 5 and 10
    for target_li in [5, 10, 13]:
        cells_d2 = defaultdict(float)
        cells_f2 = defaultdict(float)
        for k in common:
            rank, micro, p = k
            li = layer_idx(p)
            if li != target_li:
                continue
            cells_d2[(rank, micro)] += ds3[k]
            cells_f2[(rank, micro)] += fsdp[k]
        ratios = [cells_f2[c] / cells_d2[c] for c in sorted(cells_d2) if cells_d2[c] > 0]
        print(f"  layer {target_li}: mean={statistics.mean(ratios):.4f} med={statistics.median(ratios):.4f} stdev={statistics.stdev(ratios):.4f} min={min(ratios):.4f} max={max(ratios):.4f}")

    # ============ 4. Check rank-only and micro-only patterns ============
    print()
    print("=" * 90)
    print("§4. Per-rank totals (layers 0-13 only); per-micro totals (layers 0-13 only)")
    print("=" * 90)
    rank_d = defaultdict(float)
    rank_f = defaultdict(float)
    micro_d = defaultdict(float)
    micro_f = defaultdict(float)
    for k in common:
        rank, micro, p = k
        li = layer_idx(p)
        if li is None or li > 13:
            continue
        rank_d[rank] += ds3[k]
        rank_f[rank] += fsdp[k]
        micro_d[micro] += ds3[k]
        micro_f[micro] += fsdp[k]

    print(f"\n  per-RANK (layers 0-13):")
    print(f"  {'rank':>4} {'DS3':>14} {'FSDP':>14} {'ratio':>8}")
    for r in sorted(rank_d):
        ratio = rank_f[r] / rank_d[r]
        print(f"  {r:>4} {rank_d[r]:>14.4e} {rank_f[r]:>14.4e} {ratio:>8.4f}")
    print(f"\n  per-MICRO (layers 0-13):")
    print(f"  {'micro':>5} {'DS3':>14} {'FSDP':>14} {'ratio':>8}")
    for m in sorted(micro_d):
        ratio = micro_f[m] / micro_d[m]
        print(f"  {m:>5} {micro_d[m]:>14.4e} {micro_f[m]:>14.4e} {ratio:>8.4f}")

    # ============ 5. Compare layers 14-23 (sources) and 24-41 (readers) layer-by-layer ============
    print()
    print("=" * 90)
    print("§5. Sanity: layers 14-41 ratio (should be ~1.0 since detach fixes the source/reader mismatch)")
    print("=" * 90)
    layer_d_all = defaultdict(float)
    layer_f_all = defaultdict(float)
    for k in common:
        rank, micro, p = k
        li = layer_idx(p)
        if li is None or li < 14:
            continue
        layer_d_all[li] += ds3[k]
        layer_f_all[li] += fsdp[k]
    print(f"  {'layer':>6} {'DS3':>16} {'FSDP':>16} {'ratio':>8}")
    for li in sorted(layer_d_all):
        d, f = layer_d_all[li], layer_f_all[li]
        r = f / d if d > 0 else float('inf')
        print(f"  {li:>6} {d:>16.4e} {f:>16.4e} {r:>8.4f}")


if __name__ == "__main__":
    main()
