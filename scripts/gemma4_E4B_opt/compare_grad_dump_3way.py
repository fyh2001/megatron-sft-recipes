#!/usr/bin/env python3
"""
3-way per-layer comparison:
  A: DS3 (cell + no detach)
  B: FSDP2 (cell + detach)         -> the new full-run config
  C: FSDP2 (cell + NO detach)      -> the original buggy config (with template gemma4_nothinking)

Hypothesis to test: if detach is the cause of layers 0-13 deficit,
then C (no detach) should match A in layers 0-13 closely, while
B (detach) is much smaller.
"""
from collections import defaultdict


def normalize(name):
    return name.replace("._checkpoint_wrapped_module.", ".")


def load(path_glob, prefix):
    data = defaultdict(float)  # param -> total sum_sq across (rank, micro)
    for rank in range(8):
        with open(f"{path_glob}/{prefix}_rank{rank}.tsv") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 4:
                    continue
                step, micro, param, ss = parts
                data[normalize(param)] += float(ss)
    return data


def layer_idx(p):
    if "language_model.layers." in p:
        try:
            return int(p.split("language_model.layers.")[1].split(".")[0])
        except (ValueError, IndexError):
            return None
    return None


A = load("/tmp/grad_dump_ds3", "ds3")
B = load("/tmp/grad_dump_fsdp2_gemma4_template", "fsdp2")
C = load("/tmp/grad_dump_fsdp2_no_detach_gemma4", "fsdp2")

print("Loaded:")
print(f"  A (DS3 cell, no-detach, template=gemma4)        : {len(A)} params")
print(f"  B (FSDP2 cell + detach, template=gemma4)        : {len(B)} params")
print(f"  C (FSDP2 cell, no-detach, template=gemma4 (apples-to-apples))   : {len(C)} params")

la_d = defaultdict(float)
lb_d = defaultdict(float)
lc_d = defaultdict(float)
for p in A:
    li = layer_idx(p)
    if li is None:
        continue
    la_d[li] += A[p]
    lb_d[li] += B.get(p, 0.0)
    lc_d[li] += C.get(p, 0.0)

print()
print(f"{'layer':>5} | {'A: DS3':>14} | {'B: FSDP+det':>14} | {'C: FSDP no-det':>14} | "
      f"{'B/A':>7} | {'C/A':>7}")
print("-" * 90)
for li in sorted(la_d):
    a, b, c = la_d[li], lb_d[li], lc_d[li]
    ba = b / a if a > 0 else float("inf")
    ca = c / a if a > 0 else float("inf")
    marker = ""
    if li <= 13:
        marker = "  ← layers 0-13 (deficit zone)"
    elif 14 <= li <= 22:
        marker = "  ← KV-source layers"
    elif li == 23:
        marker = "  ← KV-source full"
    elif li >= 24:
        marker = "  ← KV-shared readers"
    print(f"{li:>5} | {a:>14.4e} | {b:>14.4e} | {c:>14.4e} | "
          f"{ba:>7.4f} | {ca:>7.4f}{marker}")

# Aggregate
sum_a_low = sum(la_d[i] for i in range(14))
sum_b_low = sum(lb_d[i] for i in range(14))
sum_c_low = sum(lc_d[i] for i in range(14))
sum_a_src = sum(la_d[i] for i in range(14, 24))
sum_b_src = sum(lb_d[i] for i in range(14, 24))
sum_c_src = sum(lc_d[i] for i in range(14, 24))
sum_a_rd = sum(la_d[i] for i in range(24, 42))
sum_b_rd = sum(lb_d[i] for i in range(24, 42))
sum_c_rd = sum(lc_d[i] for i in range(24, 42))

print()
print(f"{'group':<25} {'A':>14} {'B':>14} {'C':>14} {'B/A':>7} {'C/A':>7}")
print(f"{'layers 0-13 (low)':<25} {sum_a_low:>14.4e} {sum_b_low:>14.4e} {sum_c_low:>14.4e} "
      f"{sum_b_low/sum_a_low:>7.4f} {sum_c_low/sum_a_low:>7.4f}")
print(f"{'layers 14-23 (src)':<25} {sum_a_src:>14.4e} {sum_b_src:>14.4e} {sum_c_src:>14.4e} "
      f"{sum_b_src/sum_a_src:>7.4f} {sum_c_src/sum_a_src:>7.4f}")
print(f"{'layers 24-41 (readers)':<25} {sum_a_rd:>14.4e} {sum_b_rd:>14.4e} {sum_c_rd:>14.4e} "
      f"{sum_b_rd/sum_a_rd:>7.4f} {sum_c_rd/sum_a_rd:>7.4f}")
print(f"{'TOTAL (transformer)':<25} {sum_a_low+sum_a_src+sum_a_rd:>14.4e} "
      f"{sum_b_low+sum_b_src+sum_b_rd:>14.4e} "
      f"{sum_c_low+sum_c_src+sum_c_rd:>14.4e} "
      f"{(sum_b_low+sum_b_src+sum_b_rd)/(sum_a_low+sum_a_src+sum_a_rd):>7.4f} "
      f"{(sum_c_low+sum_c_src+sum_c_rd)/(sum_a_low+sum_a_src+sum_a_rd):>7.4f}")

# Print embed / norm / head, etc.
print()
print("Non-layer params:")
for p in sorted(A):
    if layer_idx(p) is None:
        a = A[p]
        b = B.get(p, 0)
        c = C.get(p, 0)
        ba = b/a if a > 0 else float("inf")
        ca = c/a if a > 0 else float("inf")
        sp = p.replace("model.language_model.", "")
        print(f"  {sp:<60} A={a:.4e} B={b:.4e} C={c:.4e} B/A={ba:.4f} C/A={ca:.4f}")
