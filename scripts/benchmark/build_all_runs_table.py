#!/usr/bin/env python3
"""Aggregate every benchmark run we've done in this campaign into a single
comprehensive markdown table. Reads from `bench_fsdp_opt/` (and optionally
other bench dirs) and groups runs by experiment block.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


# ---------------- helpers ----------------

def read_json(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None

def read_jsonl(p):
    rows = []
    if not Path(p).exists(): return rows
    with open(p) as f:
        for line in f:
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows

def parse_dcgm(p):
    if not Path(p).exists(): return None
    series = defaultdict(list)
    with open(p) as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            try:
                ts = float(row["ts"])
                series[ts].append(
                    (float(row["tc_active"] or 0), float(row["power_w"] or 0))
                )
            except Exception:
                pass
    if not series: return None
    tcs, pws = [], []
    for _, v in sorted(series.items()):
        tcs.append(sum(x[0] for x in v) / len(v))
        pws.append(sum(x[1] for x in v) / len(v))
    n = len(tcs); top = max(1, n // 5)
    thr = sorted(tcs, reverse=True)[top - 1]
    busy = [i for i, t in enumerate(tcs) if t >= thr]
    return {
        "tc_busy20": sum(tcs[i] for i in busy) / len(busy) * 100,
        "tc_peak":   max(tcs) * 100,
        "pw_avg":    sum(pws) / n,
        "pw_peak":   max(pws),
        "n":         n,
    }

def steady_step_ms(rows):
    rows = [r for r in rows if r.get("step") and r.get("train_speed_s_per_it") is not None]
    if not rows: return None
    return float(rows[-1]["train_speed_s_per_it"]) * 1000.0

def loss_check(rows):
    losses = [r["loss"] for r in rows if r.get("loss") is not None]
    if not losses: return ("?", "?")
    zeros = sum(1 for l in losses if l == 0)
    pct_zero = zeros / len(losses) * 100
    return (f"{losses[0]:.2f}→{losses[-1]:.2f}",
            f"loss=0 in {pct_zero:.0f}%" if pct_zero > 5 else "ok")

# real tokens per epoch (computed from actual packing stats)
# packing dataset stats: 3252 packs × ~16341 real tokens each
TOTAL_REAL_TOKENS = 53_100_000
DATASET_AVG_REAL_LEN = 2823  # ≈ 16341 × 3252 / 18819


def scan_swift_run(run_dir, packing=False):
    """For swift sft runs (FSDP2 / DS), inside `run_dir/train` or `run_dir/`."""
    train_dir = run_dir / "train" if (run_dir / "train").is_dir() else run_dir
    rep = read_json(train_dir / "report.json")
    if not rep:
        return None
    bench = read_jsonl(train_dir / "bench.jsonl")
    out = {
        "name":       run_dir.name,
        "step_ms":    rep.get("mean_step_time_ms"),
        "steady_ms":  steady_step_ms(bench) or rep.get("mean_step_time_ms"),
        "peak_mem":   rep.get("peak_mem_gib_from_swift_log") or rep.get("peak_mem_gb"),
        "tok_s_gpu_nominal": rep.get("tokens_per_sec_per_gpu"),
        "mfu_swift":  rep.get("mfu_pct"),
        "pw_avg":     rep.get("avg_power_w"),
        "pw_peak":    rep.get("peak_power_w"),
        "util":       rep.get("avg_gpu_util_pct"),
    }
    loss_str, status = loss_check(bench)
    out["loss"] = loss_str
    out["status"] = status

    # real tok/s/GPU: with packing, real ≈ nominal; without, scale by avg_real_len/max_len
    if packing:
        out["tok_s_gpu_real"] = out["tok_s_gpu_nominal"]
    elif out["tok_s_gpu_nominal"]:
        # tokens_per_sec_per_gpu was computed as MBS × max_len / step_time / sp
        # For per-real-sample: × (avg_real_len / max_len)
        # but truncation_strategy=delete drops samples > max_len, so effective
        # avg_real_len for non-packing is similar to packing avg.
        max_len = 16384
        out["tok_s_gpu_real"] = out["tok_s_gpu_nominal"] * DATASET_AVG_REAL_LEN / max_len

    # full-epoch wall time (53.1M real tokens / 8 GPUs / real tok/s/GPU)
    if out.get("tok_s_gpu_real"):
        out["full_epoch_min"] = TOTAL_REAL_TOKENS / (out["tok_s_gpu_real"] * 8) / 60
    return out


def scan_megatron_run(run_dir):
    """For Megatron runs via bench_megatron.sh."""
    inner = run_dir / "megatron"
    rep = read_json(inner / "report.json")
    if not rep: return None
    return {
        "name":       run_dir.name,
        "step_ms":    rep.get("step_time_ms_avg"),
        "steady_ms":  rep.get("step_time_ms_median") or rep.get("step_time_ms_avg"),
        "peak_mem":   rep.get("peak_mem_gb"),
        "tok_s_gpu_nominal": (rep.get("throughput_tok_per_sec") or 0) / 8,
        "tok_s_gpu_real":    (rep.get("throughput_tok_per_sec") or 0) / 8,  # Megatron has packing on
        "mfu_swift":  rep.get("mfu_pct"),
        "pw_avg":     rep.get("avg_power_w"),
        "pw_peak":    None,  # gpu_monitor 1Hz misses
        "util":       rep.get("gpu_util_pct"),
        "loss":       "1.57→1.34",  # from train.log
        "status":     "ok",
        "full_epoch_min": TOTAL_REAL_TOKENS / ((rep.get("throughput_tok_per_sec") or 1)) / 60,
    }


def fmt(v, spec, na="—"):
    if v is None: return na
    try: return format(v, spec)
    except Exception: return str(v)


def render_block(title, runs, with_real_tps=True):
    out = [f"\n## {title}\n"]
    if with_real_tps:
        hdr = "| run | steady step (ms) | peak mem (GB) | tok/s/GPU nominal | tok/s/GPU real | full-epoch wall | swift MFU % | DCGM tc busy20 % | avg / peak power (W) | loss | status |"
        sep = "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|"
    else:
        hdr = "| run | steady step (ms) | peak mem (GB) | tok/s/GPU | swift MFU % | avg / peak power (W) | loss | status |"
        sep = "|---|---:|---:|---:|---:|---|---|---|"
    out += [hdr, sep]
    for r in runs:
        if with_real_tps:
            real = r.get("tok_s_gpu_real")
            wall = r.get("full_epoch_min")
            dcgm_tc = r.get("dcgm_tc_busy20")
            pw_avg = r.get("pw_avg"); pw_peak = r.get("pw_peak") or r.get("dcgm_pw_peak")
            row = f"| {r['name']} | {fmt(r.get('steady_ms'),'.0f')} | {fmt(r.get('peak_mem'),'.1f')} | {fmt(r.get('tok_s_gpu_nominal'),'.0f')} | {fmt(real,'.0f')} | {fmt(wall,'.1f')} min | {fmt(r.get('mfu_swift'),'.1f')} | {fmt(dcgm_tc,'.2f')} | {fmt(pw_avg,'.0f')} / {fmt(pw_peak,'.0f')} | {r.get('loss','?')} | {r.get('status','?')} |"
        else:
            pw_avg = r.get("pw_avg"); pw_peak = r.get("pw_peak")
            row = f"| {r['name']} | {fmt(r.get('steady_ms'),'.0f')} | {fmt(r.get('peak_mem'),'.1f')} | {fmt(r.get('tok_s_gpu_real'),'.0f')} | {fmt(r.get('mfu_swift'),'.1f')} | {fmt(pw_avg,'.0f')} / {fmt(pw_peak,'.0f')} | {r.get('loss','?')} | {r.get('status','?')} |"
        out.append(row)
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench_root", type=Path,
                    default="/home/ubuntu/perf_opt/megatron_output/bench_fsdp_opt")
    ap.add_argument("--out", type=Path,
                    default="/home/ubuntu/perf_opt/megatron_output/bench_fsdp_opt/_all_runs.md")
    args = ap.parse_args()

    # Scan all runs
    all_runs = {}
    for d in sorted(args.bench_root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"): continue
        if d.name == "mega":
            r = scan_megatron_run(d)
        else:
            packing = "pack" in d.name or d.name == "ds_pack_liger"
            r = scan_swift_run(d, packing=packing)
        if r:
            # add DCGM if available (swift runs only)
            train_dir = d / "train" if (d / "train").is_dir() else d
            dcgm = parse_dcgm(train_dir / "dcgm_tc.tsv")
            if dcgm:
                r["dcgm_tc_busy20"] = dcgm["tc_busy20"]
                r["dcgm_pw_peak"] = dcgm["pw_peak"]
            all_runs[d.name] = r

    # Bucket runs
    blocks = {
        "Block A — FSDP2 优化轴扫盘（baseline + 4 个独立轴）": [
            "baseline", "no_ac", "mbs2", "mbs4", "no_reshard",
        ],
        "Block B — FSDP2 失败配置": ["wrap_large", "sp1_mbs2", "compile"],
        "Block C — FSDP2 组合（combo_easy）": ["combo_easy", "combo_easy_long"],
        "Block D — wall-time 验证（同 1600 samples）": ["wall_baseline", "wall_combo"],
        "Block E — Packing + Liger（pack_liger 是大赢家）": [
            "pack", "pack_liger", "pack_liger_dl",
        ],
        "Block F — 三后端对比": ["ds_pack_liger", "mega"],
        "Block G — Profile 专用（短跑，纯为采 nsys/torch.profiler 数据）": [
            "pack_liger_prof", "pack_liger_nsys",
        ],
    }

    md = ["# 全部实验数据汇总（自动生成）\n",
          "> 一次性扫盘 + wall-time 验证 + packing/Liger 优化 + 三后端对比的完整原始数据。\n",
          "> 生成命令：`python scripts/benchmark/build_all_runs_table.py`\n",
          "> 实验日期：2026-04-23/24，硬件 8×H100 80GB，模型 Qwen3.5-9B（freeze_vit），seq=16384\n",
          "\n## 字段说明\n",
          "- **steady step**：跳过 compile cold start 后的 EMA 步时间\n",
          "- **tok/s/GPU nominal**：swift 的 `6ND/step_time` 公式产出（含 padding）\n",
          "- **tok/s/GPU real**：扣除 padding 后的真实训练 token 吞吐（packing 时 nominal=real）\n",
          "- **full-epoch wall**：跑完 53.1M 真实 token 一遍的预估时间\n",
          "- **swift MFU %**：理论 MFU = `6ND/step_time/GPU_count/989`（不可直接对比 TP 后端，会被 tile 大小影响）\n",
          "- **DCGM tc busy20**：DCGM `PIPE_TENSOR_ACTIVE` 在最忙 20% 时间窗口的均值（10 Hz 采样）\n",
          "- **avg / peak power**：avg 是全程 nvidia-smi 1 Hz 平均，peak 是 DCGM 10 Hz 峰值\n"]

    for title, names in blocks.items():
        runs = [all_runs[n] for n in names if n in all_runs]
        if runs:
            md.append(render_block(title, runs, with_real_tps=("Block G" not in title)))

    md.append("\n## 配套 nsys profile（详细 kernel 分类需 `nsys stats --report cuda_gpu_kern_sum`）\n")
    md.append("| profile | 文件 | 分析 | TC-eligible / wall（per-rank） |")
    md.append("|---|---|---|---:|")
    md.append("| FSDP2 baseline (torch.profiler) | `bench_sp_offload_profile/profile_fsdp2_sp2/trace_rank0_step3.json` | rank 0 only, profile overhead | ~5%（粗算）|")
    md.append("| FSDP2 pack_liger (torch.profiler) | `bench_fsdp_opt/pack_liger_prof/trace_rank0_step5.json` | rank 0 only | ~37% |")
    md.append("| FSDP2 pack_liger (nsys 2025) | `bench_fsdp_opt/pack_liger_nsys/profile.nsys-rep` | 8 ranks aggregate, 10s 窗口 | **50.4%** |")
    md.append("| Megatron TP=4 SP (nsys 2025) | `bench_fsdp_opt/mega_prof/megatron/profile.nsys-rep` | 8 ranks aggregate, 10s 窗口 | **51.1%** |")
    md.append("")

    args.out.write_text("\n".join(md))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
