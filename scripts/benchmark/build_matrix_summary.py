#!/usr/bin/env python3
"""Gather all SP/offload matrix reports (DS, FSDP2, Megatron) into one
markdown table + re-parse swift logs using the train_speed(s/it)-based
timing (see docstring in report_swift_sp.py for why we prefer it over the
elapsed_time diff).

Also parses Megatron ms-swift logs (which use `'iteration': 'N/M'` instead of
`'global_step/max_steps'`) and emits a unified report row for each group.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from pathlib import Path

H100_PEAK_TFLOPS = 989.5


def _safe_num(x):
    if isinstance(x, (int, float)) and not math.isnan(x):
        return float(x)
    return None


def parse_swift_sft_log_jsonl(path: Path):
    """Parse swift sft's logging.jsonl (HF trainer format)."""
    records = []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "global_step/max_steps" not in d:
                continue
            try:
                step = int(str(d["global_step/max_steps"]).split("/")[0])
            except (ValueError, KeyError):
                continue
            records.append({
                "step": step,
                "loss": d.get("loss"),
                "memory_gib": d.get("memory(GiB)"),
                "train_speed_s_per_it": _safe_num(d.get("train_speed(s/it)")),
            })
    return records


def parse_megatron_sft_log_jsonl(path: Path):
    """Parse ms-swift megatron's logging.jsonl or stdout log.

    Format: {'loss': ..., 'iteration': '1/5', 'elapsed_time': '2m 19s',
             'memory(GiB)': 40.84, 'train_speed(s/it)': 138.75}
    If logging.jsonl isn't parseable, we fall back to regex-scraping the raw
    train.log for the same dict literals.
    """
    records = []
    # Try the JSONL first (swift writes it under ckpt/v0-*/logging.jsonl).
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "iteration" not in d:
                    continue
                try:
                    step = int(str(d["iteration"]).split("/")[0])
                except (ValueError, KeyError):
                    continue
                records.append({
                    "step": step,
                    "loss": d.get("loss"),
                    "memory_gib": d.get("memory(GiB)"),
                    "train_speed_s_per_it": _safe_num(d.get("train_speed(s/it)")),
                    "elapsed_str": d.get("elapsed_time"),
                })
    return records


def parse_megatron_stdout_log(path: Path):
    """Scrape ms-swift megatron stdout for iteration dict literals."""
    records = []
    # e.g. 'iteration': '5/5', 'elapsed_time': '2m 28s', 'memory(GiB)': 47.62,
    #      'train_speed(s/it)': 29.64321
    pat_iter = re.compile(r"'iteration':\s*'(\d+)/(\d+)'")
    pat_loss = re.compile(r"'loss':\s*([\d.]+|nan)", re.IGNORECASE)
    pat_mem = re.compile(r"'memory\(GiB\)':\s*([\d.]+)")
    pat_spd = re.compile(r"'train_speed\(s/it\)':\s*([\d.]+)")
    pat_elapsed = re.compile(r"'elapsed_time':\s*'([^']+)'")
    with open(path) as f:
        for line in f:
            m = pat_iter.search(line)
            if not m:
                continue
            step = int(m.group(1))
            row = {"step": step}
            mloss = pat_loss.search(line)
            row["loss"] = float(mloss.group(1)) if mloss else None
            mmem = pat_mem.search(line)
            row["memory_gib"] = float(mmem.group(1)) if mmem else None
            mspd = pat_spd.search(line)
            row["train_speed_s_per_it"] = float(mspd.group(1)) if mspd else None
            melap = pat_elapsed.search(line)
            row["elapsed_str"] = melap.group(1) if melap else None
            records.append(row)
    return records


def parse_gpu_log(path: Path):
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def gpu_stats(gpu_records):
    if not gpu_records:
        return {"peak_mem_gb": 0.0, "avg_util_pct": 0.0, "avg_power_w": 0.0,
                "peak_power_w": 0.0}
    mem, util, pw = [], [], []
    for rec in gpu_records:
        for g in rec.get("gpus", []):
            mem.append(g.get("mem_used_mb", 0.0))
            util.append(g.get("util_pct", 0.0))
            pw.append(g.get("power_w", 0.0))
    return {
        "peak_mem_gb": (max(mem) / 1024.0) if mem else 0.0,
        "avg_util_pct": statistics.mean(util) if util else 0.0,
        "avg_power_w": statistics.mean(pw) if pw else 0.0,
        "peak_power_w": max(pw) if pw else 0.0,
    }


def summarise_group(name: str, group_dir: Path, backend: str, params: dict):
    """Produce a normalised row dict for one benchmark group.

    ``params`` keys we care about: sp, tp, pp, gbs, max_len, recompute,
    offload, num_params_trainable.
    """
    # Pick the per-step records. Path depends on backend.
    records = []
    if backend in ("ds", "fsdp2"):
        # ${group_dir}/v0-*/logging.jsonl
        vdirs = sorted(group_dir.glob("v*-*"), reverse=True)
        if vdirs:
            p = vdirs[0] / "logging.jsonl"
            if p.exists():
                records = parse_swift_sft_log_jsonl(p)
    elif backend == "megatron":
        # bench_megatron.sh writes under ckpt/v0-*/logging.jsonl (see cmdline).
        # The wrapper in run_sp_offload_matrix.sh renamed the dir to <name>.raw.
        candidates = list(group_dir.glob("ckpt/v*-*/logging.jsonl")) + \
            list(group_dir.glob("**/logging.jsonl"))
        for p in candidates:
            records = parse_megatron_sft_log_jsonl(p)
            if records:
                break
        if not records:
            # fall back to stdout log
            train_log = group_dir / "train.log"
            if train_log.exists():
                records = parse_megatron_stdout_log(train_log)

    gpu_records = parse_gpu_log(group_dir / "gpu_metrics.jsonl")
    g = gpu_stats(gpu_records)

    # Canonical step time: for DS / FSDP (logging_steps=1) the LAST
    # `train_speed(s/it)` is the running mean and stabilises after ~3 steps.
    # For Megatron (logging_steps=5 by default) we only get 1-2 log records
    # total; `train_speed(s/it)` there is dominated by iter-1 compile cold
    # start. In that case derive steady step time from the elapsed delta.
    last_speed = None
    for r in reversed(records):
        v = r.get("train_speed_s_per_it")
        if v is not None and v > 0:
            last_speed = v
            break

    steady_step_s = None
    if backend == "megatron":
        # Parse elapsed_str in records to derive delta; fall back to
        # last_speed if we only have one record.
        elapsed_pairs = []
        for r in records:
            s = r.get("elapsed_str") or r.get("elapsed_time")
            if s is None:
                continue
            # "2m 19s" -> 139.0
            total = 0.0
            num = ""
            for ch in str(s).strip() + " ":
                if ch.isdigit() or ch == ".":
                    num += ch
                elif ch in "mhsd":
                    if num:
                        v = float(num)
                        if ch == "s": total += v
                        elif ch == "m": total += v * 60
                        elif ch == "h": total += v * 3600
                        elif ch == "d": total += v * 86400
                        num = ""
                elif num:
                    num = ""
            elapsed_pairs.append((r["step"], total))
        if len(elapsed_pairs) >= 2:
            (s1, t1), (s2, t2) = elapsed_pairs[0], elapsed_pairs[-1]
            if s2 > s1 and t2 > t1:
                steady_step_s = (t2 - t1) / (s2 - s1)
        canonical_step_s = steady_step_s or last_speed
    else:
        canonical_step_s = last_speed
    peak_mem_log = max(
        (r.get("memory_gib", 0.0) or 0.0) for r in records
    ) if records else None

    num_params = params.get("num_params_trainable", 8.95e9)
    num_gpus = 8
    max_len = params["max_len"]
    gbs = params["gbs"]
    tokens_per_step = gbs * max_len

    mean_step_ms = (canonical_step_s * 1000.0) if canonical_step_s else None
    tokens_per_sec = (tokens_per_step * 1000.0 / mean_step_ms) if mean_step_ms else None
    achieved_tflops_per_gpu = (
        6 * num_params * tokens_per_sec / 1e12 / num_gpus
        if tokens_per_sec else None
    )
    mfu = (
        achieved_tflops_per_gpu / H100_PEAK_TFLOPS * 100.0
        if achieved_tflops_per_gpu else None
    )

    # First / last loss (only informative; swift SP path has known loss=0 bug)
    loss_first = next((r.get("loss") for r in records if r.get("loss") is not None), None)
    loss_last = next((r.get("loss") for r in reversed(records) if r.get("loss") is not None), None)

    row = {
        "name": name,
        "backend": backend,
        "status": "ok" if records and mean_step_ms else "failed",
        **{k: params[k] for k in params},
        "num_measured_steps": len(records),
        "mean_step_time_ms": round(mean_step_ms, 1) if mean_step_ms else None,
        "tokens_per_sec_per_gpu": round(tokens_per_sec / num_gpus, 1) if tokens_per_sec else None,
        "achieved_tflops_per_gpu": round(achieved_tflops_per_gpu, 1) if achieved_tflops_per_gpu else None,
        "mfu_pct": round(mfu, 2) if mfu is not None else None,
        "peak_mem_gb": round(g["peak_mem_gb"], 2),
        "peak_mem_gb_from_log": round(peak_mem_log, 2) if peak_mem_log else None,
        "avg_gpu_util_pct": round(g["avg_util_pct"], 1),
        "avg_power_w": round(g["avg_power_w"], 1),
        "peak_power_w": round(g["peak_power_w"], 1),
        "loss_first": loss_first,
        "loss_last": loss_last,
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    bench_dir = Path(args.bench_dir)
    COMMON = {"max_len": 16384, "gbs": 8, "num_params_trainable": 8.95e9}

    groups = [
        ("ds_sp2_no_off", "ds",
         {"sp": 2, "offload": "none", "config": "zero3_nopin.json", **COMMON}),
        ("ds_sp2_off_opt", "ds",
         {"sp": 2, "offload": "optimizer (cpu, nopin)",
          "config": "zero3_offload_opt.json", **COMMON}),
        ("ds_sp4_no_off", "ds",
         {"sp": 4, "offload": "none", "config": "zero3_nopin.json", **COMMON}),
        ("fsdp2_sp2", "fsdp2",
         {"sp": 2, "offload": "none (act_ckpt in fsdp_config)",
          "config": "fsdp2 preset", **COMMON}),
        ("fsdp2_sp4", "fsdp2",
         {"sp": 4, "offload": "none (act_ckpt in fsdp_config)",
          "config": "fsdp2 preset", **COMMON}),
        ("megatron_tp4_sp", "megatron",
         {"tp": 4, "pp": 1, "cp": 1, "recompute": "none",
          "config": "mcore-bridge + use_distributed_optimizer",
          "max_len": 16384, "gbs": 8, "num_params_trainable": 8.95e9}),
        ("megatron_tp2_sp_sel", "megatron",
         {"tp": 2, "pp": 1, "cp": 1, "recompute": "selective",
          "config": "mcore-bridge + use_distributed_optimizer",
          "max_len": 16384, "gbs": 8, "num_params_trainable": 8.95e9}),
    ]

    rows = []
    for name, backend, params in groups:
        # Megatron groups are under <name>.raw (my matrix runner tagged them
        # that way to distinguish from the non-megatron outputs).
        primary = bench_dir / name
        megatron_raw = bench_dir / f"{name}.raw"
        group_dir = megatron_raw if backend == "megatron" and megatron_raw.exists() else primary
        if not group_dir.exists():
            rows.append({"name": name, "backend": backend, "status": "missing"})
            continue
        rows.append(summarise_group(name, group_dir, backend, params))

    with open(args.out_json, "w") as f:
        json.dump(rows, f, indent=2)

    # ---- Markdown table ----
    def _fmt(v, nd=1):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)

    with open(args.out_md, "w") as f:
        f.write("# Qwen3.5-9B SP/CP/Offload Matrix — 汇总\n\n")
        f.write("单机 8 × H100 80GB，seq=16384 × MBS=1 × GBS=8（GAS=1，短跑 5 iter，取 warmup 之后 last `train_speed(s/it)` 作为均步长）。\n\n")
        f.write("| Group | Backend | Parallel | Offload | peak_mem (GB, smi) | peak_mem (GB, swift log) | step_time (s) | tok/s/GPU | TFLOPs/GPU | MFU % | Util % | Power (W avg/peak) | Status |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            if r.get("status") == "missing":
                f.write(f"| {r['name']} | {r['backend']} | — | — | — | — | — | — | — | — | — | — | missing |\n")
                continue
            parallel_cells = []
            if "sp" in r:
                parallel_cells.append(f"SP={r['sp']}")
            if "tp" in r:
                parallel_cells.append(f"TP={r['tp']}")
            if "cp" in r:
                parallel_cells.append(f"CP={r['cp']}")
            if "recompute" in r:
                parallel_cells.append(f"recompute={r['recompute']}")
            parallel = ", ".join(parallel_cells)
            step_ms = r.get("mean_step_time_ms")
            step_s = step_ms / 1000.0 if step_ms else None
            f.write(
                f"| {r['name']} | {r['backend']} | {parallel} | "
                f"{r.get('offload', '—')} | {_fmt(r.get('peak_mem_gb'))} | "
                f"{_fmt(r.get('peak_mem_gb_from_log'))} | {_fmt(step_s, 2)} | "
                f"{_fmt(r.get('tokens_per_sec_per_gpu'))} | "
                f"{_fmt(r.get('achieved_tflops_per_gpu'))} | "
                f"{_fmt(r.get('mfu_pct'))} | {_fmt(r.get('avg_gpu_util_pct'))} | "
                f"{_fmt(r.get('avg_power_w'), 0)}/{_fmt(r.get('peak_power_w'), 0)} | "
                f"{r.get('status', '—')} |\n"
            )
        f.write("\n### 注意事项\n\n")
        f.write("* swift 4.1.2 × transformers 5.5.4 的 Ulysses SP 通路有确定性 loss=0 / grad_norm=nan 漂移（第 3 步起开始）；5 步短跑能稳定收到 peak_mem + step_time 数据，15+ 步会让 NaN 累积触发 CUDA illegal memory access。详见 `docs/sp_offload_benchmark_report.md` §5。\n")
        f.write("* `peak_mem (GB, smi)` 来自 nvidia-smi 1Hz 采样；`peak_mem (GB, swift log)` 是 swift 在每步尾部记录的 max allocated 值。两者差值是短期未被 smi 抓到的峰值毛刺。\n")
        f.write("* Megatron TP=2 SP `RECOMPUTE=selective` 组 OOM（Triton 报 CUDA OOM），保留在表中作为 \"TP 切得太少\" 的反例。\n")
        f.write("* FSDP2 + SP=4 组触发 swift SP label-sharding 的 `cur_target >= 0 && cur_target < n_classes` 越界，标 failed。同 swift × tf 版本下 SP=2 正常。\n")

    print("wrote:")
    print(" ", args.out_json)
    print(" ", args.out_md)


if __name__ == "__main__":
    main()
