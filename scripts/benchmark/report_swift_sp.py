#!/usr/bin/env python3
"""Report generator for the ms-swift `swift sft` SP benchmark runs.

Reads:
  - swift's per-step logging.jsonl (see swift.trainers.mixin). Each line is a
    dict with keys like `loss`, `grad_norm`, `learning_rate`, `token_acc`,
    `epoch`, `global_step/max_steps` ('1/15'), `elapsed_time` ('42s'),
    `memory(GiB)` (peak across all GPUs, reported by swift each log step),
    `train_speed(s/it)` (rolling average step time, seconds).
  - gpu_monitor.py's jsonl log (per-GPU util / memory / power).

Produces:
  - bench.jsonl:  one line per OPTIMIZER step (post-warmup) with step timing
                  and memory, matching the layout that FSDP bench_log writes.
  - report.json:  aggregate summary (avg/median step time, p99, peak mem,
                  avg util/power, tokens/sec, MFU).

This parser is deliberately independent from the existing
`scripts/benchmark/report.py` because swift sft + FSDP2/DS uses a different
log format (kwargs-only JSON records, no `iteration:` string) and conflating
both dialects inside one regex gauntlet has caused missed-metric bugs
before.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics

# BF16 dense TC TFLOPS per H100 SXM (H200 matches).
H100_PEAK_TFLOPS = 989.5


def _parse_elapsed(s: str | int | float) -> float:
    """swift's 'elapsed_time' is '42s' or '1m 23s' — normalise to seconds."""
    if isinstance(s, (int, float)):
        return float(s)
    s = s.strip()
    total = 0.0
    num = ""
    for ch in s:
        if ch.isdigit() or ch == ".":
            num += ch
        elif ch in ("m", "s", "h", "d"):
            if num:
                v = float(num)
                if ch == "s":
                    total += v
                elif ch == "m":
                    total += v * 60
                elif ch == "h":
                    total += v * 3600
                elif ch == "d":
                    total += v * 86400
                num = ""
    if num:
        total += float(num)
    return total


def parse_logging_jsonl(path: str):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # swift writes bare `NaN` tokens; stdlib json accepts these
                # when parse_constant isn't overridden.
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Skip trailing summary records (no per-step entries).
            if "global_step/max_steps" not in d:
                continue
            try:
                step_str = d["global_step/max_steps"]
                step = int(str(step_str).split("/")[0])
            except (ValueError, KeyError):
                continue
            entry = {
                "step": step,
                "loss": d.get("loss"),
                "grad_norm": d.get("grad_norm"),
                "learning_rate": d.get("learning_rate"),
                "token_acc": d.get("token_acc"),
                "elapsed_s": _parse_elapsed(d.get("elapsed_time", "0")),
                "elapsed_time_s": _parse_elapsed(d.get("elapsed_time", "0")),
                "memory_gib_log": d.get("memory(GiB)"),
                "train_speed_s_per_it": d.get("train_speed(s/it)"),
            }
            records.append(entry)
    return records


def step_times_from_elapsed(records: list[dict]) -> list[float]:
    """Compute per-step wallclock in ms by diffing 'elapsed_s' between steps."""
    times = []
    prev = 0.0
    for r in records:
        step_ms = max(r["elapsed_s"] - prev, 0.0) * 1000.0
        r["step_time_ms"] = step_ms
        times.append(step_ms)
        prev = r["elapsed_s"]
    return times


def parse_gpu_log(path: str):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def compute_gpu_stats(records: list[dict]) -> dict:
    if not records:
        return {
            "avg_util_pct": 0.0,
            "peak_mem_gb": 0.0,
            "mem_total_gb": 0.0,
            "avg_power_w": 0.0,
            "peak_power_w": 0.0,
        }
    utils, mems_used, mems_total, powers = [], [], [], []
    for rec in records:
        for gpu in rec.get("gpus", []):
            utils.append(gpu.get("util_pct", 0.0))
            mems_used.append(gpu.get("mem_used_mb", 0.0))
            mems_total.append(gpu.get("mem_total_mb", 0.0))
            powers.append(gpu.get("power_w", 0.0))
    return {
        "avg_util_pct": statistics.mean(utils) if utils else 0.0,
        "peak_mem_gb": (max(mems_used) / 1024.0) if mems_used else 0.0,
        "mem_total_gb": (max(mems_total) / 1024.0) if mems_total else 0.0,
        "avg_power_w": statistics.mean(powers) if powers else 0.0,
        "peak_power_w": max(powers) if powers else 0.0,
    }


def safe_median(xs):
    xs = [x for x in xs if x is not None and (isinstance(x, (int, float)))]
    return statistics.median(xs) if xs else 0.0


def safe_mean(xs):
    xs = [x for x in xs if x is not None and (isinstance(x, (int, float)))]
    return statistics.mean(xs) if xs else 0.0


def safe_p99(xs):
    xs = sorted(x for x in xs if x is not None and (isinstance(x, (int, float))))
    if not xs:
        return 0.0
    idx = max(0, int(math.ceil(0.99 * len(xs))) - 1)
    return xs[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logging_jsonl", required=True)
    ap.add_argument("--gpu_log", required=True)
    ap.add_argument("--warmup_steps", type=int, default=5)
    ap.add_argument("--num_gpus", type=int, default=8)
    ap.add_argument("--gbs", type=int, required=True)
    ap.add_argument("--max_len", type=int, required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--sp", type=int, default=1)
    ap.add_argument(
        "--num_params",
        type=float,
        default=8.95e9,
        help="trainable params (Qwen3.5-9B with VIT frozen ≈ 8.95 B)",
    )
    ap.add_argument(
        "--num_active_params",
        type=float,
        default=None,
        help="Active params per token (for MoE; e.g. gemma4-26B-A4B = 3.8e9). "
             "If set, also reports MoE-correct MFU (active-params based).",
    )
    ap.add_argument(
        "--dataset_size",
        type=int,
        default=18819,
        help="Number of training samples in the dataset; used to estimate "
             "full-epoch wall time. Default 18819 = sft-data/train.jsonl.",
    )
    ap.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="gradient_accumulation_steps; with this we can compute per-micro-step "
             "time and reconcile FSDP2 (typ. 1) vs DS prod (typ. 16).",
    )
    ap.add_argument("--bench_jsonl_out", required=True)
    ap.add_argument("--report_out", required=True)
    args = ap.parse_args()

    records = parse_logging_jsonl(args.logging_jsonl)
    if not records:
        print(f"[report_swift_sp] no per-step records in {args.logging_jsonl}")
        return 1

    step_times_from_elapsed(records)
    # First record's step_time_ms equals its elapsed, i.e. includes compile
    # cold start. We want the warmup filter to catch this — it will, because
    # step=1..WARMUP are dropped anyway.
    post_warmup = [r for r in records if r["step"] > args.warmup_steps]

    # --- Write bench.jsonl ------------------------------------------------
    with open(args.bench_jsonl_out, "w") as f:
        for r in records:
            out = {
                "step": r["step"],
                "step_time_ms": r.get("step_time_ms", 0.0),
                "loss": r.get("loss"),
                "grad_norm": r.get("grad_norm"),
                "token_acc": r.get("token_acc"),
                "memory_gib_log": r.get("memory_gib_log"),
                "train_speed_s_per_it": r.get("train_speed_s_per_it"),
            }
            # json defaults disallow NaN in strict mode; swift logs NaN freely
            # and we want the bench_jsonl to round-trip. Let allow_nan=True
            # (the Python stdlib default) keep `NaN`/`Infinity` literals.
            f.write(json.dumps(out) + "\n")

    # --- Aggregate --------------------------------------------------------
    step_ms = [r.get("step_time_ms", 0.0) for r in post_warmup if r.get("step_time_ms", 0.0) > 0]

    gpu_records = parse_gpu_log(args.gpu_log)
    gpu_stats = compute_gpu_stats(gpu_records)

    # swift's `elapsed_time` is logged as an **integer seconds** string
    # ("4s", "1m 5s"), so the per-step diff is quantised at 1 s. For 5-step
    # runs that quantisation leaks straight into mean_step_time_ms and makes
    # MFU jump around >50%. The more reliable per-run timing is the last
    # `train_speed(s/it)` value, which swift's `Trainer._maybe_log_save_evaluate`
    # computes from the *precise* elapsed wallclock divided by step count.
    # We use that as the canonical mean_step_time_ms, keep the diff-based
    # numbers as secondary fields for drill-down.
    last_speed_s = None
    for r in reversed(records):
        v = r.get("train_speed_s_per_it")
        if isinstance(v, (int, float)) and v > 0 and not math.isnan(v):
            last_speed_s = float(v)
            break
    if last_speed_s is not None:
        mean_step_ms = last_speed_s * 1000.0
    else:
        mean_step_ms = safe_mean(step_ms)
    median_step_ms = safe_median(step_ms) or mean_step_ms
    p99_step_ms = safe_p99(step_ms) or mean_step_ms

    tokens_per_step = args.gbs * args.max_len
    tokens_per_sec = (tokens_per_step * 1000.0 / mean_step_ms) if mean_step_ms > 0 else 0.0
    # MFU: 6 * N * T tokens / sec  vs peak
    achieved_tflops = 6 * args.num_params * tokens_per_sec / 1e12
    peak_cluster_tflops = H100_PEAK_TFLOPS * args.num_gpus
    mfu = (achieved_tflops / peak_cluster_tflops * 100.0) if peak_cluster_tflops > 0 else 0.0

    # MoE-correct MFU (uses active params per token, not total trainable)
    mfu_active = None
    achieved_tflops_active = None
    if args.num_active_params is not None:
        achieved_tflops_active = 6 * args.num_active_params * tokens_per_sec / 1e12
        mfu_active = (achieved_tflops_active / peak_cluster_tflops * 100.0) if peak_cluster_tflops > 0 else 0.0

    # Per micro-step (within one optimizer step)
    micro_step_ms = mean_step_ms / max(args.grad_accum, 1)

    # **REAL** total wall time (from actual run start to last step's elapsed_time)
    # NOT an estimate — pulled directly from swift's logged elapsed_time field
    total_wall_s_real = None
    if records:
        last_elapsed = None
        for r in reversed(records):
            v = r.get("elapsed_time_s")
            if isinstance(v, (int, float)) and v > 0:
                last_elapsed = float(v)
                break
        if last_elapsed is not None:
            total_wall_s_real = last_elapsed
    total_wall_min_real = (total_wall_s_real / 60.0) if total_wall_s_real else None
    total_wall_h_real = (total_wall_s_real / 3600.0) if total_wall_s_real else None
    actual_steps_completed = records[-1]["step"] if records else 0

    # Steps_per_epoch is purely informational (how many steps a full epoch would be)
    steps_per_epoch = max(1, math.ceil(args.dataset_size / args.gbs))

    peak_mem_from_log = max(
        (r.get("memory_gib_log", 0.0) or 0.0) for r in records
    ) if records else 0.0

    report = {
        "framework": f"swift_sft_{args.backend}_sp{args.sp}",
        "backend": args.backend,
        "sequence_parallel_size": args.sp,
        "gbs": args.gbs,
        "grad_accum": args.grad_accum,
        "max_len": args.max_len,
        "tokens_per_step": tokens_per_step,
        "dataset_size": args.dataset_size,
        "warmup_steps": args.warmup_steps,
        "num_measured_steps": len(step_ms),
        "mean_step_time_ms": round(mean_step_ms, 2),
        "median_step_time_ms": round(median_step_ms, 2),
        "p99_step_time_ms": round(p99_step_ms, 2),
        "micro_step_time_ms": round(micro_step_ms, 2),
        "steps_per_epoch_for_dataset": steps_per_epoch,
        "actual_steps_completed": actual_steps_completed,
        "actual_total_wall_s": round(total_wall_s_real, 1) if total_wall_s_real else None,
        "actual_total_wall_min": round(total_wall_min_real, 1) if total_wall_min_real else None,
        "actual_total_wall_human": (
            f"{int(total_wall_min_real // 60)}h {int(total_wall_min_real % 60)}m"
            if total_wall_min_real else None
        ),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "tokens_per_sec_per_gpu": round(tokens_per_sec / args.num_gpus, 1),
        "achieved_tflops_per_gpu": round(achieved_tflops / args.num_gpus, 1),
        "mfu_pct": round(mfu, 2),
        "mfu_pct_active_params": round(mfu_active, 2) if mfu_active is not None else None,
        "achieved_tflops_per_gpu_active": round(achieved_tflops_active / args.num_gpus, 1) if achieved_tflops_active is not None else None,
        "peak_mem_gb": round(gpu_stats["peak_mem_gb"], 2),
        "peak_mem_gib_from_swift_log": round(peak_mem_from_log, 2),
        "avg_gpu_util_pct": round(gpu_stats["avg_util_pct"], 1),
        "avg_power_w": round(gpu_stats["avg_power_w"], 1),
        "peak_power_w": round(gpu_stats["peak_power_w"], 1),
        "num_params_trainable": args.num_params,
        "num_active_params": args.num_active_params,
        "loss_first_step": post_warmup[0]["loss"] if post_warmup else None,
        "loss_last_step": post_warmup[-1]["loss"] if post_warmup else None,
        "bench_jsonl": args.bench_jsonl_out,
    }

    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2)

    # Pretty print
    print(f"\n=== Report for {report['framework']} ===")
    for k, v in report.items():
        if k != "bench_jsonl":
            print(f"  {k:32s} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
