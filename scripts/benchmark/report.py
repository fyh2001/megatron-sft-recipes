#!/usr/bin/env python3
"""Benchmark report generator.

Parses benchmark logs (per-step jsonl from FSDP or training stdout from Megatron),
GPU metrics from gpu_monitor.py, and computes:
  - Throughput (tokens/sec)
  - Step time (avg, median, p99)
  - MFU (Model FLOPs Utilization)
  - GPU utilization and memory

Usage:
    # Single framework report
    python scripts/benchmark/report.py \\
        --framework fsdp \\
        --bench_log benchmark_output/fsdp/bench.jsonl \\
        --gpu_log benchmark_output/fsdp/gpu_metrics.jsonl \\
        --warmup_steps 20 --num_params 12.2e9 --num_gpus 8

    # Compare two frameworks
    python scripts/benchmark/report.py --compare \\
        --fsdp_dir benchmark_output/fsdp \\
        --megatron_dir benchmark_output/megatron \\
        --num_params 12.2e9 --num_gpus 8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

# Peak BF16 TFLOPS (dense, tensor core) per GPU
GPU_PEAK_TFLOPS = {
    "h100": 989.5,
    "a100": 312.0,
    "h800": 989.5,
    "a800": 312.0,
    "l40s": 362.0,
}


def parse_fsdp_bench_log(path: str, warmup_steps: int) -> list[dict]:
    """Parse FSDP benchmark jsonl log, skip warmup steps."""
    steps = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            if record["step"] > warmup_steps:
                steps.append(record)
    return steps


def parse_megatron_train_log(path: str, warmup_steps: int) -> list[dict]:
    """Parse Megatron/swift training log for per-step metrics.

    Megatron-core logs lines like:
        [2024-xx-xx] iteration      25/ ... elapsed time per iteration (ms): 2640.3 ...
    ms-swift (HF backend) logs:
        {'loss': 2.341, 'grad_norm': ..., 'learning_rate': ..., 'epoch': 0.01}
        or: step=25 ... loss=2.341 ... tok/s=12450
    """
    steps = []
    step_num = 0

    # Pattern for megatron-core iteration log
    mcore_pat = re.compile(
        r"iteration\s+(\d+)/.*?elapsed time per iteration \(ms\):\s*([\d.]+)"
    )
    # Pattern for megatron-core tokens-per-sec
    tps_pat = re.compile(r"tokens-per-sec(?:-per-gpu)?:\s*([\d.]+)")
    # Pattern for HF trainer {'loss': ..., ...} dicts
    hf_dict_pat = re.compile(r"\{'loss':\s*([\d.]+).*?'epoch':\s*([\d.]+)")
    # Pattern for custom step= log format
    step_log_pat = re.compile(
        r"step=\s*(\d+).*?loss=([\d.]+).*?tok/s=([\d.]+)"
    )

    with open(path) as f:
        for line in f:
            # Try megatron-core format
            m = mcore_pat.search(line)
            if m:
                step_num = int(m.group(1))
                step_time_ms = float(m.group(2))
                if step_num > warmup_steps:
                    entry = {"step": step_num, "step_time_ms": step_time_ms}
                    tps_m = tps_pat.search(line)
                    if tps_m:
                        entry["tokens_per_sec_per_gpu"] = float(tps_m.group(1))
                    steps.append(entry)
                continue

            # Try HF trainer format (swift sft logs)
            m = hf_dict_pat.search(line)
            if m:
                step_num += 1
                if step_num > warmup_steps:
                    steps.append({
                        "step": step_num,
                        "loss": float(m.group(1)),
                    })
                continue

            # Try step= format
            m = step_log_pat.search(line)
            if m:
                s = int(m.group(1))
                if s > warmup_steps:
                    steps.append({
                        "step": s,
                        "loss": float(m.group(2)),
                        "tokens_per_sec": float(m.group(3)),
                    })

    return steps


def parse_gpu_log(path: str) -> list[dict]:
    """Parse GPU monitor jsonl log."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compute_gpu_stats(gpu_records: list[dict]) -> dict:
    """Compute aggregate GPU stats from monitor log."""
    if not gpu_records:
        return {"avg_util_pct": 0, "peak_mem_gb": 0, "mem_total_gb": 0, "avg_power_w": 0}

    all_utils = []
    all_mem_used = []
    all_mem_total = []
    all_power = []

    for record in gpu_records:
        for gpu in record["gpus"]:
            all_utils.append(gpu["util_pct"])
            all_mem_used.append(gpu["mem_used_mb"])
            all_mem_total.append(gpu["mem_total_mb"])
            all_power.append(gpu["power_w"])

    return {
        "avg_util_pct": statistics.mean(all_utils) if all_utils else 0,
        "peak_mem_gb": max(all_mem_used) / 1024 if all_mem_used else 0,
        "mem_total_gb": max(all_mem_total) / 1024 if all_mem_total else 0,
        "avg_power_w": statistics.mean(all_power) if all_power else 0,
    }


def compute_mfu(
    tokens_per_sec: float, num_params: float, num_gpus: int, gpu_type: str
) -> float:
    """Compute Model FLOPs Utilization.

    MFU = actual_flops / (peak_flops * num_gpus)
    actual_flops = 6 * num_params * tokens_per_sec  (fwd + bwd for transformer)
    """
    peak = GPU_PEAK_TFLOPS.get(gpu_type, 989.5)
    actual_tflops = 6 * num_params * tokens_per_sec / 1e12
    theoretical_peak = peak * num_gpus
    if theoretical_peak == 0:
        return 0.0
    return actual_tflops / theoretical_peak * 100


def generate_report(
    framework: str,
    steps: list[dict],
    gpu_stats: dict,
    num_params: float,
    num_gpus: int,
    gpu_type: str,
) -> dict:
    """Generate a structured benchmark report."""

    report = {"framework": framework, "num_measured_steps": len(steps)}

    if not steps:
        print(f"WARNING: No measured steps found for {framework}", file=sys.stderr)
        return report

    # Step time stats
    step_times = [s["step_time_ms"] for s in steps if "step_time_ms" in s]
    if step_times:
        report["step_time_ms_avg"] = round(statistics.mean(step_times), 1)
        report["step_time_ms_median"] = round(statistics.median(step_times), 1)
        report["step_time_ms_p99"] = round(
            sorted(step_times)[int(len(step_times) * 0.99)], 1
        )

    # Throughput from FSDP bench log (has per-step token counts)
    tokens_list = [s["tokens"] for s in steps if "tokens" in s]
    if tokens_list and step_times:
        total_tokens = sum(tokens_list)
        total_time_s = sum(step_times) / 1000
        tokens_per_sec = total_tokens / total_time_s
        report["throughput_tok_per_sec"] = round(tokens_per_sec)
        report["throughput_samples_per_sec"] = round(
            len(steps) / total_time_s, 2
        )
        report["mfu_pct"] = round(
            compute_mfu(tokens_per_sec, num_params, num_gpus, gpu_type), 2
        )
    elif step_times:
        # Megatron: estimate from tokens_per_sec_per_gpu if available
        tps_values = [
            s["tokens_per_sec_per_gpu"] * num_gpus
            for s in steps
            if "tokens_per_sec_per_gpu" in s
        ]
        if tps_values:
            avg_tps = statistics.mean(tps_values)
            report["throughput_tok_per_sec"] = round(avg_tps)
            report["mfu_pct"] = round(
                compute_mfu(avg_tps, num_params, num_gpus, gpu_type), 2
            )

    # GPU stats
    report["gpu_util_pct"] = round(gpu_stats["avg_util_pct"], 1)
    report["peak_mem_gb"] = round(gpu_stats["peak_mem_gb"], 1)
    report["mem_total_gb"] = round(gpu_stats["mem_total_gb"], 1)
    if gpu_stats["mem_total_gb"] > 0:
        report["mem_efficiency_pct"] = round(
            gpu_stats["peak_mem_gb"] / gpu_stats["mem_total_gb"] * 100, 1
        )
    report["avg_power_w"] = round(gpu_stats["avg_power_w"], 0)

    return report


def print_single_report(report: dict):
    """Print a formatted single-framework report."""
    fw = report.get("framework", "unknown")
    print(f"\n{'='*50}")
    print(f"  Benchmark Report: {fw.upper()}")
    print(f"{'='*50}")
    print(f"  Measured steps     : {report.get('num_measured_steps', 'N/A')}")
    if "step_time_ms_avg" in report:
        print(f"  Step time (avg)    : {report['step_time_ms_avg']:.1f} ms")
        print(f"  Step time (median) : {report['step_time_ms_median']:.1f} ms")
        print(f"  Step time (p99)    : {report['step_time_ms_p99']:.1f} ms")
    if "throughput_tok_per_sec" in report:
        print(f"  Throughput         : {report['throughput_tok_per_sec']:,} tok/s")
    if "throughput_samples_per_sec" in report:
        print(f"  Samples/sec        : {report['throughput_samples_per_sec']:.2f}")
    if "mfu_pct" in report:
        print(f"  MFU                : {report['mfu_pct']:.2f}%")
    print(f"  GPU Utilization    : {report.get('gpu_util_pct', 'N/A')}%")
    print(f"  Peak GPU Memory    : {report.get('peak_mem_gb', 'N/A')} GB")
    if "mem_efficiency_pct" in report:
        print(f"  Memory Efficiency  : {report['mem_efficiency_pct']:.1f}%")
    print(f"  Avg Power Draw     : {report.get('avg_power_w', 'N/A')} W")
    print(f"{'='*50}\n")


def print_comparison(fsdp_report: dict, megatron_report: dict):
    """Print side-by-side comparison table."""
    print(f"\n{'='*62}")
    print("  Performance Comparison: Megatron vs FSDP2+compile")
    print(f"{'='*62}")
    header = f"  {'Metric':<22s} {'Megatron':>12s} {'FSDP2+compile':>14s} {'Delta':>10s}"
    print(header)
    print(f"  {'-'*58}")

    def row(label, key, unit="", fmt=".1f", higher_better=True):
        mv = megatron_report.get(key)
        fv = fsdp_report.get(key)
        ms = f"{mv:{fmt}}{unit}" if mv is not None else "N/A"
        fs = f"{fv:{fmt}}{unit}" if fv is not None else "N/A"
        delta = ""
        if mv is not None and fv is not None and mv != 0:
            pct = (fv - mv) / abs(mv) * 100
            sign = "+" if pct > 0 else ""
            delta = f"{sign}{pct:.1f}%"
        print(f"  {label:<22s} {ms:>12s} {fs:>14s} {delta:>10s}")

    row("Step time (ms)", "step_time_ms_avg", " ms", higher_better=False)
    row("Throughput (tok/s)", "throughput_tok_per_sec", "", fmt=",d")
    row("MFU (%)", "mfu_pct", "%", fmt=".2f")
    row("GPU Util (%)", "gpu_util_pct", "%")
    row("Peak Memory (GB)", "peak_mem_gb", " GB")
    row("Memory Efficiency", "mem_efficiency_pct", "%")
    row("Avg Power (W)", "avg_power_w", " W", fmt=".0f")
    print(f"  {'-'*58}")
    print(f"{'='*62}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode")

    # Single framework mode
    single = sub.add_parser("single", help="Generate report for one framework")
    single.add_argument("--framework", required=True, choices=["fsdp", "megatron"])
    single.add_argument("--bench_log", type=str, default=None,
                        help="FSDP bench.jsonl path")
    single.add_argument("--train_log", type=str, default=None,
                        help="Megatron train.log path")
    single.add_argument("--gpu_log", type=str, required=True)
    single.add_argument("--warmup_steps", type=int, default=20)
    single.add_argument("--num_params", type=float, required=True)
    single.add_argument("--num_gpus", type=int, default=8)
    single.add_argument("--gpu_type", type=str, default="h100",
                        choices=list(GPU_PEAK_TFLOPS.keys()))
    single.add_argument("--output", type=str, default=None)

    # Compare mode
    comp = sub.add_parser("compare", help="Compare two frameworks")
    comp.add_argument("--fsdp_dir", type=str, required=True)
    comp.add_argument("--megatron_dir", type=str, required=True)
    comp.add_argument("--warmup_steps", type=int, default=20)
    comp.add_argument("--num_params", type=float, required=True)
    comp.add_argument("--num_gpus", type=int, default=8)
    comp.add_argument("--gpu_type", type=str, default="h100",
                        choices=list(GPU_PEAK_TFLOPS.keys()))

    # Also support flat args for backward compat with bench scripts
    parser.add_argument("--framework", choices=["fsdp", "megatron"], default=None)
    parser.add_argument("--bench_log", type=str, default=None)
    parser.add_argument("--train_log", type=str, default=None)
    parser.add_argument("--gpu_log", type=str, default=None)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--num_params", type=float, default=12.2e9)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--gpu_type", type=str, default="h100")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fsdp_dir", type=str, default=None)
    parser.add_argument("--megatron_dir", type=str, default=None)

    args = parser.parse_args()

    # Handle compare mode
    if args.mode == "compare" or (args.fsdp_dir and args.megatron_dir):
        fsdp_dir = args.fsdp_dir
        mega_dir = args.megatron_dir

        fsdp_steps = parse_fsdp_bench_log(
            os.path.join(fsdp_dir, "bench.jsonl"), args.warmup_steps
        )
        fsdp_gpu = compute_gpu_stats(
            parse_gpu_log(os.path.join(fsdp_dir, "gpu_metrics.jsonl"))
        )
        fsdp_report = generate_report(
            "fsdp", fsdp_steps, fsdp_gpu,
            args.num_params, args.num_gpus, args.gpu_type
        )

        mega_steps = parse_megatron_train_log(
            os.path.join(mega_dir, "train.log"), args.warmup_steps
        )
        mega_gpu = compute_gpu_stats(
            parse_gpu_log(os.path.join(mega_dir, "gpu_metrics.jsonl"))
        )
        mega_report = generate_report(
            "megatron", mega_steps, mega_gpu,
            args.num_params, args.num_gpus, args.gpu_type
        )

        print_single_report(fsdp_report)
        print_single_report(mega_report)
        print_comparison(fsdp_report, mega_report)
        return

    # Single framework mode
    framework = args.framework
    if not framework:
        parser.error("Must specify --framework or use compare mode")

    if framework == "fsdp":
        bench_log = args.bench_log
        if not bench_log:
            parser.error("--bench_log required for FSDP")
        steps = parse_fsdp_bench_log(bench_log, args.warmup_steps)
    else:
        train_log = args.train_log
        if not train_log:
            parser.error("--train_log required for Megatron")
        steps = parse_megatron_train_log(train_log, args.warmup_steps)

    gpu_records = parse_gpu_log(args.gpu_log) if args.gpu_log else []
    gpu_stats = compute_gpu_stats(gpu_records)

    report = generate_report(
        framework, steps, gpu_stats,
        args.num_params, args.num_gpus, args.gpu_type
    )

    print_single_report(report)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report JSON saved to {args.output}")


if __name__ == "__main__":
    main()
