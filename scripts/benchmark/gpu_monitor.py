#!/usr/bin/env python3
"""Background GPU metrics collector.

Polls nvidia-smi at 1-second intervals and writes per-GPU metrics to a jsonl file.
Run as a background process during benchmarks; kill with SIGTERM when done.

Usage:
    python scripts/benchmark/gpu_monitor.py --output /tmp/gpu_metrics.jsonl &
    # ... run benchmark ...
    kill $!
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time

running = True


def handle_signal(signum, frame):
    global running
    running = False


def query_gpus() -> list[dict]:
    fields = "index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    if result.returncode != 0:
        return []

    gpus = []
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        gpus.append({
            "gpu_id": int(parts[0]),
            "util_pct": float(parts[1]) if parts[1] != "[N/A]" else 0.0,
            "mem_used_mb": float(parts[2]) if parts[2] != "[N/A]" else 0.0,
            "mem_total_mb": float(parts[3]) if parts[3] != "[N/A]" else 0.0,
            "power_w": float(parts[4]) if parts[4] != "[N/A]" else 0.0,
            "temp_c": float(parts[5]) if parts[5] != "[N/A]" else 0.0,
        })
    return gpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    with open(args.output, "w") as f:
        while running:
            gpus = query_gpus()
            if gpus:
                record = {"timestamp": time.time(), "gpus": gpus}
                f.write(json.dumps(record) + "\n")
                f.flush()
            time.sleep(args.interval)

    sys.exit(0)


if __name__ == "__main__":
    main()
