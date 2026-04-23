#!/usr/bin/env python3
"""Scrape DCGM profile metrics from a Prometheus endpoint.

Sample interval defaults to 0.1 s (10 Hz) so we can resolve per-step bursts.
Override with env `DCGM_SCRAPE_INTERVAL_S` (float seconds).

Usage::

    python dcgm_scrape.py <out.tsv> [url]
"""
from __future__ import annotations

import os
import re
import sys
import time
import urllib.request

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dcgm_tc.tsv"
URL = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:9500/metrics"
INTERVAL = float(os.environ.get("DCGM_SCRAPE_INTERVAL_S", "0.1"))

PATTERNS = {
    "tc": re.compile(r'^DCGM_FI_PROF_PIPE_TENSOR_ACTIVE\{[^}]*gpu="(\d+)"[^}]*\}\s+(\S+)'),
    "dram": re.compile(r'^DCGM_FI_PROF_DRAM_ACTIVE\{[^}]*gpu="(\d+)"[^}]*\}\s+(\S+)'),
    "gr": re.compile(r'^DCGM_FI_PROF_GR_ENGINE_ACTIVE\{[^}]*gpu="(\d+)"[^}]*\}\s+(\S+)'),
    "pw": re.compile(r'^DCGM_FI_DEV_POWER_USAGE\{[^}]*gpu="(\d+)"[^}]*\}\s+(\S+)'),
}

with open(OUT, "w") as f:
    f.write("ts\tgpu\ttc_active\tdram_active\tgr_active\tpower_w\n")
    while True:
        ts = time.time()
        try:
            with urllib.request.urlopen(URL, timeout=2) as r:
                body = r.read().decode("utf-8", "ignore")
        except Exception:
            time.sleep(INTERVAL)
            continue
        vals = {k: {} for k in PATTERNS}
        for line in body.splitlines():
            for k, p in PATTERNS.items():
                m = p.match(line)
                if m:
                    vals[k][int(m.group(1))] = m.group(2)
                    break
        for gpu in range(8):
            f.write(
                f"{ts:.3f}\t{gpu}\t"
                f"{vals['tc'].get(gpu, '')}\t"
                f"{vals['dram'].get(gpu, '')}\t"
                f"{vals['gr'].get(gpu, '')}\t"
                f"{vals['pw'].get(gpu, '')}\n"
            )
        f.flush()
        time.sleep(INTERVAL)
