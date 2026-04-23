#!/usr/bin/env python3
"""Drop-in replacement for `python -m swift.cli.sft` / `swift sft`.

Applies the transformers 5.5 ↔ swift 4.1.2 Ulysses SP signature shim
(`swift_sp_patch.apply()`) **before** the swift CLI imports the trainer,
so all downstream modules see the patched `SequenceParallel._prepare_flash_attn`.

Usage (matches `swift sft` verbatim; all CLI args passed through):

    python scripts/benchmark/swift_sft_patched.py \
        --model ... --dataset ... --sequence_parallel_size 2 ...
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sibling module importable regardless of cwd
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import swift_sp_patch  # noqa: F401  (side-effect: monkey-patches swift)

# Match swift/cli/sft.py's __main__ block exactly so optional subsystems
# (unsloth / ray / single-device) initialise identically to `swift sft`.
from swift.cli.utils import try_use_single_device_mode  # noqa: E402


def _init_unsloth() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tuner_backend", type=str, default="peft")
    args, _ = parser.parse_known_args()
    if args.tuner_backend == "unsloth":
        import unsloth  # noqa: F401


if __name__ == "__main__":
    try_use_single_device_mode()
    _init_unsloth()
    from swift.ray import try_init_ray  # noqa: E402

    try_init_ray()
    from swift.pipelines import sft_main  # noqa: E402

    sft_main()
