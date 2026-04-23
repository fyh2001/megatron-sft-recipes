#!/usr/bin/env python3
"""Launcher wrapper: runs `swift sft` with a torch.profiler callback injected.

Env vars:
  TORCH_PROFILE_STEP   int, target global_step to profile (default 3)
  TORCH_PROFILE_OUT    directory to write trace + kernel table (required)
  TORCH_PROFILE_RANK   which rank profiles (default 0; -1 = all ranks)

Everything else is passed through to `swift sft`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import torch_profile_callback  # noqa: E402

_target_step = int(os.environ.get("TORCH_PROFILE_STEP", "3"))
_out_dir = os.environ.get("TORCH_PROFILE_OUT")
_rank_only = int(os.environ.get("TORCH_PROFILE_RANK", "0"))
if not _out_dir:
    raise SystemExit("TORCH_PROFILE_OUT env var is required")

torch_profile_callback.install(
    target_step=_target_step, out_dir=_out_dir, rank_only=_rank_only
)

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
