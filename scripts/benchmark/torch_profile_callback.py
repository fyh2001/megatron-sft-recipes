"""TrainerCallback that runs torch.profiler on exactly one training step.

Designed for kernel-level post-hoc analysis (per-kernel time, tensor-core
participation) without polluting step-time numbers elsewhere in the run.

Emits a Chrome-trace JSON to `out_dir/trace_step<N>.json`. Parse with
`analyze_torch_profile.py`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import TrainerCallback


class OneStepTorchProfiler(TrainerCallback):
    def __init__(self, target_step: int, out_dir: str, rank_only: int = 0) -> None:
        self.target_step = int(target_step)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rank_only = rank_only
        self._prof: profile | None = None
        self._active = False
        try:
            self._rank = int(os.environ.get("RANK", "0"))
        except Exception:
            self._rank = 0

    def _should_profile(self) -> bool:
        return self.rank_only < 0 or self._rank == self.rank_only

    def on_step_begin(self, args, state, control, **kwargs):  # noqa: D401
        if not self._should_profile():
            return
        # state.global_step is 0-indexed here: it increments AFTER on_step_end
        if state.global_step == self.target_step and not self._active:
            self._prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                with_stack=False,
                with_flops=False,
                profile_memory=False,
            )
            self._prof.__enter__()
            self._active = True
            torch.cuda.synchronize()
            print(f"[torch_profile] rank={self._rank} BEGIN step={state.global_step}")

    def on_step_end(self, args, state, control, **kwargs):  # noqa: D401
        if not self._active:
            return
        torch.cuda.synchronize()
        self._prof.__exit__(None, None, None)
        trace_path = self.out_dir / f"trace_rank{self._rank}_step{self.target_step}.json"
        print(f"[torch_profile] rank={self._rank} END step={state.global_step} → {trace_path}")
        self._prof.export_chrome_trace(str(trace_path))
        kt_txt = self._prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=40
        )
        (self.out_dir / f"ktable_rank{self._rank}.txt").write_text(kt_txt)
        summary = {
            "rank": self._rank,
            "target_step": self.target_step,
            "trace": str(trace_path),
        }
        (self.out_dir / f"summary_rank{self._rank}.json").write_text(json.dumps(summary))
        self._active = False
        self._prof = None


_DID_PATCH = False


def install(target_step: int, out_dir: str, rank_only: int = 0) -> None:
    """Monkey-patch `transformers.Trainer.__init__` to append our callback.

    Signature is preserved via `__signature__` so swift's introspection still
    sees the real parameter list (e.g. `processing_class`).
    """
    global _DID_PATCH
    if _DID_PATCH:
        return
    _DID_PATCH = True

    import functools
    import inspect

    from transformers import Trainer as _Trainer

    orig_init = _Trainer.__init__
    orig_sig = inspect.signature(orig_init)

    @functools.wraps(orig_init)
    def patched_init(self, *args, **kwargs):  # type: ignore[override]
        orig_init(self, *args, **kwargs)
        self.add_callback(
            OneStepTorchProfiler(
                target_step=target_step, out_dir=out_dir, rank_only=rank_only
            )
        )
        if int(os.environ.get("RANK", "0")) == rank_only or rank_only < 0:
            print(
                f"[torch_profile] installed OneStepTorchProfiler target_step={target_step} "
                f"out_dir={out_dir}"
            )

    patched_init.__signature__ = orig_sig  # type: ignore[attr-defined]
    _Trainer.__init__ = patched_init  # type: ignore[assignment]
