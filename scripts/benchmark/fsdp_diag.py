#!/usr/bin/env python3
"""Standalone FSDP2 sanity check. No training.

Loads the model, runs ``accelerator.prepare``, then reports on rank 0:
  * number of ``FSDPModule`` units found after wrap
  * how many parameters became ``DTensor``
  * sum of local-shard bytes vs full-tensor bytes (ratio should be ~1/world)
  * ``torch.cuda.memory_allocated`` / ``memory_reserved``

Purpose is to settle "is sharding actually working?" without waiting 10+ min
for the first training step. Uses whatever config is in
``scripts/fsdp/accelerate_config.yaml``.

Usage::

    accelerate launch --config_file scripts/fsdp/accelerate_config.yaml \
        --num_processes 8 scripts/benchmark/fsdp_diag.py

    # Override model:
    MODEL=/path/to/model accelerate launch ... scripts/benchmark/fsdp_diag.py
"""
from __future__ import annotations

import os
import sys

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText


def fmt_bytes(nbytes: int) -> str:
    return f"{nbytes / 1024**3:.3f} GB"


def main() -> None:
    model_path = os.environ.get("MODEL")
    if not model_path:
        print(
            "ERROR: set the MODEL env var to an HF hub id or a local model path, "
            "e.g. MODEL=Qwen/Qwen2.5-7B-Instruct accelerate launch ... fsdp_diag.py",
            file=sys.stderr,
        )
        sys.exit(2)

    acc = Accelerator(mixed_precision="bf16")
    rank = acc.process_index
    world = acc.num_processes

    def log(msg: str) -> None:
        if acc.is_main_process:
            print(msg, flush=True)

    log(f"[diag] world={world} rank={rank} model={model_path}")

    # VLM backbones (Qwen3.5-9B et al.) are not registered for
    # AutoModelForCausalLM; fall back to AutoModelForImageTextToText the
    # same way scripts/fsdp/train.py does.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
    except ValueError:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
    total_params_b = sum(p.numel() for p in model.parameters()) / 1e9
    log(f"[diag] model loaded, total raw params={total_params_b:.2f} B")

    before_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    log(f"[diag] BEFORE prepare (CPU, rank0): {fmt_bytes(before_bytes)}")

    # accelerate FSDP2 requires model+optimizer to be prepared together.
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=1e-5)
    model, optimizer = acc.prepare(model, optimizer)

    from torch.distributed.tensor import DTensor
    try:
        from torch.distributed.fsdp import FSDPModule
    except Exception:
        FSDPModule = None

    fsdp_unit_count = 0
    if FSDPModule is not None:
        for _, m in model.named_modules():
            if isinstance(m, FSDPModule):
                fsdp_unit_count += 1

    dtensor_count = 0
    plain_count = 0
    local_bytes = 0
    full_bytes = 0
    for _, p in model.named_parameters():
        if isinstance(p, DTensor):
            dtensor_count += 1
            loc = p.to_local()
            local_bytes += loc.numel() * loc.element_size()
            full_bytes += p.numel() * p.element_size()
        else:
            plain_count += 1
            local_bytes += p.numel() * p.element_size()
            full_bytes += p.numel() * p.element_size()

    alloc = torch.cuda.memory_allocated(acc.device)
    reserved = torch.cuda.memory_reserved(acc.device)

    log("")
    log(f"[diag] AFTER prepare (rank {rank}):")
    log(f"       FSDPModule units            : {fsdp_unit_count}")
    log(f"       DTensor params              : {dtensor_count}")
    log(f"       plain torch.Tensor params   : {plain_count}")
    log(f"       sum(local shard bytes)      : {fmt_bytes(local_bytes)}")
    log(f"       sum(full tensor bytes)      : {fmt_bytes(full_bytes)}")
    ratio = local_bytes / max(full_bytes, 1)
    log(
        f"       ratio local/full            : {ratio:.4f}  "
        f"(expected ~{1/world:.4f} if sharded)"
    )
    log(f"       torch.cuda.memory_allocated : {fmt_bytes(alloc)}")
    log(f"       torch.cuda.memory_reserved  : {fmt_bytes(reserved)}")

    for name, p in model.named_parameters():
        if isinstance(p, DTensor):
            loc = p.to_local()
            log(f"[diag] sample DTensor param: {name}")
            log(f"       full  shape = {tuple(p.shape)}")
            log(f"       local shape = {tuple(loc.shape)}")
            log(f"       placements  = {p.placements}")
            log(f"       device      = {loc.device}")
            break
    else:
        for name, p in model.named_parameters():
            log(
                f"[diag] NOT a DTensor: {name} "
                f"shape={tuple(p.shape)} device={p.device}"
            )
            break


if __name__ == "__main__":
    main()
