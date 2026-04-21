#!/usr/bin/env python3
"""Introspect an HF model without loading weights.

Prints shell-evalable lines that the bench scripts source to avoid
hard-coding per-model constants:

    NUM_PARAMS=7615616512
    FSDP_WRAP_CLS=Qwen2DecoderLayer
    HIDDEN_SIZE=3584
    NUM_LAYERS=28
    VOCAB_SIZE=152064
    MODEL_TYPE=qwen2

Implementation:
  1. ``AutoConfig.from_pretrained`` - cheap, no weights.
  2. ``AutoModelForCausalLM.from_config(config)`` under ``torch.device('meta')``
     so every parameter is a zero-bytes meta tensor - instantiation is
     instantaneous, no GPU memory, no disk I/O beyond the config.
  3. Sum ``p.numel()`` on the meta model.
  4. Walk ``model.named_modules()`` looking at the immediate children of the
     first ModuleList whose children repeat; pick that class's ``__class__.__name__``.
     This is how accelerate's ``TRANSFORMER_BASED_WRAP`` implicitly works too.

Usage::

    # print key=value lines (default)
    python scripts/benchmark/_inspect_model.py /path/to/model

    # or single field, for $(...) substitution
    python scripts/benchmark/_inspect_model.py /path/to/model --field num_params
"""
from __future__ import annotations

import argparse
import sys


def _pick_decoder_layer_cls(model) -> str | None:
    """Find the transformer decoder layer class name.

    Heuristic: the longest repeating ModuleList of homogeneous children deep
    inside the model is almost always the decoder block list
    (``model.layers``, ``transformer.h``, ``model.decoder.layers`` etc.).
    Return the class name of that repeating child.
    """
    import torch.nn as nn

    best: tuple[int, str] | None = None
    for _name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList):
            continue
        if len(module) < 2:
            continue
        child_classes = {type(c).__name__ for c in module}
        if len(child_classes) != 1:
            continue
        cls = next(iter(child_classes))
        # skip obvious non-decoder lists (experts, embeddings, ...) by size
        if best is None or len(module) > best[0]:
            best = (len(module), cls)
    return best[1] if best else None


def inspect(model_path: str) -> dict[str, str]:
    # Import inside the function so `--help` is fast and a missing torch does
    # not abort argument parsing.
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=False)

    # Some VLM/multimodal configs (Mistral3Config, Gemma3Config, ...) are not
    # registered for AutoModelForCausalLM because the causal decoder is only a
    # sub-module. Fall back to AutoModel, and if even that fails, try the
    # text-only sub-config (``config.text_config``).
    def _build(c):
        with torch.device("meta"):
            try:
                return AutoModelForCausalLM.from_config(c)
            except (ValueError, KeyError):
                return AutoModel.from_config(c)

    try:
        model = _build(cfg)
    except (ValueError, KeyError):
        text_cfg = getattr(cfg, "text_config", None) or getattr(cfg, "llm_config", None)
        if text_cfg is None:
            raise
        model = _build(text_cfg)
        # surface the text-config attributes below.
        cfg = text_cfg

    num_params = sum(p.numel() for p in model.parameters())

    wrap_cls = _pick_decoder_layer_cls(model) or ""

    # For VLM wrapper configs (Mistral3Config, Gemma3Config, ...) the numeric
    # fields live on the text-backbone sub-config. Prefer the sub-config when
    # the outer one does not expose them.
    text_cfg = getattr(cfg, "text_config", None) or getattr(cfg, "llm_config", None) or cfg

    def _first(*candidates):
        for c in candidates:
            if c not in (None, "", 0):
                return c
        return ""

    out: dict[str, str] = {
        "NUM_PARAMS": str(num_params),
        "FSDP_WRAP_CLS": wrap_cls,
        "MODEL_TYPE": str(_first(getattr(cfg, "model_type", None), getattr(text_cfg, "model_type", None))),
        "HIDDEN_SIZE": str(_first(getattr(cfg, "hidden_size", None), getattr(text_cfg, "hidden_size", None))),
        "NUM_LAYERS": str(_first(
            getattr(cfg, "num_hidden_layers", None),
            getattr(text_cfg, "num_hidden_layers", None),
            getattr(cfg, "num_layers", None),
            getattr(text_cfg, "num_layers", None),
        )),
        "VOCAB_SIZE": str(_first(getattr(cfg, "vocab_size", None), getattr(text_cfg, "vocab_size", None))),
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model_path", help="HF hub id or local path")
    p.add_argument(
        "--field",
        default=None,
        help="Print only the value of this key (e.g. num_params).",
    )
    args = p.parse_args()

    info = inspect(args.model_path)

    if args.field is not None:
        key = args.field.upper()
        if key not in info:
            print(
                f"ERROR: unknown field {args.field!r}, "
                f"available: {sorted(k.lower() for k in info)}",
                file=sys.stderr,
            )
            return 2
        print(info[key])
        return 0

    for k, v in info.items():
        # Shell-safe: no spaces possible in the values produced here.
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
