#!/usr/bin/env python3
"""FSDP2 + torch.compile SFT training script.

Hand-written Accelerate training loop with:
  - FSDP2 (fully_shard) for parameter/gradient/optimizer sharding
  - torch.compile for fused kernels
  - Per-message loss masking (reads `loss` field from each assistant message)
  - Cosine LR schedule with warmup and min_lr
  - BF16 mixed precision, gradient checkpointing, flash attention

Usage:
    # Single GPU (debug)
    python scripts/fsdp/train.py --model_name_or_path Qwen/Qwen2.5-7B-Instruct ...

    # Multi-GPU via accelerate
    accelerate launch --config_file scripts/fsdp/accelerate_config.yaml \\
        scripts/fsdp/train.py --model_name_or_path Qwen/Qwen2.5-7B-Instruct ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from functools import partial

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--valid_file", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument("--freeze_vision", action="store_true")

    p.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"])

    p.add_argument("--debug_tokenization", action="store_true",
                    help="Print token-message alignment for first 3 samples, then exit")

    p.add_argument("--max_steps", type=int, default=-1,
                    help="Stop after N optimizer steps (-1 = run full epochs)")
    p.add_argument("--benchmark", action="store_true",
                    help="Benchmark mode: per-step jsonl metrics, no checkpointing")
    p.add_argument("--synthetic", action="store_true",
                    help="Replace ChatDataset with fixed-shape random token batches "
                         "(for benchmarking without data pipeline overhead).")
    p.add_argument("--pad_to_max", action="store_true",
                    help="Pad every batch to --max_length instead of batch-max. "
                         "Required to keep torch.compile/cudagraphs from "
                         "recompiling on every new shape.")
    p.add_argument("--benchmark_log", type=str, default=None,
                    help="Path for benchmark jsonl output (default: output_dir/bench.jsonl)")
    p.add_argument("--warmup_steps_bench", type=int, default=20,
                    help="Steps to skip before measuring in benchmark mode")

    # nsys profiling. Paired with `nsys profile --capture-range=cudaProfilerApi`
    # in bench_fsdp.sh PROFILE=true mode, so the resulting nsys-rep only
    # contains the requested [start, end) step window (~5 steps ~1.8s
    # instead of 150s of the full run).
    p.add_argument("--profile", action="store_true",
                    help="Emit cudaProfilerStart()/Stop() around the step window "
                         "[--profile_start_step, --profile_end_step).")
    p.add_argument("--profile_start_step", type=int, default=5)
    p.add_argument("--profile_end_step", type=int, default=9)

    args = p.parse_args()
    if args.no_compile:
        args.compile = False
    if args.no_gradient_checkpointing:
        args.gradient_checkpointing = False
    if args.benchmark:
        args.save_steps = 999999
        if args.benchmark_log is None:
            args.benchmark_log = os.path.join(args.output_dir, "bench.jsonl")
    return args


# ---------------------------------------------------------------------------
# Per-message loss masking
# ---------------------------------------------------------------------------

def _legacy_build_labels_with_loss_mask(
    messages: list[dict], tokenizer, max_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slow O(N_msg^2) reference implementation, only used when the tokenizer
    is a slow (Python) tokenizer that cannot return offset_mapping."""
    prev_ids: list[int] = []
    all_labels: list[int] = []

    for i in range(len(messages)):
        partial_ids = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=True, add_generation_prompt=False
        )
        if hasattr(partial_ids, "keys") and "input_ids" in partial_ids.keys():
            partial_ids = partial_ids["input_ids"]
        elif hasattr(partial_ids, "ids"):
            partial_ids = partial_ids.ids
        new_tokens = partial_ids[len(prev_ids):]

        msg = messages[i]
        # `loss` in the jsonl is a loss WEIGHT / MASK, not a strict bool:
        # ms-swift's chatml schema puts float 1.0 (train this turn) or 0.0
        # (skip). Old code tested `is True`, which never matched float 1.0
        # and silently masked every assistant turn -> CE over all -100
        # -> NaN loss. Truthy check handles 1.0 / True / positive weights;
        # 0.0 / None / False stays masked as intended.
        should_train = msg["role"] == "assistant" and bool(msg.get("loss"))
        if should_train:
            all_labels.extend(new_tokens)
        else:
            all_labels.extend([-100] * len(new_tokens))
        prev_ids = partial_ids

    all_ids = prev_ids[:max_length]
    all_labels = all_labels[:max_length]
    return (
        torch.tensor(all_ids, dtype=torch.long),
        torch.tensor(all_labels, dtype=torch.long),
    )


def build_labels_with_loss_mask(
    messages: list[dict], tokenizer, max_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a conversation and build per-token loss labels.

    O(N_msg) implementation:
      1. Incrementally render the chat template (tokenize=False) per message
         to obtain each message's character span in the final text.
      2. Tokenize the full text once with return_offsets_mapping=True.
      3. Map each token to a span via its char offset; tokens whose owning
         message is an assistant turn with loss=True get their id as label,
         others get -100.

    Falls back to the legacy O(N_msg^2) loop for slow tokenizers (no fast
    backend / no offset_mapping support).
    """
    if not getattr(tokenizer, "is_fast", False):
        return _legacy_build_labels_with_loss_mask(messages, tokenizer, max_length)

    prev_text = ""
    char_segments: list[tuple[int, int, bool]] = []
    for i, msg in enumerate(messages):
        full_text = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=False, add_generation_prompt=False
        )
        # See note in _legacy_build_labels_with_loss_mask: jsonl `loss`
        # field is a float weight (1.0 train / 0.0 skip), not a bool.
        should_train = msg["role"] == "assistant" and bool(msg.get("loss"))
        char_segments.append((len(prev_text), len(full_text), should_train))
        prev_text = full_text

    enc = tokenizer(
        prev_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    labels = [-100] * len(input_ids)
    seg_idx = 0
    for ti, (s, _e) in enumerate(offsets):
        while seg_idx < len(char_segments) and char_segments[seg_idx][1] <= s:
            seg_idx += 1
        if seg_idx >= len(char_segments):
            break
        if char_segments[seg_idx][2]:
            labels[ti] = input_ids[ti]

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


class ChatDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int):
        with open(jsonl_path) as f:
            self.samples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        messages = self.samples[idx]["messages"]
        input_ids, labels = build_labels_with_loss_mask(
            messages, self.tokenizer, self.max_length
        )
        return {"input_ids": input_ids, "labels": labels}


class SyntheticDataset(Dataset):
    """Fixed-length random token batches, avoids dataloader+tokenization overhead."""

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, seed: int = 0):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self._gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = torch.randint(
            0, self.vocab_size, (self.seq_len,), generator=self._gen, dtype=torch.long
        )
        return {"input_ids": ids, "labels": ids.clone()}


def collate_fn(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int,
    pad_to_max_length: int | None = None,
) -> dict[str, torch.Tensor]:
    if pad_to_max_length is not None:
        max_len = pad_to_max_length
    else:
        max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids, labels, attention_mask = [], [], []
    for b in batch:
        pad_len = max_len - b["input_ids"].size(0)
        input_ids.append(F.pad(b["input_ids"], (0, pad_len), value=pad_token_id))
        labels.append(F.pad(b["labels"], (0, pad_len), value=-100))
        mask = torch.ones(b["input_ids"].size(0), dtype=torch.long)
        attention_mask.append(F.pad(mask, (0, pad_len), value=0))
    out = {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }
    if pad_to_max_length is not None:
        # When every batch has identical shape, the attention_mask is
        # irrelevant for a causal LM: real tokens come first, pad tokens last,
        # pad positions have labels=-100 (no loss contribution), and causal
        # masking ensures real tokens cannot attend to trailing pads. Drop the
        # mask so transformers skips its `.all()` GPU sync in create_causal_mask
        # (see masking_utils.py flash_attention_mask), which otherwise serializes
        # the FSDP2 all-gather stream at every forward.
        out.pop("attention_mask")
    return out


# ---------------------------------------------------------------------------
# LR scheduler: cosine with warmup and min_lr
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_min_lr(
    optimizer, num_warmup_steps: int, num_training_steps: int, min_lr: float
):
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def lr_lambda(current_step: int, base_lr: float) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        target_lr = min_lr + (base_lr - min_lr) * cosine_decay
        return target_lr / base_lr

    lambdas = [partial(lr_lambda, base_lr=blr) for blr in base_lrs]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)


# ---------------------------------------------------------------------------
# Debug tokenization
# ---------------------------------------------------------------------------

def debug_tokenization(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    with open(args.train_file) as f:
        samples = [json.loads(next(f)) for _ in range(3)]

    for si, sample in enumerate(samples):
        messages = sample["messages"]
        print(f"\n{'='*60}\nSample {si}: {len(messages)} messages\n{'='*60}")
        prev_ids: list[int] = []
        total_tokens, train_tokens = 0, 0
        for i, msg in enumerate(messages):
            partial_ids = tokenizer.apply_chat_template(
                messages[: i + 1], tokenize=True, add_generation_prompt=False
            )
            if hasattr(partial_ids, "keys") and "input_ids" in partial_ids.keys():
                partial_ids = partial_ids["input_ids"]
            elif hasattr(partial_ids, "ids"):
                partial_ids = partial_ids.ids
            new_tokens = partial_ids[len(prev_ids):]
            should_train = msg["role"] == "assistant" and bool(msg.get("loss"))
            total_tokens += len(new_tokens)
            if should_train:
                train_tokens += len(new_tokens)
            decoded = tokenizer.decode(new_tokens[:30])
            suffix = "..." if len(new_tokens) > 30 else ""
            loss_str = f"loss={msg.get('loss')}" if msg["role"] == "assistant" else ""
            print(
                f"  msg[{i:2d}] {msg['role']:10s} {loss_str:12s} "
                f"{len(new_tokens):5d} tok  "
                f"train={'Y' if should_train else 'N'}  "
                f"preview: {decoded!r}{suffix}"
            )
            prev_ids = partial_ids
        print(f"  TOTAL: {total_tokens} tokens, {train_tokens} train tokens "
              f"({train_tokens/max(total_tokens,1)*100:.1f}%)")
    print(f"\n{'='*60}\nDebug complete. Exiting.\n{'='*60}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.debug_tokenization:
        debug_tokenization(args)
        return

    # NOTE: do NOT pass gradient_accumulation_steps here. On accelerate 1.13 +
    # FSDP2 + transformers 5.x, the GradientAccumulationPlugin installed by
    # that kwarg causes the post-prepare rank0 allocation to land at full
    # model size (15.96 GB on Qwen2.5-7B) instead of the sharded 1.77 GB
    # that fsdp_diag.py sees with the same config but without GAS. The
    # grad-accum semantics we actually use are driven entirely by the outer
    # `accelerator.accumulate(model)` context manager below, which does NOT
    # depend on this constructor arg being set.
    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Load tokenizer + model ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )


    # Activation checkpointing:
    #   * When the accelerate FSDP config has `fsdp_activation_checkpointing:
    #     true`, accelerate installs torch.distributed's
    #     `apply_activation_checkpointing` inside `accelerator.prepare()`,
    #     AFTER the decoder layers have been wrapped with `fully_shard`. This
    #     is the correct order for FSDP2: sharding first, ckpt wrapper second.
    #   * Transformers' own `model.gradient_checkpointing_enable()` rewrites
    #     module internals BEFORE prepare. With TRANSFORMER_BASED_WRAP matching
    #     on `Qwen2DecoderLayer`, this changes the module layout enough that
    #     accelerate's auto_wrap policy no longer matches any layer, FSDP falls
    #     back to per-leaf sharding (or no sharding), and you end up with a
    #     full model replica per rank + an all_gather storm.
    # So: only run the transformers path when accelerate's FSDP config is NOT
    # taking care of it.
    _fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
    _fsdp_acx = bool(getattr(_fsdp_plugin, "activation_checkpointing", False))
    if args.gradient_checkpointing and not _fsdp_acx:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    elif args.gradient_checkpointing and accelerator.is_main_process:
        print("[train] gradient_checkpointing handled by FSDP2 plugin")

    if args.freeze_vision:
        frozen = 0
        for name, param in model.named_parameters():
            if "vision" in name or "visual" in name or "projector" in name:
                param.requires_grad = False
                frozen += 1
        if accelerator.is_main_process:
            print(f"Froze {frozen} vision parameters")

    # NOTE: torch.compile is intentionally deferred until AFTER
    # accelerator.prepare() so that compile sees the FSDP2 fully_shard unit
    # boundaries instead of the unwrapped model.

    # --- Datasets ---
    if args.synthetic:
        vocab = getattr(model.config, "vocab_size", None) or tokenizer.vocab_size
        train_dataset = SyntheticDataset(
            num_samples=max(args.max_steps, 100) * args.per_device_train_batch_size
            * max(args.gradient_accumulation_steps, 1) * 8,
            seq_len=args.max_length,
            vocab_size=vocab,
        )
    else:
        train_dataset = ChatDataset(args.train_file, tokenizer, args.max_length)
    collate = partial(
        collate_fn,
        pad_token_id=tokenizer.pad_token_id,
        pad_to_max_length=args.max_length if args.pad_to_max else None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # --- Optimizer + scheduler ---
    # CRITICAL: pass a GENERATOR to AdamW, do NOT pre-materialize a list into a
    # named local. On accelerate 1.13 + FSDP2 the `fully_shard` inside
    # `accelerator.prepare()` only physically reshards a parameter if it is
    # the sole external holder of the pre-shard tensor; any additional Python
    # strong reference (e.g. `trainable_params = [p for p in ...]`) keeps the
    # full tensor alive on every rank and prepare silently degrades to "mark
    # DTensor on the parameter but skip the memory reshard".
    # Symptom when this regression slips back: rank0 allocated == full model
    # size (15.96 GB for Qwen2.5-7B instead of 1.77 GB = 14/8), first training
    # step hangs forever because every decoder layer issues a full-model
    # all_gather that never completes. See scripts/benchmark/fsdp_diag.py for
    # the known-good post-prepare baseline.
    # AdamW consumes the generator into its own internal list inside
    # param_groups; that list is the only external holder when prepare() runs.
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    num_training_steps = steps_per_epoch * args.num_train_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_min_lr(
        optimizer, num_warmup_steps, num_training_steps, min_lr=args.min_lr
    )

    # --- Accelerate prepare ---
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # Diagnostic: confirm FSDP actually sharded the transformer layers.
    # If wrap policy misfired, wrap_units will be 0/1 and rank0 allocated
    # memory will be full model size instead of ~1/N. Use the same checks as
    # scripts/benchmark/fsdp_diag.py: FSDP2 marks modules with the FSDPModule
    # mixin, FSDP1 wraps them in FullyShardedDataParallel; parameters on
    # sharded modules become DTensor.
    if accelerator.is_main_process:
        try:
            from torch.distributed.fsdp import FSDPModule as _FSDPModule
        except Exception:
            _FSDPModule = None
        from torch.distributed.tensor import DTensor as _DTensor

        fsdp_units = []
        for name, module in model.named_modules():
            is_fsdp1 = type(module).__name__ == "FullyShardedDataParallel"
            is_fsdp2 = _FSDPModule is not None and isinstance(module, _FSDPModule)
            if is_fsdp1 or is_fsdp2:
                n_params = sum(
                    p.numel() for p in module.parameters(recurse=False)
                )
                fsdp_units.append((name or "<root>", n_params, "fsdp2" if is_fsdp2 else "fsdp1"))

        dtensor_params = sum(
            1 for _, p in model.named_parameters() if isinstance(p, _DTensor)
        )
        total_params = sum(1 for _ in model.named_parameters())

        print(f"[FSDP diag] total wrap units: {len(fsdp_units)}")
        for name, n, kind in fsdp_units[:5]:
            print(f"  {kind}  {name:60s}  own_params={n/1e6:.2f}M")
        if len(fsdp_units) > 5:
            print(f"  ... ({len(fsdp_units) - 5} more units)")
        print(
            f"[FSDP diag] DTensor params: {dtensor_params}/{total_params} "
            f"(0 means nothing was sharded)"
        )
        alloc_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        print(
            f"[FSDP diag] rank0 GPU mem after prepare: "
            f"allocated={alloc_gb:.2f} GB  reserved={reserved_gb:.2f} GB"
        )

    if args.compile:
        # FSDP2 + compile: wrap AFTER accelerator.prepare so torch.compile
        # specializes per fully_shard unit. dynamic=True avoids re-compiling
        # on every new sequence-length shape (combined with --pad_to_max,
        # this should keep compile cost to a one-shot warm-up).
        model = torch.compile(model, dynamic=True)

    if accelerator.is_main_process:
        gbs = (
            args.per_device_train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )
        print("Training config:")
        print(f"  model          = {args.model_name_or_path}")
        print(f"  train samples  = {len(train_dataset)}")
        print(f"  epochs         = {args.num_train_epochs}")
        print(f"  per_device_bs  = {args.per_device_train_batch_size}")
        print(f"  grad_accum     = {args.gradient_accumulation_steps}")
        print(f"  global_bs      = {gbs}")
        print(f"  total steps    = {num_training_steps}")
        print(f"  warmup steps   = {num_warmup_steps}")
        print(f"  LR             = {args.learning_rate} -> {args.min_lr}")
        print(f"  compile        = {args.compile}")
        print(f"  grad_ckpt      = {args.gradient_checkpointing}")
        print(f"  num_processes  = {accelerator.num_processes}")

    # --- Training loop ---
    global_step = 0
    log_loss_accum = 0.0
    log_token_accum = 0
    t0 = time.time()
    step_t0 = time.time()
    bench_file = None
    if args.benchmark and accelerator.is_main_process:
        bench_file = open(args.benchmark_log, "w")

    done = False
    # nsys profiling control: start right BEFORE the chosen step begins its
    # forward, stop right AFTER the chosen end step's optimizer.step. Gates
    # on `--profile` so non-profile runs skip the syscall.
    _nsys_started = False
    for epoch in range(args.num_train_epochs):
        if done:
            break
        model.train()
        for _step, batch in enumerate(train_loader):
            if (
                args.profile
                and not _nsys_started
                and global_step + 1 == args.profile_start_step
            ):
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                _nsys_started = True

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            if (
                args.profile
                and _nsys_started
                and global_step >= args.profile_end_step
            ):
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                _nsys_started = False  # don't call Stop twice
                # Keep training so nsys has time to flush the buffer on app
                # exit; TOTAL_STEPS should already be small (e.g. 12).
            step_time_ms = (time.time() - step_t0) * 1000
            if "attention_mask" in batch:
                step_tokens = (batch["attention_mask"] == 1).sum().item()
            else:
                step_tokens = batch["input_ids"].numel()
            log_loss_accum += loss.detach().float().item()
            log_token_accum += step_tokens

            # Benchmark per-step logging.
            # CRITICAL: accelerator.gather() is a COLLECTIVE (all_gather under
            # the hood). Every rank must call it or the collective hangs
            # forever. The previous code wrapped this inside
            # `if accelerator.is_main_process:`, which silently deadlocked on
            # multi-rank runs: rank 0 blocked in gather waiting for peers,
            # peers meanwhile tried to start step N+1's FSDP per-layer
            # all_gathers that needed rank 0 to participate, and both ends
            # waited on each other. Symptom: every rank pinned at 100% util
            # 120 W (NCCL polling) and bench.jsonl stayed empty.
            if args.benchmark:
                if accelerator.num_processes > 1:
                    total_tokens_t = accelerator.gather(
                        torch.tensor([step_tokens], device=accelerator.device)
                    )
                    total_tokens = int(total_tokens_t.sum().item())
                else:
                    total_tokens = step_tokens
                if accelerator.is_main_process:
                    bench_file.write(json.dumps({
                        "step": global_step,
                        "step_time_ms": round(step_time_ms, 2),
                        "tokens": total_tokens,
                        "loss": round(loss.detach().float().item(), 4),
                    }) + "\n")
                    bench_file.flush()

            step_t0 = time.time()

            if global_step % args.logging_steps == 0:
                avg_loss = log_loss_accum / args.logging_steps
                elapsed = time.time() - t0
                tokens_per_sec = log_token_accum / max(elapsed, 1e-9)
                lr_current = scheduler.get_last_lr()[0]
                if accelerator.is_main_process:
                    print(
                        f"step={global_step:5d}  "
                        f"epoch={epoch+1}  "
                        f"loss={avg_loss:.4f}  "
                        f"lr={lr_current:.2e}  "
                        f"tok/s={tokens_per_sec:.0f}"
                    )
                log_loss_accum = 0.0
                log_token_accum = 0
                t0 = time.time()

            if global_step % args.save_steps == 0:
                save_dir = os.path.join(
                    args.output_dir, f"checkpoint-{global_step}"
                )
                if accelerator.is_main_process:
                    print(f"Saving checkpoint to {save_dir}")
                accelerator.wait_for_everyone()
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(
                    save_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    safe_serialization=True,
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(save_dir)

            if args.max_steps > 0 and global_step >= args.max_steps:
                done = True
                break

    if bench_file:
        bench_file.close()

    # --- Final save (skip in benchmark mode) ---
    if not args.benchmark:
        final_dir = os.path.join(args.output_dir, "final")
        if accelerator.is_main_process:
            print(f"Saving final model to {final_dir}")
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(
            final_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            safe_serialization=True,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(final_dir)

    if accelerator.is_main_process:
        print("Training complete.")


if __name__ == "__main__":
    main()
