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
    p.add_argument("--benchmark_log", type=str, default=None,
                    help="Path for benchmark jsonl output (default: output_dir/bench.jsonl)")
    p.add_argument("--warmup_steps_bench", type=int, default=20,
                    help="Steps to skip before measuring in benchmark mode")

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

def build_labels_with_loss_mask(
    messages: list[dict], tokenizer, max_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a conversation and build per-token loss labels.

    For each message, we tokenize the conversation up to and including that
    message, then diff against the previous tokenization to identify which
    new tokens belong to it. Tokens from assistant messages with loss=True
    get their token id as label; everything else gets -100 (ignored by CE).
    """
    prev_ids: list[int] = []
    all_ids: list[int] = []
    all_labels: list[int] = []

    for i in range(len(messages)):
        partial_ids = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=True, add_generation_prompt=False
        )
        new_tokens = partial_ids[len(prev_ids):]

        msg = messages[i]
        should_train = (
            msg["role"] == "assistant" and msg.get("loss") is True
        )

        if should_train:
            all_labels.extend(new_tokens)
        else:
            all_labels.extend([-100] * len(new_tokens))

        prev_ids = partial_ids

    all_ids = prev_ids  # final full sequence

    all_ids = all_ids[:max_length]
    all_labels = all_labels[:max_length]

    return (
        torch.tensor(all_ids, dtype=torch.long),
        torch.tensor(all_labels, dtype=torch.long),
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


def collate_fn(
    batch: list[dict[str, torch.Tensor]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids, labels, attention_mask = [], [], []
    for b in batch:
        pad_len = max_len - b["input_ids"].size(0)
        input_ids.append(F.pad(b["input_ids"], (0, pad_len), value=pad_token_id))
        labels.append(F.pad(b["labels"], (0, pad_len), value=-100))
        mask = torch.ones(b["input_ids"].size(0), dtype=torch.long)
        attention_mask.append(F.pad(mask, (0, pad_len), value=0))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


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
            new_tokens = partial_ids[len(prev_ids):]
            should_train = msg["role"] == "assistant" and msg.get("loss") is True
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

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
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

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if args.freeze_vision:
        frozen = 0
        for name, param in model.named_parameters():
            if "vision" in name or "visual" in name or "projector" in name:
                param.requires_grad = False
                frozen += 1
        if accelerator.is_main_process:
            print(f"Froze {frozen} vision parameters")

    if args.compile:
        model = torch.compile(model)

    # --- Datasets ---
    train_dataset = ChatDataset(args.train_file, tokenizer, args.max_length)
    collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # --- Optimizer + scheduler ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay
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
    for epoch in range(args.num_train_epochs):
        if done:
            break
        model.train()
        for _step, batch in enumerate(train_loader):
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
            step_time_ms = (time.time() - step_t0) * 1000
            step_tokens = (batch["attention_mask"] == 1).sum().item()
            log_loss_accum += loss.detach().float().item()
            log_token_accum += step_tokens

            # Benchmark per-step logging
            if args.benchmark and accelerator.is_main_process:
                total_tokens = accelerator.gather(
                    torch.tensor([step_tokens], device=accelerator.device)
                ).sum().item() if accelerator.num_processes > 1 else step_tokens
                bench_file.write(json.dumps({
                    "step": global_step,
                    "step_time_ms": round(step_time_ms, 2),
                    "tokens": int(total_tokens),
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
