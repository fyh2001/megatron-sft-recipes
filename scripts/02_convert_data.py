#!/usr/bin/env python3
"""02_convert_data.py

把 HuggingFace Arrow 数据集（messages + per-message loss）转成
ms-swift 原生支持的 jsonl 格式。

原始样本：
    {
        "messages": [
            {"role": "system" | "user" | "assistant",
             "content": str,
             "loss": float | None},   # 只有 assistant 才有意义
            ...
        ],
        "lengths": [int, ...]  # 本脚本不使用
    }

输出样本（一行一个 JSON）：
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "...", "loss": true},   # loss==1
            {"role": "assistant", "content": "...", "loss": false},  # loss==0
            {"role": "assistant", "content": "..."}                  # loss==None 时省略
        ]
    }

ms-swift 文档（Customization/Custom-dataset.md）明确：
    - "loss" 字段只对 role == "assistant" 生效
    - true  等同于 loss_scale=1（参与训练）
    - false 等同于 loss_scale=0（不参与 loss 计算，但仍作为上下文）
    - 不写（None）= 走 --loss_scale 命令行默认（通常就是 last_round/all）
    - per-message loss 的优先级高于命令行 --loss_scale

用法：
    uv run scripts/02_convert_data.py
    uv run scripts/02_convert_data.py --src /path/to/train --out-dir ./sft-data
    uv run scripts/02_convert_data.py --valid-ratio 0.01 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any


def convert_message(m: dict[str, Any]) -> dict[str, Any]:
    """把原始 message（role/content/loss）转成 ms-swift 期望的格式。"""
    role = m["role"]
    content = m["content"] if m["content"] is not None else ""
    out: dict[str, Any] = {"role": role, "content": content}

    # 只有 assistant 的 loss 才生效（ms-swift 文档明确）
    if role == "assistant":
        loss = m.get("loss")
        if loss is not None:
            # 原数据中的 0.0 / 1.0 / 偶尔可能是 0.5 之类的中间值
            # ms-swift 只接受 bool，按 >=0.5 做阈值
            out["loss"] = bool(loss >= 0.5)

    return out


def convert_sample(sample: dict[str, Any]) -> dict[str, Any] | None:
    """把一条原始样本转成 ms-swift 格式的一条 jsonl 记录。

    如果该样本的所有 assistant turn 都没有 loss=1.0，返回 None 跳过
    （没有任何可训练的 turn，留着也没用）。
    """
    messages = [convert_message(m) for m in sample["messages"]]

    has_trainable = any(
        m["role"] == "assistant" and m.get("loss") is True for m in messages
    )
    if not has_trainable:
        return None

    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--src",
        default="/Users/huangye/Downloads/train",
        help="Arrow 数据集目录（load_from_disk 可读）",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sft-data"),
        help="输出目录，生成 train.jsonl / valid.jsonl",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.01,
                        help="验证集比例（默认 1%%）")
    parser.add_argument("--seed", type=int, default=42, help="打乱/切分的随机种子")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="仅处理前 N 条（调试用，0=全量）")
    args = parser.parse_args()

    try:
        from datasets import load_from_disk
    except ImportError:
        print("错误：请先在 Mac 本地跑 'uv sync' 安装 datasets", file=sys.stderr)
        sys.exit(1)

    print(f"[load] {args.src}")
    ds = load_from_disk(args.src)
    n_src = len(ds)
    print(f"[load] {n_src} 条原样本")

    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, n_src)))
        print(f"[load] 调试模式：仅处理前 {len(ds)} 条")

    os.makedirs(args.out_dir, exist_ok=True)

    converted: list[dict[str, Any]] = []
    skipped = 0
    for idx, sample in enumerate(ds):
        row = convert_sample(sample)
        if row is None:
            skipped += 1
            continue
        converted.append(row)
        if (idx + 1) % 2000 == 0:
            print(f"  processed {idx+1}/{len(ds)}  kept={len(converted)} skipped={skipped}")

    print(f"[convert] kept={len(converted)}, skipped={skipped}")

    if not converted:
        print("错误：没有任何可训练样本，请检查数据 loss 字段", file=sys.stderr)
        sys.exit(2)

    rng = random.Random(args.seed)
    rng.shuffle(converted)
    n_valid = max(1, int(len(converted) * args.valid_ratio))
    valid = converted[:n_valid]
    train = converted[n_valid:]
    print(f"[split] train={len(train)}  valid={len(valid)}  (seed={args.seed})")

    train_path = os.path.join(args.out_dir, "train.jsonl")
    valid_path = os.path.join(args.out_dir, "valid.jsonl")

    for path, rows in [(train_path, train), (valid_path, valid)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[save] {path}  ({size_mb:.1f} MB)")

    print("\n[preview] train[0].messages[:3]:")
    for m in train[0]["messages"][:3]:
        content = m["content"]
        preview = content[:120] + ("..." if len(content) > 120 else "")
        loss_str = f"  loss={m['loss']}" if "loss" in m else ""
        print(f"  {m['role']:10s} {preview!r}{loss_str}")


if __name__ == "__main__":
    main()
