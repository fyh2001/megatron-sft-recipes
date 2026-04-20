#!/usr/bin/env python3
"""convert_ministral3_to_llama.py

把 Ministral-3-3B-Instruct-2512-BF16 的 VLM 结构砍成纯文本 LlamaForCausalLM，
这样就能走 mcore_bridge 的 Llama 分支用 Megatron 后端训练（不用改 bridge 源码）。

做三件事：
  1. 重写 config.json：
     - architectures → ["LlamaForCausalLM"]
     - model_type → "llama"
     - text_config 的字段全部提升到顶层
     - 关掉 YaRN scaling（rope_type=yarn 只在长上下文外推时需要；SFT max_length<=4096
       完全走不到 YaRN 的 factor=16 放缩）。保留 rope_theta=1e6，max_position=16384。
     - 删除 vision_config / image_token_index / multimodal_projector_* 等 VLM 字段
  2. 复制 + rename 权重 shard：
     language_model.model.*  →  model.*
     vision_tower.*          →  [删]
     multi_modal_projector.* →  [删]
     tie_word_embeddings=true 所以不需要写 lm_head.weight
  3. 复制 tokenizer（保留 mistral-common 的 tekken），chat_template 也一起带。

用法：
  python scripts/convert_ministral3_to_llama.py \
      --src  /home/ubuntu/perf_opt/.cache/huggingface/hub/models--mistralai--Ministral-3-3B-Instruct-2512-BF16/snapshots/<hash> \
      --dst  /home/ubuntu/perf_opt/models/ministral3-3b-text-llama

若 --src 省略，自动从 HF cache 里找最新的 Ministral-3-3B-Instruct-2512-BF16 snapshot。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

LM_PREFIX = "language_model."
STRIPPED_PREFIXES = ("vision_tower.", "multi_modal_projector.")


def find_default_src() -> str:
    pattern = "/home/ubuntu/perf_opt/.cache/huggingface/hub/models--mistralai--Ministral-3-3B-Instruct-2512-BF16/snapshots/*"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"Could not find Ministral-3-3B-Instruct-2512-BF16 snapshot under HF cache. "
            f"Pass --src explicitly."
        )
    return max(matches, key=os.path.getmtime)


def build_new_config(src_cfg: dict) -> dict:
    """Build a clean LlamaForCausalLM config from Ministral-3 mistral3 config."""
    tc = src_cfg["text_config"]
    new = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "dtype": "bfloat16",
        "torch_dtype": "bfloat16",
        "hidden_size": tc["hidden_size"],
        "num_hidden_layers": tc["num_hidden_layers"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "intermediate_size": tc["intermediate_size"],
        "head_dim": tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"]),
        "vocab_size": tc["vocab_size"],
        # YaRN scaling 仅在外推到 262144 时才用；SFT max_len<=4096 走原生 16384 足够。
        # rope_theta 从 YaRN 参数里取（是 base θ，不是放缩因子）。
        "max_position_embeddings": 16384,
        "rope_theta": float(tc.get("rope_parameters", {}).get("rope_theta", 1_000_000.0)),
        "rope_scaling": None,
        "sliding_window": None,  # 关键：Ministral 没有 SWA
        "tie_word_embeddings": tc.get("tie_word_embeddings", False),
        "hidden_act": tc.get("hidden_act", "silu"),
        "rms_norm_eps": tc.get("rms_norm_eps", 1e-5),
        "attention_bias": False,
        "attention_dropout": tc.get("attention_dropout", 0.0),
        "initializer_range": tc.get("initializer_range", 0.02),
        "use_cache": tc.get("use_cache", True),
        "mlp_bias": False,
        "pretraining_tp": 1,
        "transformers_version": src_cfg.get("transformers_version", "4.50.0"),
        # 特殊 token id：Ministral tokenizer 用 tekken，bos/eos 在 tokenizer_config 里；
        # 这里留空，transformers 会从 tokenizer 里自动补。
    }
    return new


def rename_weight_key(old: str) -> str | None:
    """Return new key name, or None to drop this tensor."""
    for bad in STRIPPED_PREFIXES:
        if old.startswith(bad):
            return None
    if old.startswith(LM_PREFIX):
        return old[len(LM_PREFIX):]  # language_model.model.X → model.X
    # 其它顶层字段（比如 image_newline 如果存在也扔了）
    if old.startswith(("model.", "lm_head.")):
        return old
    return None  # 保险起见，不认识就丢


def convert_weights(src_dir: str, dst_dir: str) -> tuple[int, int, int]:
    """Walk safetensors shards, rename keys, re-shard uniformly, update index."""
    import safetensors
    from safetensors import safe_open
    from safetensors.torch import save_file

    # 读原 index
    idx_path = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        src_index = json.load(open(idx_path))
        weight_map = src_index["weight_map"]
    else:
        # 无 index → 遍历所有 safetensors 自己建
        weight_map = {}
        for shard in sorted(glob.glob(os.path.join(src_dir, "model-*.safetensors"))):
            with safe_open(shard, framework="pt") as f:
                for k in f.keys():
                    weight_map[k] = os.path.basename(shard)

    # 分组：原 shard 文件 → 它里面的老 key
    by_shard: dict[str, list[str]] = {}
    for k, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(k)

    # 生成新 weight_map + 新 shard 写出
    new_weight_map: dict[str, str] = {}
    total_bytes = 0
    n_kept = 0
    n_dropped = 0

    # 为了简单，每个原 shard 对应一个新 shard（丢掉 vision 键，文件会变小，但 index 自洽）
    src_shards = sorted(by_shard)
    for shard_idx, shard_name in enumerate(src_shards):
        new_shard_name = f"model-{shard_idx+1:05d}-of-{len(src_shards):05d}.safetensors"
        src_path = os.path.join(src_dir, shard_name)
        dst_path = os.path.join(dst_dir, new_shard_name)

        tensors = {}
        with safe_open(src_path, framework="pt") as f:
            for old_key in f.keys():
                new_key = rename_weight_key(old_key)
                if new_key is None:
                    n_dropped += 1
                    continue
                t = f.get_tensor(old_key)
                tensors[new_key] = t
                new_weight_map[new_key] = new_shard_name
                total_bytes += t.numel() * t.element_size()
                n_kept += 1
        if not tensors:
            print(f"  [skip] {shard_name}: no LM tensors (全被当成 vision 丢了)")
            continue
        save_file(tensors, dst_path, metadata={"format": "pt"})
        print(f"  [write] {new_shard_name}: {len(tensors)} tensors, "
              f"{os.path.getsize(dst_path)/1e9:.2f} GB")

    # 写新 index
    new_index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": new_weight_map,
    }
    with open(os.path.join(dst_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2, sort_keys=True)
    return n_kept, n_dropped, total_bytes


def copy_tokenizer_files(src_dir: str, dst_dir: str) -> list[str]:
    """Copy tokenizer / chat template related files."""
    keep = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "tekken.json",
        "SYSTEM_PROMPT.txt",  # Mistral 需要用到
        "generation_config.json",
    ]
    copied = []
    for name in keep:
        src = os.path.join(src_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, name))
            copied.append(name)
    return copied


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None, help="Source snapshot dir (default: auto-detect from HF cache)")
    ap.add_argument("--dst", default="/home/ubuntu/perf_opt/models/ministral3-3b-text-llama")
    args = ap.parse_args()

    src = args.src or find_default_src()
    dst = args.dst
    print(f"[src] {src}")
    print(f"[dst] {dst}")
    os.makedirs(dst, exist_ok=True)

    # 1. config.json
    src_cfg = json.load(open(os.path.join(src, "config.json")))
    new_cfg = build_new_config(src_cfg)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(new_cfg, f, indent=2, sort_keys=True)
    print(f"[config] wrote config.json: {len(new_cfg)} keys, "
          f"architectures={new_cfg['architectures']}, rope_theta={new_cfg['rope_theta']}")

    # 2. tokenizer / chat template
    copied = copy_tokenizer_files(src, dst)
    print(f"[tokenizer] copied: {copied}")

    # 3. 权重
    print(f"[weights] rewriting safetensors shards ...")
    n_kept, n_dropped, total_bytes = convert_weights(src, dst)
    print(f"[weights] kept={n_kept}, dropped={n_dropped}, total={total_bytes/1e9:.2f} GB")

    print()
    print(f"[done] new model at: {dst}")
    print(f"       quick sanity:")
    print(f"         python -c 'from transformers import AutoModelForCausalLM, AutoTokenizer; "
          f"m = AutoModelForCausalLM.from_pretrained(\"{dst}\", dtype=\"bfloat16\"); "
          f"print(sum(p.numel() for p in m.parameters())/1e9, \"B params\")'")


if __name__ == "__main__":
    main()
