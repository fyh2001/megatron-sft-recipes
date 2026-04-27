# Gemma-4-26B-A4B-it FSDP2 SFT — DIY 踩坑之旅（按 P 阶段推进）

> 这份文档把"项目完成版"反向拆成 **按 phase 推进的 quest**。每个 P 阶段对应一个"想干啥 → 撞啥坑 → 写啥 patch → 跑出啥数字"的小剧场。**答案折叠在 `<details>` 里**，先想再展开。
>
> 想直接复制最终配置的去 [`gemma4_setup_from_scratch.md`](gemma4_setup_from_scratch.md)。这边是给想 walk through 整个项目优化叙事的人。
>
> **预计耗时**：每个 P 阶段 30-90 min（含跑 30 step bench 时间），全部 9 个阶段 ≈ 6-12 小时。

---

## 推进总览

```
P0a  Naive FSDP2 baseline           [6 个 quest，让 baseline 跑起来]
                ↓
P0c  DS prod baseline                [无 quest，跑 user 线上命令]
                ↓
P0g  FSDP2 ↔ DS 数值对齐             [1 个 quest（token stats，可选）]
                ↓
P1   GBS sweep                       [1 个 quest（offload 时的 clip_grad_norm）]
                ↓
P2   AC off                          [无 quest，配置 flip]
                ↓
P3   reshard=false                   [无 quest，配置 flip，不采纳]
                ↓
P4   packing                         [1 个 quest（swift template 强解锁）]
                ↓
P5 ★ Liger                           [1 个 quest（最大头：liger gemma4 dispatch 从零写）]
                ↓
完成 — 11.2× faster than DS prod baseline
```

| Phase | 想干啥 | 撞到的 quest | 改的代码文件 | 难度 |
|---|---|---|---|---|
| P0a | 让 FSDP2 baseline 能跑 | 0a-1 ~ 0a-6 | 配置 + modeling_gemma4 + sitecustomize 4 个 patch | ⭐⭐ x 6 |
| P0c | DS prod baseline | — | — | ⭐ |
| P0g | FSDP2 ↔ DS 对齐 | 0g-token (可选) | ms-swift fork 2 文件 | ⭐⭐⭐ |
| P1 | GBS 扫盘 | 1-offload | sitecustomize +1 patch | ⭐⭐⭐ |
| P2 | 关 AC | — | — | ⭐ |
| P3 | reshard=false | — | — | ⭐ |
| P4 | packing | 4-template | sitecustomize +1 patch | ⭐⭐ |
| P5 | Liger | 5-liger | liger_gemma4_patch.py 从零写 + sitecustomize +1 patch | ⭐⭐⭐⭐ |

总共 9 处代码改动：6 个 phase 触发，刚好对应项目里所有 patch。

---

## 准备：把容器重置到"未踩坑"状态

```bash
cd /home/ubuntu/fyh/megatron-sft-recipes
bash scripts/gemma4_opt/diy_reset.sh
```

这个脚本会：
1. 重装 upstream `transformers==5.5.4`（恢复未 patch 的 `modeling_gemma4.py`）
2. 把 `sitecustomize.py` 重命名为 `.applied`（PYTHONPATH 不再加载）
3. 把 `liger_gemma4_patch.py` 重命名为 `.applied`

校验：
```bash
docker exec fsdp_sft md5sum /usr/local/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py
# 应该 NOT 等于 39ebf386a992fea9eac0883f459ac658
```

> **逃生通道**：
> - 比对你写的 vs 参考答案：`diff <your_file> <ref>.applied`
> - 一键恢复：`bash scripts/gemma4_opt/diy_restore.sh`

---

# Phase 0a：让 FSDP2 baseline 跑起来（6 关）

> **目标**：跑 5 步 `swift sft --tuner_type full --fsdp ... gemma-4-26B-A4B-it`，输出 `loss / grad_norm / memory(GiB)` 5 行。
>
> **现状**：reset 之后什么 patch 都没有。直接跑 → 撞 6 个独立的坑（项目里 P0a 实际经历的）。

## P0a 起点：第一次 naive 启动

打开 [`p0_baseline_fsdp2.sh`](../scripts/gemma4_opt/p0_baseline_fsdp2.sh)（已经写好的最小命令），但**先把里面 `FSDP_TRANSFORMER_CLS_NAMES` 这一行注释掉、也不要传任何 sitecustomize 相关环境变量**。  最朴素版本就是：

```bash
docker exec fsdp_sft bash -lc "
cd /home/ubuntu/fyh/megatron-sft-recipes && \
NPROC_PER_NODE=8 swift sft \
    --model /home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it \
    --model_type gemma4 --template gemma4 \
    --dataset /home/ubuntu/fyh/sft-data/train.jsonl \
    --max_length 16384 --truncation_strategy delete \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
    --max_steps 5 --logging_steps 1 --save_strategy no \
    --tuner_type full --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --freeze_vit true --freeze_aligner true \
    --learning_rate 2e-5 --warmup_ratio 0.05 \
    --use_liger_kernel true \
    --fsdp '{\"fsdp\": \"full_shard auto_wrap\", \"fsdp_config\": {\"fsdp_version\": 2, \"reshard_after_forward\": true, \"auto_wrap_policy\": \"TRANSFORMER_BASED_WRAP\", \"cpu_ram_efficient_loading\": true, \"state_dict_type\": \"SHARDED_STATE_DICT\", \"activation_checkpointing\": true}}' \
    --sequence_parallel_size 2 \
    --output_dir /tmp/q1_run
"
```

跑完会**连续撞 6 个错**（每修一个进下一个）。下面 6 关按顺序解。

---

## Quest 0a-1：`Gemma4AudioLayer in the model` ❌ ⭐

```
ValueError: Could not find the transformer layer class Gemma4AudioLayer in the model.
File ".../accelerate/utils/dataclasses.py", line 2059, in __post_init__
```

<details><summary>Hint 1</summary>

`accelerate` 在找 `Gemma4AudioLayer` class，但 model 里没有。`TRANSFORMER_BASED_WRAP` 默认从 `model._no_split_modules` 拿 layer 名字列表。

```bash
docker exec fsdp_sft grep -n '_no_split_modules' /usr/local/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py
```
会看到列表里硬编码了 `["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer", "Gemma4AudioLayer"]`。但 `gemma-4-26B-A4B-i**t**` 是 text variant，**没有 audio encoder**（只有 E2B/E4B 才有）。

</details>

<details><summary>Hint 2</summary>

需要在 fsdp config 里加显式列表覆盖默认。key 名是什么？

</details>

<details><summary>答案</summary>

在 `fsdp_config` JSON 里加：
```json
"transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer"]
```

key 名是 `transformer_layer_cls_to_wrap`（transformers TrainingArguments 接口名），不是 accelerate 文档里的 `transformer_cls_names_to_wrap`。

</details>

---

## Quest 0a-2：`Tensor has no attribute device_mesh` ❌ ⭐

```
AttributeError: 'Tensor' has no attribute 'device_mesh'
File ".../accelerate/utils/fsdp_utils.py", line 537, in <listcomp>
    sharded_state[k] = DTensor.from_local(v, device_mesh=v.device_mesh, ...)
```

<details><summary>Hint 1</summary>

某个参数 `v` 不是 `DTensor`，是普通 `Tensor`。但 accelerate 假设所有 root params 都已经被 shard 成 DTensor。

哪些参数会留在"非 DTensor"状态？gemma4 是 VLM：vision encoder + text decoder + alignment projector + embed_tokens。auto_wrap 只包 transformer layer class（你 0a-1 加的两个），**root-level 的 embed_tokens/lm_head/projector 没被包，自然没 shard**。

那 accelerate 为什么以为它们是 DTensor？看 fsdp_config 里某个 bool。

</details>

<details><summary>答案</summary>

把 `cpu_ram_efficient_loading: true` → **false**。

`true` 时 rank 0 在 CPU 上装满 model 然后 broadcast；root params 没被 shard 的话广播完后还是 plain Tensor → accelerate 把 root 都当 DTensor 处理就崩。

代价：每 rank 自己 load 51.6 GB 模型（主机 RAM 峰值 ≈ 400 GB），加载 ~30 sec → ~90 sec。

</details>

---

## Quest 0a-3：FA2 不支持 `head_dim=512` ❌ ⭐⭐ — 第一次改代码

forward 第 1 步崩：
```
RuntimeError: FlashAttention only supports head dimension at most 256
File ".../modeling_gemma4.py", line 1230, in forward
    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
```

### 背景

gemma4 attention：
- 5 个 sliding window 层，head_dim = 256（FA2/3 都支持）
- 每隔几层一个 **global attention 层，head_dim = 512** —— FA2 max=256，FA3 也 max=256（截至 2026-04，Tri Dao 说 "soon ish"）

`--attn_impl flash_attention_2` 全用 FA2 → global 层炸。

<details><summary>Hint 1</summary>

打开 modeling_gemma4.py 找 `attention_interface = ALL_ATTENTION_FUNCTIONS[`。
你需要在这一行**之前**判断："如果是 global 层（head_dim=512），把 attn_impl 强制改 sdpa"。

怎么知道这层是不是 global？看 `Gemma4TextAttention.__init__` 有什么 layer-type 标记。
```bash
docker exec fsdp_sft grep -n 'layer_type' /usr/local/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py | head -10
```

你会看到 `self.layer_type = config.layer_types[layer_idx]`，值是 `"sliding_attention"` 或 `"full_attention"`。`full_attention` 就是 global（head_dim=512）。

</details>

<details><summary>答案 — modeling_gemma4.py 第 1 处 patch</summary>

把约 1230 行的：
```python
attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
```
替换成：
```python
_impl = self.config._attn_implementation
if _impl in ["flash_attention_2", "flash_attention_3"] and getattr(self, "layer_type", None) == "full_attention":
    _impl = "sdpa"
attention_interface = ALL_ATTENTION_FUNCTIONS[_impl]
```

> 改完别忘了清缓存：
> ```bash
> docker exec fsdp_sft pkill -9 python || true
> docker exec fsdp_sft find /usr/local/lib/python3.12/site-packages/transformers/models/gemma4 -name __pycache__ -exec rm -rf {} +
> ```

</details>

---

## Quest 0a-4：SDPA forward 撞 OOM `8.60 GiB` ❌ ⭐⭐ — 第一次写 sitecustomize

接着撞：
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.60 GiB.
GPU 0 has a total capacity of 79.10 GiB of which 6.42 GiB is free.
```

### 背景

PyTorch SDPA 4 个 backend：FlashAttention（head_dim ≤ 256，禁了）、**mem_efficient (CUTLASS, O(N) mem)**、math (O(N²))、eager。

PyTorch 对 head_dim=512 + bf16 默认选 **math**：
```
seq=16384, head=16, head_dim=512, bf16
attn matrix [16384, 16384, 16] × 2 bytes = 8.6 GiB / layer × 30 layers → OOM
```

<details><summary>Hint 1</summary>

强制用 mem_efficient。process-wide toggle：
```python
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

但 `swift sft` 启动后再设已经晚了（model load 时就 dispatch SDPA 了）。怎么办？

Python 启动时会自动 import `sitecustomize`（如果在 `sys.path` 里）。

</details>

<details><summary>Hint 2</summary>

写一个 `sitecustomize.py` 文件，里面调三个 toggle，启动 swift sft 时把这个 py 所在目录加到 PYTHONPATH 最前面：
```bash
PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble:$PYTHONPATH \
    swift sft ...
```

文件位置约定：`scripts/gemma4_opt/_sdp_preamble/sitecustomize.py`。

</details>

<details><summary>答案 — sitecustomize.py 第 1 个 patch</summary>

```python
"""sitecustomize.py — auto-imported at every Python startup."""
import os
import sys

if os.environ.get("GEMMA4_FORCE_MEM_EFFICIENT_SDP") == "1":
    _is_rank0 = os.environ.get("LOCAL_RANK", "0") == "0"
    try:
        import torch
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        if _is_rank0:
            print(f"[gemma4 sdp_preamble] flash={torch.backends.cuda.flash_sdp_enabled()} "
                  f"mem_eff={torch.backends.cuda.mem_efficient_sdp_enabled()} "
                  f"math={torch.backends.cuda.math_sdp_enabled()}",
                  file=sys.stderr, flush=True)
    except Exception as _e:
        print(f"[gemma4 sdp_preamble] WARN: {_e}", file=sys.stderr, flush=True)
```

**关键点**：
- 用 `GEMMA4_FORCE_MEM_EFFICIENT_SDP=1` 门控（避免影响其他项目）
- print 一定走 `sys.stderr`（**绝对不能走 stdout**！否则后面 bench wrapper 用 `$(python3 ...)` 拿 JSON 会被污染）
- 只 rank 0 print

</details>

启动命令前缀变成：
```bash
PYTHONPATH=...sdp_preamble:$PYTHONPATH GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
    swift sft ...
```

---

## Quest 0a-5：mem_efficient 撞 `Invalid backend` (GQA) ❌ ⭐⭐ — sitecustomize +1

```
RuntimeError: No available kernel. Aborting execution.
INFO: Memory efficient kernel does not support `enable_gqa=True`.
```

### 背景

gemma-4-26B-A4B 是 GQA：`num_attention_heads=16`, `num_global_key_value_heads=2` (8 query 共享 1 KV)。

PyTorch SDPA 处理 GQA 的 KV expansion 有 2 条路：
1. **手动 pre-expand**：caller `repeat_kv` 把 KV 复制成和 Q 一样 head 数
2. **kernel-side expand**：caller 传 `enable_gqa=True`，kernel 自己 broadcast

transformers 5.5 在 PyTorch ≥ 2.5 + `attention_mask is None` 时**自动选路径 2**。但 mem_efficient backend **不支持 `enable_gqa=True`** —— 它要求手动 repeat_kv。

<details><summary>Hint 1</summary>

```bash
docker exec fsdp_sft grep -n 'use_gqa_in_sdpa\|enable_gqa' /usr/local/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py
```
会看到一个函数 `use_gqa_in_sdpa(...)` 控制走哪条路，返回 True 时传 `enable_gqa=True` 给 SDPA。

让它**永远返回 False**，强制 caller 走 repeat_kv pre-expand 路径。

</details>

<details><summary>答案 — sitecustomize.py 第 2 个 patch（追加）</summary>

```python
    # (2) Force GQA pre-expand path (mem_efficient doesn't support enable_gqa kwarg)
    try:
        from transformers.integrations import sdpa_attention as _sdpa_mod
        def _force_no_gqa_kwarg(attention_mask, key):
            return False
        _sdpa_mod.use_gqa_in_sdpa = _force_no_gqa_kwarg
        if _is_rank0:
            print("[gemma4 sdp_preamble] patched use_gqa_in_sdpa → False",
                  file=sys.stderr, flush=True)
    except Exception as _e:
        print(f"[gemma4 sdp_preamble] WARN: skip GQA patch ({_e})",
              file=sys.stderr, flush=True)
```

**显存对比**（gemma4 head_dim=512, seq=16384）：
- math + enable_gqa: **8.6 GiB / 层**（O(N²)）
- mem_efficient + repeat_kv: **384 MiB / 层**（O(N)）
- **53× 节省**

数学 100% 等价。

</details>

---

## Quest 0a-6：backward 撞 `4.27 GiB` (logits OOM) ❌ ⭐⭐⭐ — modeling_gemma4 第 2 处 patch

forward 通了，backward 第 1 步：
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.27 GiB.
File ".../modeling_gemma4.py", line 2487, in forward
    logits = logits.float()
```

### 背景

gemma4 vocab = **262144**（一般 ~32k 的 8 倍）。CE 前 transformers 做 `logits = logits.float()` 把 bf16 logits upcast 到 fp32：
```
[B=1, S=16384, V=262144] × 4 bytes = 17.2 GiB （fp32 logits）
```
+ 此刻 bf16 + fp32 同时活 = 25.7 GB + grad buffer + 4.27 GiB CE 中间结果 → OOM。

<details><summary>Hint 1：能不 upcast 吗？</summary>

不能，CE 在 vocab=262144 上的 sum-exp 在 bf16 下会 NaN。

但是可以 **chunk**：把 `[B*S, V]` 切小块，每块 upcast 到 fp32 算 partial CE 再加。chunk=1024 → 1024 × 262144 × 4 = **1 GiB** 可接受。

</details>

<details><summary>Hint 2：在哪改</summary>

`modeling_gemma4.py` 第 ~2485-2503 行：

```python
logits = logits.float()
shift_logits = logits[..., :-1, :]
...
loss_fct = nn.CrossEntropyLoss()
flat_logits = shift_logits.reshape(-1, self.config.text_config.vocab_size)
flat_labels = shift_labels.view(-1).to(shift_logits.device)
loss = loss_fct(flat_logits, flat_labels)
```

要改成：保留 bf16，循环 chunk-wise upcast + 累加 partial loss，最后除 valid count。

</details>

<details><summary>答案 — modeling_gemma4.py 第 2 处 patch</summary>

替换原版 5 行 CE 为：
```python
shift_logits = logits[..., :-1, :]  # keep bfloat16
shift_labels = labels[..., 1:]
flat_logits = shift_logits.reshape(-1, self.config.text_config.vocab_size)
flat_labels = shift_labels.view(-1).to(flat_logits.device)

import os
_chunk = int(os.environ.get('GEMMA4_CE_CHUNK', '1024'))
_n = flat_logits.shape[0]
_total, _valid = flat_logits.new_zeros(()), 0
for _s in range(0, _n, _chunk):
    _e = min(_s + _chunk, _n)
    _cl = flat_logits[_s:_e].float()
    _ll = flat_labels[_s:_e]
    _total = _total + nn.functional.cross_entropy(
        _cl, _ll, ignore_index=-100, reduction='sum'
    )
    _valid += (_ll != -100).sum().item()
loss = _total / max(_valid, 1)
```

文件顶部记得加 `import os`。`reduction='sum'` 是关键（chunk 间能直接相加），最后除真实 valid count，跟原版 `mean` 数学等价。

</details>

---

## P0a 完成 ✅

到这里 5 步 baseline 应该跑通了。**记下数字**（你会跟后面所有 phase 对比）：

```
P0a baseline (FSDP2 default, AC=on, GBS=4, 5 step):
  step time:   ~1881 ms
  peak mem:    ~43.6 GiB (per rank)
  loss head:   ~2.41 (step 1) → ~2.39 (step 5)
  tokens/step: ~3300 (avg sample 2823 tok × 4 sample / SP=2)
```

如果你的数字相差 ±10% 以内都算正常（数据集差异）。差太多 / 撞错回去检查最后一关（modeling 改错的可能性最高）。

---

# Phase 0c：DeepSpeed prod baseline（无 quest，跑 user 线上命令）

> **目标**：跑 user 线上 production 命令做对照基线，跟 P0a 比看 FSDP2 vs DS。**这步无新代码**，只调用 [`p0_baseline_ds_prod.sh`](../scripts/gemma4_opt/p0_baseline_ds_prod.sh)。

```bash
cd /home/ubuntu/fyh/megatron-sft-recipes
bash scripts/gemma4_opt/p0_baseline_ds_prod.sh
# 跑 ~25 min
```

`report.json` 看：
```jsonc
{
  "mean_step_time_ms": ~86000,        // ≈ 86 sec/step（GAS=16）
  "tokens_per_sec_per_gpu": ~1521,
  "peak_mem_gib_from_swift_log": ~43, // 因为 DS offload(opt+param) 到 CPU
  "actual_total_wall_min": ~20.0,     // 40 step
}
```

理论 full-epoch wall = `18819 / 64 × 86 sec / 60` = **422 min**。

> **重要**：这是后面所有 speedup 的对照基线。把 422 min 记牢。

---

# Phase 0g：FSDP2 ↔ DS 数值对齐

> **目标**：让 P0a 的 FSDP2 跑出来 step 1 loss / grad_norm 跟 P0c 的 DS prod 一致（差 < 0.1%）。**调超参不改代码**，但顺便加一个让吞吐 metric 更准的 patch（quest 0g-token）。

### 不改代码的部分

把 P0a 的 swift sft 命令改成跟 P0c 完全一致（数据顺序、lr=2e-5、warmup_ratio=0.05、`--truncation_strategy right`、`--seed 42`、GBS=64 = MBS=1×GAS=16）。具体见 [`p0_train_align_fsdp2_gbs64.sh`](../scripts/gemma4_opt/p0_train_align_fsdp2_gbs64.sh)。

GAS=16 在 FSDP2 native 下会撞 OOM（FSDP2 no_sync 模式留 unsharded grads），所以 P0g 实操 = **加 fsdp offload + adamw_torch + 解 clip_grad_norm 冲突**。**这就触发 Quest 1-offload**（在 Phase 1 详讲，因为它是 GAS≥2 通用问题）。

如果你不想跑 P0g 完整 GBS=64 align（很慢），可以跳过，直接看下面这个可选 quest 然后进 P1。

### Quest 0g-token：让 logging.jsonl 有 `tokens_this_step`（可选）⭐⭐⭐

到目前为止 swift logging.jsonl 长这样：
```json
{"loss": 2.41, "grad_norm": 0.05, "memory(GiB)": 43.6, "epoch": ..., "global_step": 1, ...}
```
**没有 tokens_this_step 字段**。但是 bench tooling 算 real_TPS 必须有这个。

报告里 `tokens_per_sec_per_gpu` 字段是用 padded 公式 `MAX_LEN × GBS / step_s / NPROC` 算的，packing 后 OK，**non-packing 时假设每 micro 都填满 16k，会虚高 5×**。real_TPS 就是要修这个。

<details><summary>Hint 1：哪边加</summary>

要两端 patch：
- **trainer 端** (`swift/trainers/seq2seq_trainer.py`)：在 `training_step` 里 `inputs['attention_mask'].sum()` 累加，存到 `self.state.token_stats`
- **logger 端** (`swift/trainers/patcher.py`)：在 `add_train_message` 里读 `state.token_stats` emit 到 logs

注意 `swift` 用的是 ms-swift fork（在 `/home/ubuntu/fyh/ms-swift-fork`，`gemma4-complete` branch）。

</details>

<details><summary>Hint 2：SP 因素</summary>

SP=2 时同一 sample 被 2 个 rank 分别处理一半 sequence。所以：
- 每 DP-rank 处理的 token = `attention_mask.sum()` × GAS（GAS 累加）
- per-GPU TPS = tokens / sp_size / step_time
- global TPS = tokens × dp_size / step_time

`dp_size` 在训练前可能是 None，需要 lazy init（首次 training_step 调用时取）。

</details>

<details><summary>答案 — swift/trainers/seq2seq_trainer.py</summary>

`Seq2SeqTrainer.__init__` 末尾加：
```python
        self._step_token_count = 0
        self._last_reset_step = -1
        self._sp_size = getattr(self.template, 'sequence_parallel_size', 1)
        self._dp_size = None  # lazy init
```

`training_step` 改为：
```python
    def training_step(self, model, inputs, *args, **kwargs):
        if self._dp_size is None:
            from swift.sequence_parallel import sequence_parallel
            if self._sp_size > 1 and sequence_parallel.dp_world_size is not None:
                self._dp_size = sequence_parallel.dp_world_size
            else:
                self._dp_size = self.args.world_size // self._sp_size

        if self.state.global_step != self._last_reset_step:
            self._step_token_count = 0
            self._last_reset_step = self.state.global_step

        if 'attention_mask' in inputs:
            self._step_token_count += inputs['attention_mask'].sum().item()
        elif 'input_ids' in inputs:
            self._step_token_count += inputs['input_ids'].numel()

        with self.template.forward_context(self.model, inputs):
            loss = super().training_step(model, inputs, *args, **kwargs)

        if not hasattr(self.state, 'token_stats'):
            self.state.token_stats = {}
        self.state.token_stats['tokens_this_step'] = int(self._step_token_count)
        self.state.token_stats['sp_size'] = self._sp_size
        self.state.token_stats['dp_size'] = self._dp_size
        return loss
```

</details>

<details><summary>答案 — swift/trainers/patcher.py</summary>

`add_train_message()` 函数里、`for k, v in logs.items():` 之前加：
```python
    if hasattr(state, 'token_stats') and state.token_stats and train_speed > 0:
        tokens = state.token_stats.get('tokens_this_step', 0)
        sp_size = state.token_stats.get('sp_size', 1)
        dp_size = state.token_stats.get('dp_size', 1)
        if tokens > 0:
            tokens_per_gpu = round(tokens / sp_size / train_speed, 1)
            tokens_global = round(tokens * dp_size / train_speed, 1)
            logs['tokens_per_gpu_per_sec'] = tokens_per_gpu
            logs['tokens_global_per_sec'] = tokens_global
            logs['tokens_this_step'] = tokens
```

</details>

### 验证

```bash
docker exec fsdp_sft python3 -c "
import inspect
from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
src = inspect.getsource(Seq2SeqTrainer.training_step)
assert '_step_token_count' in src, 'NOT patched!'
print('OK')
"
```

跑一次 P0a 命令，logging.jsonl 应该出现 `tokens_this_step / tokens_per_gpu_per_sec / tokens_global_per_sec`。

---

# Phase 1：GBS sweep

> **目标**：基于 P0g stack 扫 GBS，看哪个 (MBS, GAS) 组合 throughput 最高。

启动脚本 [`p1_gbs_sweep.sh`](../scripts/gemma4_opt/p1_gbs_sweep.sh) 会自动扫这 7 个组合：
- `MBS=1, GAS=1` (GBS=4)
- `MBS=1, GAS=2/4/8/16` (GBS=8/16/32/64)
- `MBS=2, GAS=1/2` (GBS=8/16)

GAS=1 都不开 offload；GAS≥2 自动开 offload（脚本里有逻辑）。

为啥 GAS≥2 必须开 offload？跑一次 `MBS=1 GAS=2` 不开 offload，会撞这个 quest。

## Quest 1-offload：GAS≥2 撞 OOM；开 offload 撞 `clip_grad_norm` ❌ ⭐⭐⭐

不开 offload + GAS=2：
```
torch.OutOfMemoryError: 78.78 GiB peak
```

### 背景：FSDP2 `no_sync` 模式（GAS≥2 时）

PyTorch FSDP2 在 GAS=1 之外的 micro 会进 `no_sync` 模式：**不立即 reduce-scatter grads，而是临时保留 full unsharded grads in memory** 等下一个 micro。代价：**每个 GAS 内 rank 持有完整 grad ≈ 50 GB**（gemma4 25B params × 2 bytes）。这就是 +45 GB 比 GAS=1 多。

解法：**把 grad / opt state offload 到 CPU**。fsdp 串改成：
```json
"fsdp": "full_shard auto_wrap offload"
```

但是！开 offload 后跑下去，optimizer step 前 `clip_grad_norm_` 撞：
```
RuntimeError: No backend type associated with device type cpu
File ".../torch/distributed/_tensor/_dispatch.py", in op
    return self.dispatch(op_call, args, kwargs)
```

### 背景：CPU offload + clip_grad_norm 冲突

FSDP2 `CPUOffloadPolicy` 把 grad 放 CPU 上。`clip_grad_norm_` 内部要算所有 grad 的 vector_norm，这是 `all_reduce(SUM)` —— **NCCL 不支持 CPU tensor**。

PyTorch ≥ 2.6 支持 device-typed backend：
```python
dist.init_process_group(backend="cpu:gloo,cuda:nccl")
```
意思是 CPU collective 走 gloo，GPU 走 NCCL。但 swift / accelerate / torchrun 默认调 `init_process_group(backend="nccl")`。

<details><summary>Hint 1</summary>

monkey-patch `torch.distributed.init_process_group`，把 `backend="nccl"`/`None` 换成 `"cpu:gloo,cuda:nccl"`。

</details>

<details><summary>答案 — sitecustomize.py 第 3 个 patch（追加）</summary>

```python
    # (3) Mixed cpu:gloo,cuda:nccl backend (required for FSDP2 CPU offload + clip_grad_norm)
    try:
        import torch.distributed as _dist
        _orig_init_pg = _dist.init_process_group
        def _patched_init_pg(*args, **kwargs):
            backend = kwargs.get("backend", args[0] if args else None)
            if backend in (None, "nccl"):
                if args:
                    args = ("cpu:gloo,cuda:nccl",) + args[1:]
                else:
                    kwargs["backend"] = "cpu:gloo,cuda:nccl"
                if _is_rank0:
                    print("[gemma4 sdp_preamble] init_pg: nccl → cpu:gloo,cuda:nccl",
                          file=sys.stderr, flush=True)
            return _orig_init_pg(*args, **kwargs)
        _dist.init_process_group = _patched_init_pg
    except Exception as _e:
        print(f"[gemma4 sdp_preamble] WARN: skip mixed-backend patch ({_e})",
              file=sys.stderr, flush=True)
```

**还要**：optimizer 不能用 bnb paged_adamw（它要 CUDA device.index），换 `--optim adamw_torch`。

</details>

### 跑完 P1

```bash
bash scripts/gemma4_opt/p1_gbs_sweep.sh   # ~70 min（7 个配置 × ~10 min 每个）
```

期望结论（项目实测）：**MBS=1 GAS=1 (GBS=4 native) 最快**，1763 ms / step，比 GAS=2 offload 快 3-7 倍。GAS≥2 offload 路径 step time 因为 D2H/H2D 而暴涨。

> **意味着** P0g 那种 GBS=64 offload align 配置只是为了"逐字对齐 DS"，不是性能上的最优。**P1 之后默认 GBS=4 native**，不再 offload。

---

# Phase 2：AC off（无 quest，配置 flip）

> **目标**：关 activation checkpointing 看是否更快（gemma4 avg sample 2823 token，远低于 16k，activation 余量充裕）。

启动 [`p2_no_ac.sh`](../scripts/gemma4_opt/p2_no_ac.sh)。fsdp config 里 `activation_checkpointing: false`。

实测结果：**+6.4% throughput, peak mem 几乎不变**（43.6 → 43.7 GiB）。锁 `AC=off` 进入后续。

---

# Phase 3：reshard=false（无 quest，实测不采纳）

> **目标**：尝试 ZeRO-2 模式（`reshard_after_forward=false` 让 forward 后 param 不再 reshard，省一次 all-gather）。

启动 [`p3_reshard.sh`](../scripts/gemma4_opt/p3_reshard.sh)。

实测：native 路径直接 OOM（params 不 reshard 一直占 GPU）；offload 路径 step time -90%（被 D2H 主导）。**不采纳，回到 reshard=true**。

---

# Phase 4：packing（解锁最大单期增益）

> **目标**：开 `--packing true`，把短样本拼到 16k 满。期望 token 密度 4-6×，throughput 大涨。

加 `--packing true` 跑 P2 stack：

```
ValueError: Template `gemma4` does not support padding free or packing
File ".../swift/pipelines/train/sft.py", line 75, in __init__
```

## Quest 4-template：swift 拒绝 VLM template 用 packing ❌ ⭐⭐

### 背景

swift 的 `--packing` 走 `support_padding_free` 类属性。`Gemma4Template` 是 VLM template，默认 `None` → `not is_multimodal` → False。

但我们 `freeze_vit=true` + 纯文本 dataset，VLM template `_encode()` 内部 `processor(text=..., images=[], videos=[], audios=[])` 在空 list 下**等价于 text-only tokenizer**。所以 packing 在功能上是安全的，只是 swift 默认拒绝。

<details><summary>答案 — sitecustomize.py 第 4 个 patch（追加）</summary>

```python
    # (4) Force Gemma4Template.support_padding_free = True (VLM template, text-only training)
    try:
        from swift.template.templates.gemma import Gemma4Template
        if Gemma4Template.support_padding_free in (None, False):
            Gemma4Template.support_padding_free = True
            if _is_rank0:
                print("[gemma4 sdp_preamble] patched Gemma4Template.support_padding_free → True",
                      file=sys.stderr, flush=True)
    except Exception:
        pass  # don't fail if swift not on path yet
```

**注意**：这个 patch **只在 freeze_vit=true + 纯文本 dataset 下安全**。真有 image 时 packing 会让 attention mask 跨样本错乱。

</details>

### 跑完 P4

```bash
bash scripts/gemma4_opt/p4_packing.sh   # ~12 min
```

期望（项目实测）：
- step time: 1656 ms → **3193 ms**（每 micro 真填 16k 比之前多 5.8×）
- tokens/step: 3300 → **65536**（packing 后每 micro 满 16k × 4 sample）
- throughput per H100 (real): 651 → **2564 tok/s/GPU** (3.94×)
- full-epoch wall: 130 min → **43 min** (3.0×)

**P4 是项目里最大单期增益**。

---

# Phase 5 ★：Liger（最大头，从零写 dispatch）

> **目标**：开 `--use_liger_kernel true`，让 RMSNorm + GeGLU 走 Liger triton fused kernel。期望 +10~15% throughput。

启动 P5 时 `--use_liger_kernel true` 已经打开。但 logging 里看不到任何变化（step ms 不变、peak mem 不变）。

```bash
docker exec fsdp_sft python3 -c "
from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
print(list(MODEL_TYPE_TO_APPLY_LIGER_FN.keys()))
"
# ['gemma', 'gemma2', 'gemma3', 'llama', 'qwen2', ...]  ← 没有 gemma4
```

swift 调 liger 时 `MODEL_TYPE_TO_APPLY_LIGER_FN.get("gemma4", None)` 拿到 None → silent no-op。

## Quest 5-liger：写 liger_gemma4_patch.py ❌ ⭐⭐⭐⭐ — 项目最大头

### 背景：dispatch 函数干啥

参考 gemma3 dispatch（在 `liger_kernel/transformers/monkey_patch.py`），3 件事：
1. **RMSNorm fusion**：`Gemma4RMSNorm.forward` → `LigerRMSNorm.forward`（fp32 fused triton kernel）
2. **GeGLU MLP fusion**：`Gemma4TextMLP.forward` → `LigerGEGLUMLP.forward`（gate × act × up 一个 kernel）
3. **Fused linear-CE (FLCE)**：`Gemma4ForConditionalGeneration.forward` → `gemma3.multimodal_forward`（用 `LigerForCausalLMLoss`，不再 materialize `[B, N, V]` logits，省 8.6 GiB）

### 坑 1：RMSNorm offset 不一样

不同 gemma 版本：
- gemma1 / **gemma4**: `output * weight`，**offset = 0**
- gemma2 / gemma3: `output * (1 + weight)`，offset = 1

LigerRMSNorm 接受 offset 参数。要传 `offset=0.0, casting_mode='gemma', in_place=False`。

### 坑 2：with_scale=False 的 RMSNorm 实例

gemma4 有几个 RMSNorm 实例 `with_scale=False`：`v_norm`, `router.norm`, `embedding_pre_projection_norm`。它们没有 weight 参数，LigerRMSNorm 不支持，**要在 `_patch_one()` 里跳过**。

### 坑 3：FLCE 不能用 `gemma3.causal_forward`

`causal_forward` 假设 `config.final_logit_softcapping`，但 gemma4 没这个字段（`AttributeError`）。

要用 `gemma3.multimodal_forward` —— 它用 `getattr(config.text_config, 'final_logit_softcapping', None)`，gracefully 处理缺失。

### 坑 4：FLCE loss 数值偏高

项目实测：FLCE on 之后 loss 系统性偏高 60-180%（reduction/normalization 在 SP=2 + packing 下不匹配 modeling_gemma4 的 chunked CE）。token_acc 一致 → 模型预测对，只是 loss 值算错。

**解法**：默认关 FLCE（`fused_linear_cross_entropy=False`），只开 RMSNorm + GeGLU。这两个就够给 14.7% 提速。FLCE bug 留作上游 PR 候选。

<details><summary>答案 — liger_gemma4_patch.py 核心结构</summary>

```python
"""Liger Kernel dispatch for gemma4."""
from __future__ import annotations
from typing import Any, Optional


def _apply_liger_kernel_to_gemma4(
    rms_norm: bool = True,
    geglu: bool = True,
    fused_linear_cross_entropy: bool = False,  # off — known reduction bug
    cross_entropy: bool = False,
    rope: bool = False,
    model: Optional[Any] = None,
) -> None:
    if cross_entropy and fused_linear_cross_entropy:
        raise ValueError("cross_entropy and fused_linear_cross_entropy cannot both be True.")

    from functools import partial
    import transformers.models.gemma4.modeling_gemma4 as modeling_gemma4
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4ForConditionalGeneration, Gemma4TextDecoderLayer, Gemma4TextModel,
    )
    from liger_kernel.transformers.geglu import LigerGEGLUMLP
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.monkey_patch import (
        _patch_rms_norm_module, _bind_method_to_module,
    )

    # gemma4 RMSNorm: output * weight (offset=0)
    _patch_gemma4_rms_norm = partial(
        _patch_rms_norm_module, offset=0.0, casting_mode="gemma", in_place=False
    )

    def _patch_one(rms_module):
        if rms_module is None:
            return
        if not getattr(rms_module, "with_scale", True):  # skip no-weight variants
            return
        if not hasattr(rms_module, "weight"):
            return
        _patch_gemma4_rms_norm(rms_module)

    if geglu:
        modeling_gemma4.Gemma4TextMLP.forward = LigerGEGLUMLP.forward

    if fused_linear_cross_entropy:
        try:
            from liger_kernel.transformers.model.gemma3 import multimodal_forward
            modeling_gemma4.Gemma4ForConditionalGeneration.forward = multimodal_forward
        except (ImportError, AttributeError):
            pass

    if model is None:
        return

    # gemma4 VLM layout: model.model.language_model
    inner = getattr(model, "model", model)
    if hasattr(inner, "language_model"):
        text_model = inner.language_model
    else:
        text_model = inner

    if rms_norm:
        _patch_one(getattr(text_model, "norm", None))

    n_patched_rms, n_patched_mlp = 0, 0
    for decoder_layer in getattr(text_model, "layers", []):
        if not isinstance(decoder_layer, Gemma4TextDecoderLayer):
            continue
        if rms_norm:
            for attr in ["input_layernorm", "post_attention_layernorm",
                         "pre_feedforward_layernorm", "post_feedforward_layernorm"]:
                rms = getattr(decoder_layer, attr, None)
                _patch_one(rms)
                if rms is not None:
                    n_patched_rms += 1
            self_attn = getattr(decoder_layer, "self_attn", None)
            if self_attn is not None:
                for attr in ["q_norm", "k_norm", "v_norm"]:
                    _patch_one(getattr(self_attn, attr, None))
        if geglu:
            mlp = getattr(decoder_layer, "mlp", None)
            if mlp is not None and type(mlp).__name__ == "Gemma4TextMLP":
                _bind_method_to_module(mlp, "forward", LigerGEGLUMLP.forward)
                n_patched_mlp += 1

    import logging
    logging.getLogger(__name__).info(
        f"liger_gemma4: patched {n_patched_rms} RMSNorms + {n_patched_mlp} GeGLUs"
    )


def register_gemma4_dispatch():
    try:
        from liger_kernel.transformers import monkey_patch as _mp
        _mp.MODEL_TYPE_TO_APPLY_LIGER_FN["gemma4"] = _apply_liger_kernel_to_gemma4
        return True
    except ImportError:
        return False
```

> 完整版（含更多 RMSNorm 变体如 `post_per_layer_input_norm` 等）见 `scripts/benchmark/liger_gemma4_patch.py.applied`。

</details>

写完后还要在 sitecustomize.py 注册：

<details><summary>答案 — sitecustomize.py 第 5 个 patch（追加）</summary>

```python
    # (5) Register gemma4 → _apply_liger_kernel_to_gemma4 in Liger's dispatch table
    try:
        import sys as _sys
        _sb_path = "/home/ubuntu/fyh/megatron-sft-recipes/scripts/benchmark"
        if _sb_path not in _sys.path:
            _sys.path.insert(0, _sb_path)
        import liger_gemma4_patch
        if liger_gemma4_patch.register_gemma4_dispatch():
            if _is_rank0:
                print("[gemma4 sdp_preamble] registered gemma4 → liger dispatch",
                      file=sys.stderr, flush=True)
    except Exception:
        pass
```

</details>

### 跑完 P5

```bash
bash scripts/gemma4_opt/p5_liger.sh   # ~12 min
```

stdout 应该有 `liger_gemma4: patched 120-210 RMSNorms + 30 GeGLUs`。

期望（项目实测）：
- step time: 3193 → **2781 ms** (-12.9%)
- peak mem: 64.91 GiB（基本不变；FLCE off 没省 logits）
- throughput per H100: 5132 → **5891 tok/s/GPU** (+14.8%)
- full-epoch wall: 43 min → **38 min**

---

# 全部完成 ✅

汇总：从 reset 一路写到这里，你产出了：
- `modeling_gemma4.py` 2 处改动（FA fallback + chunked CE）
- `sitecustomize.py` 5 个 patch（mem_eff SDPA + GQA + mixed backend + template + liger reg）
- `liger_gemma4_patch.py` 1 个新文件
- ms-swift fork 2 处改动（trainer + patcher 加 token stats）

效果：
| Phase | step ms | tok/s/GPU | full-epoch | vs DS prod |
|---|---:|---:|---:|---:|
| DS prod baseline | 86,180 | 1,521 | 422 min | 1.00× |
| P0a FSDP2 (math, AC=on) | 1,881 | 8,710 | 147 min | 2.87× |
| P1 (mem_eff, GBS=4) | 1,763 | 9,295 | 138 min | 3.06× |
| P2 (+AC=off) | 1,656 | 9,893 | 130 min | 3.25× |
| P4 (+packing) | 3,193 | 5,132 | 43 min | 9.81× |
| **P5 ★ (+Liger)** | **2,781** | **5,891** | **37.6 min** | **11.22×** |

> 注意 P4/P5 的 tok/s/GPU 虽然在 padded 公式下看着低，但 packing 后每 micro 真到 16k → real_TPS 大幅领先（看 logging.jsonl 的 `tokens_global_per_sec`）。详见 walkthrough §0.4 末尾。

最终验证完整 P5 stack：
```bash
bash scripts/gemma4_opt/p5_liger.sh
cat /home/ubuntu/fyh/megatron_output/gemma4_opt/p5_liger/run_*/report.json | python3 -m json.tool | head -20
```

期望 `mfu_pct_active_params` ≈ 3.3-3.7。**对得上的话恭喜，你完整复现了项目**。

恢复全部 patch 模式：
```bash
bash scripts/gemma4_opt/diy_restore.sh
```

---

## 中途卡死？

每个 quest 都有"现象 → hint 1 → hint 2 → 答案"渐进路径。如果 hint 1 + 2 都没头绪，直接展开"答案"看代码 + 解释。

不要硬扛 —— 这些 puzzle 个个都是项目里我们卡了 2-12 小时才搞定的真实坑。能在 30-60 min/quest 内消化已经很厉害了。

debug 真实流程的复盘见 [`gemma4_debug_log.md`](gemma4_debug_log.md)（4 段式：现象 → 定位 → 修改 → 验证）。

---

**祝调通！**
