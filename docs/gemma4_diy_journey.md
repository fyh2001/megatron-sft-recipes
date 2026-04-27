# Gemma-4-26B-A4B-it FSDP2 SFT — DIY 踩坑之旅

> 这份文档把"项目完成版"反向拆成 **10 个 puzzle quest**。每个 quest 给你"启动状态 + naive 命令 + 错误现象 + 渐进式 hint"，**答案折叠在 `<details>` 里**，你想看再展开。
>
> 想自己写补丁、自己读 stack trace、自己 debug 的人来这边。想直接复制最终配置上手跑的人去 [`gemma4_setup_from_scratch.md`](gemma4_setup_from_scratch.md)。
>
> **预计耗时**：每个 quest 平均 30-60 min（含 30 step bench 时间），全部 10 关 ≈ 6-10 小时。

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

> 任何时候卡住想看答案：
> ```bash
> diff <your_file> <reference>.applied   # 比对你的版本和参考答案
> ```
> 想直接放弃 DIY：`bash scripts/gemma4_opt/diy_restore.sh` 一键恢复。

---

## Quest 概览

| # | 卡点 | 改什么 | 难度 |
|---|---|---|---|
| 1 | gemma4 default FSDP2 撞 `Gemma4AudioLayer` 类不存在 | 配置（环境变量/JSON） | ⭐ |
| 2 | 撞 `Tensor has no attribute device_mesh` | 配置 | ⭐ |
| 3 | FA2 不支持 `head_dim=512` | **改 modeling_gemma4.py** | ⭐⭐ |
| 4 | SDPA OOM 撞 `8.6 GiB attn matrix` | **写 sitecustomize.py 第 1/5 patch** | ⭐⭐ |
| 5 | mem_efficient backend 撞 `Invalid backend` (GQA) | **写 sitecustomize.py 第 2/5 patch** | ⭐⭐ |
| 6 | logits OOM at vocab=262144 | **改 modeling_gemma4.py 第 2 处** | ⭐⭐⭐ |
| 7 | 想开 packing 但 swift 拒绝 (`does not support padding free`) | **写 sitecustomize.py 第 4/5 patch** | ⭐⭐ |
| 8 | 想开 Liger 但 silent no-op | **写 liger_gemma4_patch.py** | ⭐⭐⭐⭐ |
| 9 | 想开 CPU offload 但 `clip_grad_norm` 撞 `No backend type associated with device type cpu` | **写 sitecustomize.py 第 3/5 patch** | ⭐⭐⭐ |
| 10 | 想算 real_TPS 但 `logging.jsonl` 没 `tokens_this_step` 字段 | **改 ms-swift fork 3 个文件** | ⭐⭐⭐ |

---

## Quest 1：第一次启动就崩 —— audio layer 不存在

### 情境
你刚 reset 完容器，想跑最简单的 FSDP2 baseline 看看。打开 [`p0_baseline_fsdp2.sh`](../scripts/gemma4_opt/p0_baseline_fsdp2.sh) 看长什么样：

```bash
docker exec fsdp_sft bash -lc "
cd /home/ubuntu/fyh/megatron-sft-recipes && \
NPROC_PER_NODE=8 swift sft \
    --model /home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it \
    --model_type gemma4 --template gemma4 \
    --dataset /home/ubuntu/fyh/sft-data/train.jsonl \
    --max_length 16384 --truncation_strategy delete \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_steps 5 --logging_steps 1 \
    --save_strategy no \
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

### 错误
```
File "/usr/local/lib/python3.12/site-packages/accelerate/utils/dataclasses.py", line 2059, in __post_init__
    raise ValueError(
ValueError: Could not find the transformer layer class Gemma4AudioLayer in the model.
```

### Hints
<details><summary>Hint 1（先想想）</summary>

报错说 `accelerate` 在找 `Gemma4AudioLayer` 这个 class，但 model 里没有。

`accelerate` 的 FSDP 包装策略 `TRANSFORMER_BASED_WRAP` 默认从 `model._no_split_modules` 拿 layer class 名字列表。

去 `transformers/models/gemma4/modeling_gemma4.py` 搜 `_no_split_modules =` 看里面写了啥。

</details>

<details><summary>Hint 2</summary>

```bash
docker exec fsdp_sft grep -n '_no_split_modules' /usr/local/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py
```

会看到列表里硬编码了 `["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer", "Gemma4AudioLayer"]`。

但是 `gemma-4-26B-A4B-it`（Audio for B-i**t**ext, A-for-Audio 的 B 变体）**没有 audio encoder**。只有 `E2B` / `E4B` 那两个变体才有。

你需要告诉 accelerate "只用前两个 class，忽略 audio"。在 FSDP config JSON 里加哪个 key？

</details>

<details><summary>答案 + 验证</summary>

**改 fsdp config**：在 `fsdp_config` 里加显式列表覆盖默认：

```json
"transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer"]
```

完整 fsdp 串：
```bash
--fsdp '{"fsdp": "full_shard auto_wrap", "fsdp_config": {"fsdp_version": 2, "reshard_after_forward": true, "auto_wrap_policy": "TRANSFORMER_BASED_WRAP", "cpu_ram_efficient_loading": true, "state_dict_type": "SHARDED_STATE_DICT", "activation_checkpointing": true, "transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer"]}}'
```

**为什么 key 名是 `transformer_layer_cls_to_wrap`，不是 accelerate 文档里的 `transformer_cls_names_to_wrap`？**

因为 swift 走 `transformers.TrainingArguments` → `accelerate`，TrainingArguments 收的是 `transformer_layer_cls_to_wrap`（`accelerate` 内部 alias）。

</details>

### 验证
重跑命令，应该不再撞 `Gemma4AudioLayer` 错误，进入下一个错误（Quest 2）。

---

## Quest 2：FSDP2 加载完模型 → `Tensor has no attribute device_mesh`

### 情境
Quest 1 修了之后，你应该会撞这个：

```
File "/usr/local/lib/python3.12/site-packages/accelerate/utils/fsdp_utils.py", line 537, in <listcomp>
    sharded_state[k] = DTensor.from_local(v, device_mesh=v.device_mesh, placements=v.placements)
                                            ^^^^^^^^^^^^^^
AttributeError: 'Tensor' has no attribute 'device_mesh'
```

### Hints

<details><summary>Hint 1</summary>

错的是某个参数 `v` 不是 `DTensor`，是普通 `Tensor`。但 accelerate 在做 FSDP2 sharded state init 时假设所有参数都已经是 `DTensor`。

哪些参数会留在"非 DTensor"状态？回想 gemma4 是个 **VLM**：vision encoder + text decoder + alignment projector + embed_tokens。auto_wrap 只包 transformer layer class（你 Quest 1 加的 `Gemma4TextDecoderLayer` + `Gemma4VisionEncoderLayer`），**root-level 的 embed_tokens / lm_head / projector 没被包，也就没 shard 成 DTensor**。

那 accelerate 为什么以为它们应该是 DTensor？

</details>

<details><summary>Hint 2</summary>

去翻 fsdp_config 里这个布尔：`cpu_ram_efficient_loading`。

行为：rank 0 在 CPU 上装满整个 model，然后在 FSDP shard 之前 **broadcast 到其他 rank 的 GPU**。这条路径假定参数都属于"将被 shard 的 layer"。但 root-level 参数没被 shard，所以广播完之后还是 plain Tensor，accelerate 后面又把所有 root params 当 DTensor 处理 → 报错。

解法很简单。

</details>

<details><summary>答案</summary>

把 fsdp_config 里：
```json
"cpu_ram_efficient_loading": true
```
改成：
```json
"cpu_ram_efficient_loading": false
```

代价：每个 rank 自己从磁盘加载完整 51.6 GB model（8 张卡 × 51.6 GB ≈ 410 GB 主机 RAM 占用峰值），加载慢一些（≈30 sec → ≈90 sec）。但能跑通。

> 这是 gemma4 / FSDP2 / `cpu_ram_efficient_loading=true` 三方都没正面解决的兼容问题。upstream PR 候选：让 accelerate 在 `cpu_ram_efficient_loading=true` 下识别 root-level non-DTensor 参数。

</details>

### 验证
重跑应该看到 8 个 rank 各自从磁盘 load model（rank0 输出会有 "Loading checkpoint shards"），主机 RAM 涨到 ~400 GB，然后开始 forward。然后撞 Quest 3。

---

## Quest 3：forward 第 1 步崩 —— `head_dim=512 not supported`

### 情境
load 通过了，开始 forward。崩了：

```
RuntimeError: FlashAttention only supports head dimension at most 256
File "/usr/local/lib/python3.12/site-packages/transformers/integrations/flash_attention.py", line 23, in flash_attention_forward
    attn_output, _ = _flash_attention_forward(...)
File "/usr/local/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py", line 1230, in forward
    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
```

### 背景

gemma4 的 attention 设计很特殊：
- 5 个 sliding window 层，head_dim = 256（FA2/3 都支持）
- 然后有 **global attention 层（每隔几层一个），head_dim = 512** — FA2 max head_dim = 256，FA3 max head_dim = 256（截至 2026-04，Tri Dao 说 "soon ish"）

你 `--attn_impl flash_attention_2` 全用 FA2 → global 层炸。

### Hints

<details><summary>Hint 1：怎么定位代码位置</summary>

打开 modeling_gemma4.py，找 `attention_interface = ALL_ATTENTION_FUNCTIONS[`。这是 transformers 里所有 model 共用的 dispatch 模式：拿 `self.config._attn_implementation`（"flash_attention_2"、"sdpa"、"eager" 等）当 key 查 lookup table 拿 callable。

你需要在这一行**之前**判断："如果这一层是 global 层（head_dim=512），强制改 sdpa；否则保持 config 默认"。

怎么知道这层是不是 global？看 `Gemma4TextAttention.__init__` 里有没有什么 layer-type 标记。

</details>

<details><summary>Hint 2：layer_type 是什么</summary>

```bash
docker exec fsdp_sft grep -n 'layer_type' /usr/local/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py | head -10
```

会看到：
- attention 层 `__init__` 里 `self.layer_type = config.layer_types[layer_idx]`
- `config.layer_types` 是个长度=`num_hidden_layers` 的 list，每项是 `"sliding_attention"` 或 `"full_attention"`

`full_attention` 就是 head_dim=512 的 global 层。

</details>

### 你要写的 patch

修改 `modeling_gemma4.py` 第 ~1230 行（Gemma4TextAttention.forward 的 `attention_interface =` 这一行）。

<details><summary>答案</summary>

把这行（约 1230 行）：
```python
attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
```
改成：
```python
_impl = self.config._attn_implementation
if _impl in ["flash_attention_2", "flash_attention_3"] and getattr(self, "layer_type", None) == "full_attention":
    _impl = "sdpa"
attention_interface = ALL_ATTENTION_FUNCTIONS[_impl]
```

逻辑：FA2/FA3 只能跑 sliding 层（head_dim=256），global 层强行 fallback 到 SDPA（不限 head_dim）。

> 改完别忘了重启所有 python 进程（容器里的 swift 运行进程会缓存旧代码）：
> ```bash
> docker exec fsdp_sft pkill -9 python || true
> ```

</details>

### 验证
重跑，应该不再撞 `head_dim 256`。但你会看到 SDPA 接管 global 层后下一个 OOM（Quest 4）。

---

## Quest 4：SDPA forward 第 1 步 OOM —— "Tried to allocate 8.60 GiB"

### 情境
Quest 3 patch 完之后跑：

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.60 GiB.
GPU 0 has a total capacity of 79.10 GiB of which 6.42 GiB is free.
```

8.6 GB 一次性分配，发生在 forward 早期的某个 attention call。

### 背景

PyTorch SDPA 有 4 个 backend，按优先级 dispatch：
1. **FlashAttention** — 最快，但 max head_dim = 256（你刚禁了，因为 head_dim=512）
2. **mem_efficient (CUTLASS)** — O(N) 显存，支持 head_dim=512，但默认在某些情况下被跳过
3. **math** — O(N²) 显存，支持任意 shape
4. **eager** — 纯 python，慢

PyTorch 当前对 head_dim=512 + bf16 的默认选择是 **math**。算一下：seq=16384, head=16, head_dim=512, bf16 → attn matrix `[16384, 16384, 16] × 2 bytes = 8.6 GiB **per layer**`，30 层 forward 全部 alloc 一遍 → OOM。

### Hints

<details><summary>Hint 1</summary>

我们要强迫 PyTorch 用 **mem_efficient** 而不是 math。

PyTorch 提供 process-wide 的 SDPA backend toggle：
- `torch.backends.cuda.enable_flash_sdp(bool)`
- `torch.backends.cuda.enable_mem_efficient_sdp(bool)`
- `torch.backends.cuda.enable_math_sdp(bool)`

如果你只 enable mem_efficient，PyTorch 就会用它。但这些设置在 swift sft 启动后**已经晚了**（model load 时就已经 dispatch 过 SDPA 了）—— 你需要在 **python 进程刚启动**时就设。

Python 有个机制：进程启动时会自动 import 一个叫 `sitecustomize` 的 module（如果在 `sys.path` 里）。

</details>

<details><summary>Hint 2</summary>

写一个 `sitecustomize.py` 文件，里面调用上面三个 toggle，然后在启动 swift sft 时把这个 py 所在目录加到 PYTHONPATH 最前面。

文件位置约定：`scripts/gemma4_opt/_sdp_preamble/sitecustomize.py`

启动命令前缀：
```bash
PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble:$PYTHONPATH \
    swift sft ...
```

</details>

### 你要写的代码

在 `scripts/gemma4_opt/_sdp_preamble/sitecustomize.py` 写一个最小化版本（环境变量 `GEMMA4_FORCE_MEM_EFFICIENT_SDP=1` 才生效，避免影响其他项目）。

<details><summary>答案 — sitecustomize.py 第 1 个 patch</summary>

```python
"""sitecustomize.py — auto-imported by every Python process started under
this directory's PYTHONPATH."""
import os
import sys

if os.environ.get("GEMMA4_FORCE_MEM_EFFICIENT_SDP") == "1":
    _is_rank0 = os.environ.get("LOCAL_RANK", "0") == "0"
    try:
        import torch
        torch.backends.cuda.enable_flash_sdp(False)         # FA2 doesn't support head_dim=512
        torch.backends.cuda.enable_math_sdp(False)          # OOM trigger; O(N²) attn matrix
        torch.backends.cuda.enable_mem_efficient_sdp(True)  # CUTLASS, O(N) mem, head_dim=512 ok
        if _is_rank0:
            print(f"[gemma4 sdp_preamble] flash={torch.backends.cuda.flash_sdp_enabled()} "
                  f"mem_eff={torch.backends.cuda.mem_efficient_sdp_enabled()} "
                  f"math={torch.backends.cuda.math_sdp_enabled()}",
                  file=sys.stderr, flush=True)
    except Exception as _e:
        print(f"[gemma4 sdp_preamble] WARN: skip backend pref ({type(_e).__name__}: {_e})",
              file=sys.stderr, flush=True)
```

**关键点**：
- 用 `if os.environ.get("GEMMA4_FORCE_MEM_EFFICIENT_SDP") == "1":` 门控，避免影响其他项目
- print 一定要走 `sys.stderr`（不能走 stdout！）—— 否则后面 bench wrapper 用 `$(python3 -c "...")` 拿 JSON 时会被 stdout 污染
- 只在 rank 0 print，避免 8 卡刷屏

</details>

### 验证
启动命令前面加：
```bash
PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble:$PYTHONPATH \
GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
```
重跑，应该看到 stderr 上有 `[gemma4 sdp_preamble] flash=False mem_eff=True math=False`，然后崩到 Quest 5。

---

## Quest 5：mem_efficient SDPA 撞 `Invalid backend`

### 情境
Quest 4 装了 sitecustomize 之后跑：

```
RuntimeError: No available kernel. Aborting execution.

INFO: Memory efficient kernel not used because:
- Math kernel disabled
- Flash kernel disabled
- Memory efficient kernel does not support `enable_gqa=True`.
```

mem_efficient 因为某个 `enable_gqa=True` kwarg 被拒了。

### 背景

`gemma-4-26B-A4B` 是 GQA：
- `num_attention_heads = 16`（query heads）
- `num_global_key_value_heads = 2`（KV heads，被 8 个 query head 共享）

PyTorch 的 SDPA dispatch 对 GQA 的 KV expansion 有两种路径：
1. **手动 pre-expand**：caller `torch.repeat_interleave(K/V, repeat_kv_factor, dim=...)` 把 KV 复制成和 Q 一样的 head 数，然后调 SDPA without `enable_gqa`
2. **kernel-side expand**：caller 调 `F.scaled_dot_product_attention(Q, K, V, enable_gqa=True)`，让 kernel 自己 broadcast

transformers 5.5 的 `sdpa_attention.py` 在 PyTorch ≥ 2.5 + `attention_mask is None` 时**自动选路径 2**（因为 kernel 实现更高效）。但 mem_efficient 后端**不支持** `enable_gqa=True` —— 它要求 caller 手动 repeat_kv 后再传。

### Hints

<details><summary>Hint 1：定位</summary>

```bash
docker exec fsdp_sft grep -n 'use_gqa_in_sdpa\|enable_gqa' /usr/local/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py
```

会看到一个函数：
```python
def use_gqa_in_sdpa(attention_mask, key):
    return attention_mask is None and torch_version >= (2, 5)
```

这个函数返回 True 时，wrapper 会传 `enable_gqa=True` 给 SDPA。返回 False 时会先 `repeat_kv` pre-expand。

我们要让它**总是返回 False**，强制 caller 走 repeat_kv pre-expand 路径。

</details>

<details><summary>Hint 2</summary>

monkey-patch `transformers.integrations.sdpa_attention.use_gqa_in_sdpa`，覆盖成永远返回 False。

注意：要在 transformers 被 import 后再 patch，但 `model.from_pretrained` 之前。

简单做法：在 sitecustomize.py 里 import + 替换。Python 的 import 是 lazy 的，但你只要：
```python
from transformers.integrations import sdpa_attention as _sdpa_mod
_sdpa_mod.use_gqa_in_sdpa = ...
```
后续所有 `from .sdpa_attention import use_gqa_in_sdpa` 拿到的都是新函数（因为它是 module attribute lookup）。

</details>

### 你要写的代码

往 `sitecustomize.py` 里**追加**第 2 个 patch（在第 1 个 patch 后面）。

<details><summary>答案 — sitecustomize.py 第 2 个 patch</summary>

```python
    # (2) Monkey-patch transformers GQA-in-SDPA detection to always pre-expand
    #     KV (avoids enable_gqa=True kwarg that mem_efficient backend rejects).
    try:
        from transformers.integrations import sdpa_attention as _sdpa_mod
        def _force_no_gqa_kwarg(attention_mask, key):
            return False
        _sdpa_mod.use_gqa_in_sdpa = _force_no_gqa_kwarg
        if _is_rank0:
            print("[gemma4 sdp_preamble] patched use_gqa_in_sdpa → always False",
                  file=sys.stderr, flush=True)
    except Exception as _e:
        print(f"[gemma4 sdp_preamble] WARN: skip GQA patch ({type(_e).__name__}: {_e})",
              file=sys.stderr, flush=True)
```

**功能等价性**：手动 `repeat_kv` pre-expand 在数学上 100% 等价于 `enable_gqa=True`，只是显存峰值多一份 expanded KV（但 mem_efficient 在 forward 内部立刻丢弃，整体仍 O(N)）。

**显存对比**（gemma4 head_dim=512, seq=16384）：
- math + enable_gqa: **8.6 GiB / 层**（O(N²) attn matrix）
- mem_efficient + repeat_kv: **384 MiB / 层**（O(N) tile-based）
- **53× 节省**

</details>

### 验证
重跑命令，应该 forward 通了，进 backward。然后撞 Quest 6。

---

## Quest 6：backward 撞 `OutOfMemoryError: 4.27 GiB`

### 情境

forward 通了，开始 backward 第 1 步：

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.27 GiB.
File ".../modeling_gemma4.py", line 2487, in forward
    logits = logits.float()
```

### 背景

gemma4 vocab = 262144（比一般 32k vocab 大 8×）。在 CE loss 计算前 transformers 会做 `logits = logits.float()` 把 bf16 logits upcast 到 fp32。算一下：

```
[B=1, S=16384, V=262144] × 4 bytes = 17.2 GiB （fp32 logits）
```

虽然 grad checkpointing 释放了之前的 activation，**这一刻 fp32 logits + bf16 logits 同时活着 = 25.7 GB**，加上 backward 已经积累的 gradient buffer，再开 4.27 GiB 给 CE 中间结果 → OOM。

### Hints

<details><summary>Hint 1：能不能不 upcast 到 fp32？</summary>

不能。CE 在 vocab=262144 上的 sum-exp 在 bf16 下会 NaN（精度不够）。

但是可以 **chunk**：把 `[B*S, V]` 切成小块，每次 chunk × `[chunk, V]` upcast 到 fp32 算一次 partial CE，加起来。这样峰值 fp32 内存 = `chunk * V * 4 bytes`。chunk=1024 → 1024 × 262144 × 4 = **1 GiB**（可以接受）。

</details>

<details><summary>Hint 2：在哪改</summary>

`modeling_gemma4.py` 第 2485-2503 行附近，是 CE loss 计算的核心。原版长这样：

```python
# Upcast to float if we need to compute the loss to avoid potential precision issues
logits = logits.float()
shift_logits = logits[..., :-1, :]
...
loss_fct = nn.CrossEntropyLoss()
flat_logits = shift_logits.reshape(-1, self.config.text_config.vocab_size)
flat_labels = shift_labels.view(-1).to(shift_logits.device)
loss = loss_fct(flat_logits, flat_labels)
```

你要改成：
- 不一次性 `.float()` 整个 logits（保留 bf16）
- 在 flatten 后用循环 chunk-wise upcast + 累加 partial loss
- 用 `reduction='sum'` 累加，最后除以 valid token count

</details>

### 你要写的 patch

修改 `modeling_gemma4.py` 第 ~2485-2503 行的 loss 计算。

<details><summary>答案</summary>

替换原版的 5 行 CE 为：

```python
# Don't upcast logits.float() yet — vocab=262144 makes [B*S, V] fp32 too big.
shift_logits = logits[..., :-1, :]  # keep bfloat16
shift_labels = labels[..., 1:]
flat_logits = shift_logits.reshape(-1, self.config.text_config.vocab_size)
flat_labels = shift_labels.view(-1).to(flat_logits.device)

# Chunked CE: upcast to float per chunk to avoid peak OOM.
import os
_chunk = int(os.environ.get('GEMMA4_CE_CHUNK', '1024'))
_n = flat_logits.shape[0]
_total, _valid = flat_logits.new_zeros(()), 0
for _s in range(0, _n, _chunk):
    _e = min(_s + _chunk, _n)
    _cl = flat_logits[_s:_e].float()  # upcast just this chunk
    _ll = flat_labels[_s:_e]
    _total = _total + nn.functional.cross_entropy(
        _cl, _ll, ignore_index=-100, reduction='sum'
    )
    _valid += (_ll != -100).sum().item()
loss = _total / max(_valid, 1)
```

**关键点**：
- chunk = 1024 token 一切（可调）
- `reduction='sum'` 而不是默认 `'mean'`，否则 chunk 间没法直接相加
- 最后除以**真实 valid token count**（不含 -100 ignore），保证和原版 reduction='mean' 数学等价
- 用 env var 让 chunk size 可调：`GEMMA4_CE_CHUNK=512` 显存更省

记得在文件顶部加 `import os`（第 21 行附近）。

</details>

### 验证
backward 应该不再炸 logits OOM。如果你的 dataset 不大（avg seq < 4k）可能直接跑通 5 步。如果还撞 OOM 但是别的位置，恭喜你解锁 Quest 7+8（packing + Liger 的预热环境）。

---

## Quest 7：你想开 packing 提密度，但 swift 报 `Template gemma4 does not support`

### 情境
搞定 forward+backward 之后你想把 throughput 推高。看 walkthrough 知道 P4 packing 是最大单期增益（5.8× tokens/step）。

加 `--packing true` 到命令里再跑：

```
ValueError: Template `gemma4` does not support padding free or packing
File ".../swift/pipelines/train/sft.py", line 75, in __init__
    raise ValueError(f"Template `{template_meta.template_type}` does not support padding free or packing")
```

### 背景

swift 对 packing 有个保护机制：每个 template 有个 `support_padding_free` 类属性，默认 `None`。`None` 时 swift 会 fallback 到 `not is_multimodal`（多模态默认关）。`Gemma4Template` 是 VLM template → `support_padding_free = False`。

但是！我们的 dataset 是**纯文本对话**，VLM template 的 `_encode()` 内部分支：

```python
def _encode(self, ...):
    ...
    inputs = self.processor(text=texts, images=[], videos=[], audios=[], ...)
    ...
```

当 images/videos/audios 都是空 list 时，processor 实际上**等价于 text-only tokenizer**。所以 packing 在功能上是安全的，只是 swift 默认拒绝。

### Hints

<details><summary>Hint 1：怎么定位</summary>

```bash
docker exec fsdp_sft grep -rn 'support_padding_free' /usr/local/lib/python3.12/site-packages/swift/template/templates/gemma.py
```

会看到 `Gemma4Template` 类定义。它没显式声明 `support_padding_free`，所以走父类 default `None` → 被 swift `or False` 转成 False。

我们要在它身上覆盖一个 `support_padding_free = True`。

</details>

<details><summary>Hint 2</summary>

最干净的做法：在 sitecustomize.py 里再加一个 monkey-patch。

```python
from swift.template.templates.gemma import Gemma4Template
Gemma4Template.support_padding_free = True
```

但要注意：sitecustomize 在 python 启动时就跑，而 swift 此时**可能还没装**。所以要包 try/except，import 失败也别让 sitecustomize 自己崩。

</details>

### 你要写的代码

往 sitecustomize.py 追加第 4 个 patch。

<details><summary>答案 — sitecustomize.py 第 4 个 patch</summary>

```python
    # (4) Force swift's Gemma4Template to declare support_padding_free=True.
    try:
        from swift.template.templates.gemma import Gemma4Template
        if Gemma4Template.support_padding_free in (None, False):
            Gemma4Template.support_padding_free = True
            if _is_rank0:
                print("[gemma4 sdp_preamble] patched Gemma4Template.support_padding_free → True",
                      file=sys.stderr, flush=True)
    except Exception as _e:
        # Don't fail if swift not on path yet (sitecustomize runs early).
        pass
```

**注意**：这个 patch **只在 freeze_vit=true + 纯文本 dataset 下安全**。如果你训练真的有 image，packing 会 break（attention mask 跨样本错乱）。

</details>

### 验证
加 `--packing true` 再跑，应该不再报 `does not support`，进入正常 forward。
观察 log 里 `tokens_per_step` 大幅增加（每个 micro 真到 16k token）。

---

## Quest 8：开 Liger 但完全没动静

### 情境
你看 walkthrough P5 说 Liger 还能再提 14.7%。命令里有 `--use_liger_kernel true`，但日志里好像啥都没变（step time 一样、peak mem 一样）。

为什么？

### Hints

<details><summary>Hint 1</summary>

swift 的 `--use_liger_kernel true` 内部调 `transformers.integrations.liger_kernel.apply_liger_kernel(model, ...)`，再 dispatch 到 liger_kernel 包里的 `MODEL_TYPE_TO_APPLY_LIGER_FN` 字典。

```bash
docker exec fsdp_sft python3 -c "from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN; print(list(MODEL_TYPE_TO_APPLY_LIGER_FN.keys()))"
```

你会看到 `gemma`、`gemma2`、`gemma3`，**但没有 `gemma4`**。

所以 swift 调 liger 时，`MODEL_TYPE_TO_APPLY_LIGER_FN.get("gemma4", None)` 拿到 None，silent fallback no-op。

</details>

<details><summary>Hint 2</summary>

我们要：
1. 写一个 `_apply_liger_kernel_to_gemma4(model, ...)` 函数（gemma4 dispatch）
2. 把它塞进 `MODEL_TYPE_TO_APPLY_LIGER_FN["gemma4"] = _apply_liger_kernel_to_gemma4`

### 关于 (1)：dispatch 函数干啥

参考 gemma3 的 dispatch（在 `liger_kernel/transformers/monkey_patch.py` 里），主要 3 件事：
- **RMSNorm fusion**：把 `Gemma4RMSNorm` 实例的 forward 替换成 `LigerRMSNorm.forward`（fp32 fused kernel）
- **GeGLU MLP fusion**：把 `Gemma4TextMLP.forward` 替换成 `LigerGEGLUMLP.forward`（gate × act × up 一个 kernel）
- **Fused linear-CE**（FLCE）：把 `Gemma4ForConditionalGeneration.forward` 替换成 `gemma3.multimodal_forward`，这个 forward 用 `LigerForCausalLMLoss`，不再 materialize `[B, N, V]` logits

### 关于 RMSNorm 的细节

不同 RMSNorm 版本不一样：
- gemma1: `output * weight`，offset = 0
- gemma2/3: `output * (1 + weight)`，offset = 1
- **gemma4: `output * weight`，offset = 0**（同 gemma1）

LigerRMSNorm 接受 `offset` 参数。要传 `offset=0.0, casting_mode='gemma', in_place=False`。

另外 gemma4 有 **with_scale=False** 的 RMSNorm 实例（`v_norm`、`router.norm`、`embedding_pre_projection_norm`），它们没有 weight 参数。Liger 不支持 with_scale=False，所以要在 `_patch_one()` 里**跳过它们**。

### 关于 FLCE 的坑

不能用 `gemma3.causal_forward` —— 它假设 `config.final_logit_softcapping`，但 gemma4 没有这个字段（attribute error）。

要用 `gemma3.multimodal_forward` —— 它用 `getattr(config.text_config, 'final_logit_softcapping', None)`，gracefully 处理缺失。

但是！我们项目里实测 FLCE on 之后 loss 系统性偏高 60-180%（reduction/normalization 在 SP=2 + packing 下不匹配 modeling_gemma4 的 chunked CE）。所以**默认关 FLCE**，只开 RMSNorm + GeGLU。

</details>

### 你要写的代码

在 `scripts/benchmark/liger_gemma4_patch.py` 写一个新 module。

<details><summary>答案 — liger_gemma4_patch.py（核心结构）</summary>

```python
"""Liger Kernel dispatch for gemma4."""
from __future__ import annotations
from typing import Any, Optional


def _apply_liger_kernel_to_gemma4(
    rms_norm: bool = True,
    geglu: bool = True,
    fused_linear_cross_entropy: bool = False,  # Off by default — known bug
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

    # gemma4 RMSNorm = output * weight (offset=0)
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
            pass  # fall back to chunked CE in modeling patch

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

    n_patched_rms = 0
    n_patched_mlp = 0
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

> 完整版（含更多 RMSNorm 变体如 `post_per_layer_input_norm` 等）见 `scripts/benchmark/liger_gemma4_patch.py.applied`，可 diff 你的版本。

</details>

接下来还要在 sitecustomize.py 里**注册**这个 dispatch。

<details><summary>答案 — sitecustomize.py 第 5 个 patch</summary>

```python
    # (5) Register gemma4 → _apply_liger_kernel_to_gemma4 in Liger's dispatch table.
    try:
        import sys as _sys
        _sb_path = "/home/ubuntu/fyh/megatron-sft-recipes/scripts/benchmark"
        if _sb_path not in _sys.path:
            _sys.path.insert(0, _sb_path)
        import liger_gemma4_patch
        if liger_gemma4_patch.register_gemma4_dispatch():
            if _is_rank0:
                print("[gemma4 sdp_preamble] registered gemma4 → "
                      "_apply_liger_kernel_to_gemma4 in Liger's MODEL_TYPE_TO_APPLY_LIGER_FN",
                      file=sys.stderr, flush=True)
    except Exception:
        pass  # don't fail if liger / swift not yet importable
```

</details>

### 验证

跑完后看 stdout：应该有 `liger_gemma4: patched 120-210 RMSNorms + 30 GeGLUs` 这行。
对比 P4 vs P5 的 `mean_step_time_ms`：应该 -10~14% 左右。

---

## Quest 9：开 CPU offload 撞 `clip_grad_norm` —— `No backend type associated with device type cpu`

### 情境

GAS=2 的时候你想开 CPU offload 节省显存（FSDP2 no_sync 模式下不开会 OOM）。fsdp config 加 `offload`：

```json
"fsdp": "full_shard auto_wrap offload"
```

forward+backward 通过，但优化器步前的 `clip_grad_norm_` 撞了：

```
RuntimeError: No backend type associated with device type cpu
File ".../torch/distributed/_tensor/_dispatch.py", in op
    return self.dispatch(op_call, args, kwargs)
File ".../torch/distributed/c10d_logger.py", line 81, in wrapper
    return func(*args, **kwargs)
```

### 背景

FSDP2 `CPUOffloadPolicy` 把 gradient 也放在 CPU 上。`clip_grad_norm_` 内部要算所有 grad 的 vector_norm，这是个 `all_reduce(SUM)` 的 collective —— 但 grad 是 CPU tensor，**NCCL 不支持 CPU tensor 的 all_reduce**。

PyTorch ≥ 2.6 支持 device-typed backend 语法：
```python
dist.init_process_group(backend="cpu:gloo,cuda:nccl")
```
意思是"CPU 集合通信走 gloo，GPU 走 NCCL"。这样 GPU 训练保持 NCCL 原速度，**CPU 张量的 collective 自动走 gloo**。

但是 swift / accelerate / torchrun 默认调 `init_process_group(backend="nccl")`，没暴露这个开关。

### Hints

<details><summary>Hint 1</summary>

我们要 monkey-patch `torch.distributed.init_process_group`，拦截调用，把 `backend="nccl"`（或默认 None）改成 `backend="cpu:gloo,cuda:nccl"`。

</details>

### 你要写的代码

往 sitecustomize.py 追加第 3 个 patch（顺序无所谓，但项目里编号是 (3/5)）。

<details><summary>答案 — sitecustomize.py 第 3 个 patch</summary>

```python
    # (3) Monkey-patch torch.distributed.init_process_group to use mixed
    #     backend (cpu:gloo + cuda:nccl).  Required when FSDP2 CPUOffloadPolicy
    #     puts gradients on CPU.
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
                    print("[gemma4 sdp_preamble] init_process_group: nccl → cpu:gloo,cuda:nccl",
                          file=sys.stderr, flush=True)
            return _orig_init_pg(*args, **kwargs)
        _dist.init_process_group = _patched_init_pg
    except Exception as _e:
        print(f"[gemma4 sdp_preamble] WARN: skip mixed-backend patch ({_e})",
              file=sys.stderr, flush=True)
```

**注意**：
- `cpu:gloo,cuda:nccl` 是 PyTorch 2.6+ 的 device-typed backend 语法
- gloo 会自动初始化但**不消耗资源**（idle 时空转），所以即使你不用 offload，留这个 patch 也无害
- 如果有时撞 `RuntimeError: gloo not built`，把容器 torch 重装 `pip install torch==2.10.0 --force-reinstall`

</details>

### 验证
fsdp 串里加 `offload`，命令里加 `--optim adamw_torch`（bnb paged_adamw 和 CPU-resident params 不兼容），重跑应该 clip_grad_norm 通过。

---

## Quest 10：tokens_this_step 字段不在 logging.jsonl 里

### 情境
你想算 real_TPS（真实 token throughput，区别于 padded tokens 公式），需要 swift logging.jsonl 里有 `tokens_this_step` 字段。但是默认没有：

```bash
docker exec fsdp_sft python3 -c "
import json
with open('/path/to/logging.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if 'loss' in d:
            print(list(d.keys()))
            break
"
# 输出: ['loss', 'grad_norm', 'learning_rate', 'memory(GiB)', 'epoch', 'global_step', ...]
# 没有 tokens_this_step
```

### 背景
swift 自带的 `Seq2SeqTrainer.training_step()` 只调 `super().training_step()`，**完全不统计 token 数**。要拿 `attention_mask.sum()` 或 `input_ids.numel()`，必须在 caller 拦截 inputs。

logging 这边，`add_train_message()` 只打印 trainer state 已有的字段，没法凭空创造 token 字段。

### Hints

<details><summary>Hint 1：路径设计</summary>

需要两端 patch：
1. **trainer 端**（`swift/trainers/seq2seq_trainer.py`）：在 `training_step` 里统计 `attention_mask.sum()`，存到 `self.state.token_stats` 字典里
2. **logger 端**（`swift/trainers/patcher.py`）：在 `add_train_message` 里读 `state.token_stats`，emit 到 logs

注意 swift 用 ms-swift fork（你的项目里在 `/home/ubuntu/fyh/ms-swift-fork`，commit `gemma4-complete` branch）。

</details>

<details><summary>Hint 2：SP 怎么算 token？</summary>

SP=2 时，**同一个 sample 被两个 rank 分别处理一半 sequence**。所以：
- 每个 DP-rank 处理的 token = `attention_mask.sum()`（每 micro 一次，GAS 累加）
- per-GPU throughput = tokens / sp_size / step_time（除以 SP 因为 2 个 rank 实际共做一份工作）
- global throughput = tokens × dp_size / step_time

`dp_size` 在训练开始前可能是 None（accelerate/swift 延迟初始化），所以要 lazy-init：训练第一步访问时取。

</details>

### 你要改的两个文件

注意你已经在 `/home/ubuntu/fyh/ms-swift-fork` 的 `gemma4-complete` 分支，但 reset 脚本没把这俩文件还原到 upstream 状态（因为 reset 不操作 fork）。**为了真 DIY，先把这俩文件还原**：

```bash
cd /home/ubuntu/fyh/ms-swift-fork
git checkout HEAD~1 -- swift/trainers/seq2seq_trainer.py swift/trainers/patcher.py
# 现在这俩是 commit 44b7aa705 之前的状态（没 token stats）
```

然后你来重新加上 token stats 逻辑。

<details><summary>答案 — swift/trainers/seq2seq_trainer.py</summary>

在 `Seq2SeqTrainer.__init__` 后面加：
```python
        # Token stats
        self._step_token_count = 0
        self._last_reset_step = -1
        from swift.sequence_parallel import sequence_parallel
        self._sp_size = getattr(self.template, 'sequence_parallel_size', 1)
        self._dp_size = None  # lazy init in training_step
```

修改 `training_step` 方法：
```python
    def training_step(self, model, inputs, *args, **kwargs):
        # Lazy init dp_size
        if self._dp_size is None:
            from swift.sequence_parallel import sequence_parallel
            if self._sp_size > 1 and sequence_parallel.dp_world_size is not None:
                self._dp_size = sequence_parallel.dp_world_size
            else:
                self._dp_size = self.args.world_size // self._sp_size

        # Reset token count at the start of each global step
        if self.state.global_step != self._last_reset_step:
            self._step_token_count = 0
            self._last_reset_step = self.state.global_step

        # Count tokens for this micro-batch (BEFORE SP split)
        if 'attention_mask' in inputs:
            self._step_token_count += inputs['attention_mask'].sum().item()
        elif 'input_ids' in inputs:
            self._step_token_count += inputs['input_ids'].numel()

        with self.template.forward_context(self.model, inputs):
            loss = super().training_step(model, inputs, *args, **kwargs)

        # Stash on state for the logger callback to read
        if not hasattr(self.state, 'token_stats'):
            self.state.token_stats = {}
        self.state.token_stats['tokens_this_step'] = int(self._step_token_count)
        self.state.token_stats['sp_size'] = self._sp_size
        self.state.token_stats['dp_size'] = self._dp_size

        return loss
```

</details>

<details><summary>答案 — swift/trainers/patcher.py</summary>

在 `add_train_message()` 函数末尾、`for k, v in logs.items():` 之前加：

```python
    # Token statistics (added by Seq2SeqTrainer.training_step)
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
# 期望: OK
```

跑训练后，`logging.jsonl` 第一行带 loss 的应该有：
```json
{"loss": ..., "tokens_this_step": 16384, "tokens_per_gpu_per_sec": 5891, "tokens_global_per_sec": 47128, ...}
```

---

## Bonus Quest 11：FA3 + Ulysses SP（可选）

ms-swift 自带的 Ulysses SP attention dispatch 只走 FA2，没有 FA3 路径。如果你想让 FA3 也能跟 SP 一起用（虽然 gemma4 用不到，因为 head_dim=512 FA3 也不支持，但其他模型可以），需要在 `swift/sequence_parallel/ulysses.py` 加 FA3 dispatch。

直接看 [`fyh2001/ms-swift@gemma4-complete`](https://github.com/fyh2001/ms-swift/tree/gemma4-complete) 的最新 commit diff（`swift/sequence_parallel/ulysses.py`）即可，逻辑：
```python
origin_fn = (ALL_ATTENTION_FUNCTIONS.get('flash_attention_3_origin') or
            ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'])
return origin_fn(...)
```

---

## 全部 Quest 完成后

跑一个完整的 P5 peak 验证你的所有 patch：
```bash
bash scripts/gemma4_opt/p5_liger.sh
```

期望数字（`report.json`）：
- `mean_step_time_ms`: 2700-2900
- `tokens_per_sec_per_gpu`: 5800-6000（padded 公式）
- `peak_mem_gib_from_swift_log`: 64-65
- `mfu_pct_active_params`: 3.3-3.7

如果数字对得上，**恭喜，你完整复现了项目**。

如果想看完整答案，所有 quest 的 reference 都在：
- `scripts/gemma4_opt/_sdp_preamble/sitecustomize.py.applied`（全 5 个 patch）
- `scripts/benchmark/liger_gemma4_patch.py.applied`
- `scripts/gemma4_opt/gemma4_modeling_compat.patch`（modeling_gemma4 完整 ed-format diff）
- ms-swift fork `gemma4-complete` 分支 commit `44b7aa705`

恢复全部 patch（结束 DIY 模式）：
```bash
bash scripts/gemma4_opt/diy_restore.sh
```

---

## 中途卡死了怎么办？

每一关都有"先看现象 → hint 1 → hint 2 → 答案"的渐进路径。如果 hint 1 + 2 都没头绪，直接展开"答案"看代码 + 解释。

不要硬扛 —— 这 10 个 puzzle 个个都是项目里我们卡了 2-12 小时才搞定的真实坑，能在 30-60 min 内消化已经很厉害了。

debug 真实流程的复盘见 [`gemma4_debug_log.md`](gemma4_debug_log.md)，每个错误都有 4 段式记录（现象 → 定位 → 修改 → 验证）。

---

**祝调通！**
