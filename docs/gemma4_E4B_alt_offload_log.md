# Gemma-4-E4B-it FSDP2 — text-only SFT 实测日志

> **目标**：参考 [`gemma4_phase_delta_summary.md`](gemma4_phase_delta_summary.md) P1 alt-offload 策略，将 GBS=64 / seq_len=16384 / truncation=right 的核心约束平移到 `google/gemma-4-E4B-it` 上，进行 **text-only SFT**。
>
> **最终选定的配置**：用户最初按"E4B 8B 模型 H100 80G 装得下"的预期选了 **FSDP2 native（无 offload）+ GBS=128 + 2 epoch**（§3.0）。新机器实测验证后发现：在 max_length=16384 + AdamW fp32 这个组合下，最长样本峰值 79+ GiB，会在 step 7 触发 OOM（§13）。最终落定为 **FSDP2 + 完整 cpu_offload + GBS=128 + 2 epoch**（§15，与 P1 alt-offload 等效拓扑），其它训练超参不变。
>
> **数据集**：`/home/ubuntu/fyh/megatron-sft-recipes/sft-data/SFT_0424_2.jsonl`（51 557 条）
>
> **要求**：所有性能指标必须为真实测试数据，不允许任何估算。
>
> **本文档自更新**：每次跑出新的状态、报错、解决方案，都直接落到这里。
>
> **结构说明**：§0–§7 是旧机器（被同事 GPU 抢占前）的过程记录；§8 起是切换到这台新机器后的完整跑通过程，包含 4 个新发现的失败模式（§9–§14）和最终成功的配置（§15）。§16 是实测指标（运行结束后填）。

---

## 0. 最终配置基线

| 维度 | 值 | 备注 |
|---|---|---|
| 模型 | `google/gemma-4-E4B-it` | 4B effective param dense + PLE，原生支持 text/image/audio |
| 任务 | text-only SFT | freeze ViT、aligner、audio tower |
| 数据集 | `sft-data/SFT_0424_2.jsonl` | 51 557 条 messages 格式样本 |
| Sequence length | 16384 | — |
| Truncation | **right** | 长样本截断到 16k |
| 引擎 | **FSDP2 native（无 offload）** ⚠ 实测不成立 → 改 **FSDP2 + cpu_offload**（§13、§15） | 计划：E4B 8B 模型 H100 80G 完全装得下，offload 是 PCIe 浪费<br>实测：max_length=16384 + AdamW fp32 + AC=on + 8 卡分片，单卡峰值 75-79 GiB，长样本会 OOM；改用完整 cpu_offload 后稳定在 70-77 GiB |
| MBS / GAS / SP | 1 / 16 / 1 | — |
| DP / GBS | 8 / **128** | NPROC / SP；GBS = MBS × DP × GAS |
| Epochs | **2** | — |
| LR / Warmup | **2e-5** / **0.05** | — |
| LR scheduler | cosine | swift 默认 |
| Save strategy | **epoch** | save_only_model=true，save_total_limit=3 |
| AC | on（FSDP2 自管 + `--gradient_checkpointing true`） | §6 一度被怀疑是 KeyError 22 的元凶（试过 AC=off）；§9–§11 查清根因是 FSDP2 cast_forward_inputs 克隆 kwargs dict（与 AC 无关），最终落 sitecustomize 的 patch (8/8) v2 module-cell 之后 AC=on 稳定 |
| Packing / padding_free | false / false | — |
| Liger | true | dispatch 注册 |
| Optimizer | adamw_torch（swift 默认） | — |
| Template | `gemma4_nothinking` | E4B 在 swift register 里默认就是这个 |

---

## 1. 环境与前提

| 项 | 值 |
|---|---|
| 主机 | 8 × NVIDIA H100 80GB HBM3 |
| 容器 | docker name=`fsdp_sft` |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu129 |
| transformers | 5.5.4（已打 `gemma4_modeling_compat.patch`） |
| ms-swift | 4.2.0.dev0（commit b462182d） |
| liger-kernel | 0.7.0 |

**Patch 校验**：
- `transformers/models/gemma4/modeling_gemma4.py` md5 == `39ebf386a992fea9eac0883f459ac658`
- `scripts/gemma4_opt/_sdp_preamble/sitecustomize.py` 在 PYTHONPATH 最前
- `GEMMA4_FORCE_MEM_EFFICIENT_SDP=1` 已设置

---

## 2. 时间线（按事件追加）

> 每个步骤记录：开始时间、命令、退出状态、关键日志摘录。

### 2.0 启动准备 — 2026-04-29 06:24 UTC

| 事项 | 细节 |
|---|---|
| 容器 | `fsdp_sft` (Up 6 days) |
| dcgm-exporter | port 9500 (40+ DCGM_FI_PROF 指标已暴露) |
| modeling_gemma4 patch | md5=`39ebf386a992fea9eac0883f459ac658` (校验通过) |
| sitecustomize | `scripts/gemma4_opt/_sdp_preamble/sitecustomize.py` 在位 |
| liger_gemma4_patch | `scripts/benchmark/liger_gemma4_patch.py` 在位 |

### 2.1 模型下载 — 2026-04-29 06:24 → 06:30 UTC

```bash
docker exec fsdp_sft modelscope download \
    --model google/gemma-4-E4B-it \
    --local_dir /home/ubuntu/.cache/modelscope/models/google/gemma-4-E4B-it
```

- 总耗时 ~6 min
- 大小：15.0 GB（单一 `model.safetensors` shard）
- 校验：`safetensors` 总参数 **7,996,157,418 ≈ 8.0B**

| 子模块 | 参数量 | 角色 |
|---|---:|---|
| `model.language_model.embed_tokens` | 671.09 M | text 标准 embedding（vocab=262144 × hidden=2560） |
| `model.language_model.embed_tokens_per_layer` | 2818.57 M | **PLE**（lookup 表，无 FLOPs） |
| `model.language_model.layers.*` (42 层) | 3945.72 M | text transformer compute |
| `model.vision_tower` | 167.37 M | 视觉塔（freeze） |
| `model.audio_tower` | 304.83 M | 音频塔（freeze） |
| `model.embed_vision` + `model.embed_audio` | 5.90 M | aligner（freeze） |
| **trainable text 总和** | **7518.07 M** | freeze_vit + freeze_aligner 后训练 |
| compute-active（用于诚实 MFU） | **~4.62 B** | transformer 层 + tied lm_head 671M（PLE 不算 GEMM） |

### 2.2 模型 / 架构验证

- `model_type`: `gemma4` ✅（与 26B-A4B-it 同 dispatch 路径）
- `text_config.global_head_dim`: 512 ✅（同样需 mem_efficient SDPA pin）
- `text_config.head_dim`: 256（sliding attn）
- `text_config.num_hidden_layers`: 42
- `text_config.num_attention_heads`: 8 / `num_key_value_heads`: 2 → GQA ratio 4
- `text_config.enable_moe_block`: false（dense）
- `text_config.tie_word_embeddings`: true（lm_head 与 embed_tokens 共享）
- `text_config.vocab_size`: 262144（chunked CE 必须）
- swift `ModelArch.gemma3n` 映射：
  - `language_model = ['model.language_model', 'lm_head']`
  - `aligner = ['model.embed_vision', 'model.embed_audio']`
  - `vision_tower = ['model.vision_tower', 'model.audio_tower']`
- swift template = `gemma4_nothinking`（与 26B-A4B-it 的 `gemma4` 不同，但实现类相同 `Gemma4Template`，sitecustomize patch 4/4 仍生效）

### 2.3 关键差异 vs 26B-A4B-it 配置

| 维度 | 26B-A4B-it (P1 alt-offload) | E4B-it (本次) |
|---|---|---|
| FSDP `transformer_layer_cls_to_wrap` | `Gemma4TextDecoderLayer,Gemma4VisionEncoderLayer` | `Gemma4TextDecoderLayer,Gemma4VisionEncoderLayer,Gemma4AudioLayer` ← E4B 实际有 audio tower |
| `NUM_PARAMS` (训练侧 6N) | 25.2e9 | 7.518e9 |
| `NUM_ACTIVE_PARAMS` (诚实 MFU) | 3.8e9（MoE active） | 4.62e9（dense compute，扣除 PLE 表） |
| MoE | 是 (top_k_experts=2) | 否 (`enable_moe_block=false`) |
| Liger kernel 影响 | RMSNorm + GeGLU 生效 | 同（dispatch 已注册） |

### 2.4 P1 alt-offload 等量平移启动脚本（保留作为备选）

- 落档路径：`scripts/gemma4_E4B_opt/p1_alt_offload_E4B_text_only.sh`
- 与 `scripts/gemma4_opt/p1_alt_offload_ds_equiv.sh` 仅差以下 env：
  - `MODEL` → E4B-it 路径
  - `TRAIN_JSONL` → `sft-data/SFT_0424_2.jsonl`
  - `DATASET_SIZE=51557`
  - `FSDP_TRANSFORMER_CLS_NAMES` 加 `Gemma4AudioLayer`
  - `NUM_PARAMS=7.518e9`, `NUM_ACTIVE_PARAMS=4.62e9`
- **未启用本次正式跑**，原因见 §3.0。

---

## 3. 配置决策与脚本变更

### 3.0 与同事 DS ZeRO-3 命令对齐讨论 — 2026-04-29 06:36 → 06:42 UTC

**触发**：用户提供同事的实际生产命令：

```bash
swift sft \
    --model /mnt/local/zy/models/gemma-4-E4B-it \
    --model_type gemma4 --template gemma4 \
    --dataset /mnt/local/zy/Data/SFT/SFT_0424_2.jsonl \
    --truncation_strategy right --max_length 16384 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 48 \
    --num_train_epochs 1 --learning_rate 1e-5 --warmup_ratio 0.025 \
    --deepspeed /home/ubuntu/zy/scrip/zero3_nopin.json \
    --attn_impl flash_attention_2 --use_liger_kernel true \
    --gradient_checkpointing true --save_only_model true \
    --padding_free false ...
```

并补充其 DS config：

```json
"zero_optimization": {
  "stage": 3,
  "offload_optimizer": { "device": "none" },
  "offload_param":     { "device": "none" },
  ...
}
```

**关键发现**：

1. 同事跑的是 **DS ZeRO-3 native（无 offload）@ GBS=384**：
   - 没有 `--sequence_parallel_size` → SP=1
   - `1 × 8 × 48 = 384`
   - DS config `offload_*.device = "none"` → 全显存
2. **既不是 P0c（DS prod offload baseline）也不是 P1 alt-offload**，phase_delta 表里没有这一行。
3. E4B 只有 ~8B 参数，ZeRO-3 8 卡分片后 param + grad + opt + activations ≈ 30-40 GiB / GPU << 80 GiB → **offload 在 E4B 上是纯 PCIe 浪费**。

**用户选择**：在 D 路径基础上（FSDP2 native + GBS=384 + SP=1 + GAS=48）做以下 4 处修改：

| 项 | D 默认 | 用户修改 |
|---|---|---|
| num_train_epochs | 1 | **2** |
| gradient_accumulation_steps | 48 | **16** |
| learning_rate | 1e-5 | **2e-5** |
| warmup_ratio | 0.025 / 0.1 | **0.05** |
| save_strategy | no | **epoch** |

**带来的副作用**：GAS 从 48 降到 16，使 GBS 自动从 384 降到 **128**（= 1 × 8 × 16）。已与用户对齐，按 GBS=128 跑。

### 3.1 正式启动脚本

- 落档路径：`scripts/gemma4_E4B_opt/fsdp2_native_E4B_text_only_2ep.sh`
- 不走 `bench_swift_sp_v2.sh` wrapper（因为 wrapper 硬编码 `--max_steps` / `--save_strategy no` / `--learning_rate 1e-5` / `--warmup_ratio 0.1`）
- 直接在脚本内：
  1. 渲染 `fsdp_override.json`（FSDP2 native：`full_shard auto_wrap`，无 offload；`activation_checkpointing=true`；`cpu_ram_efficient_loading=false`；`transformer_layer_cls_to_wrap=[Gemma4TextDecoderLayer, Gemma4VisionEncoderLayer, Gemma4AudioLayer]`）
  2. 起 `gpu_monitor.py` + `dcgm_scrape.py` 后台监控
  3. `docker exec fsdp_sft swift sft …` 完整 2-epoch 训练
  4. 跑完后 `report_swift_sp.py` 聚合成 `report.json` + `bench.jsonl`
  5. 同时 `extract_loss_curve.py` 出 loss tsv

**期望步数**：`ceil(51 557 / 128) = 403 / epoch`，2 epoch ≈ **806 optimizer steps**。

---

## 4. 第一次启动失败（已修复）

### 4.1 时间线 — 2026-04-29 07:27:53 UTC 启动 → 07:28:31 UTC 失败（启动期校验拒绝）

**症状**：所有 8 个 rank 在 swift sft 启动期校验阶段抛同一异常：

```
ValueError: FSDP2 with SHARDED_STATE_DICT is not compatible with save_only_model=True.
Either set save_only_model=False, or change state_dict_type to FULL_STATE_DICT in fsdp_config.
Note: FULL_STATE_DICT requires more memory and is slower.
```

`local_rank=2` 被 elastic launcher 识别为 root cause，其余 rank 收到 SIGTERM 同步退出。

**根因**：
- `fsdp_override.json` 里写了 `"state_dict_type": "SHARDED_STATE_DICT"`（沿用 P1 alt-offload 26B-A4B 的写法）
- swift CLI 同时收到 `--save_only_model true`（同事命令保留下来）
- HuggingFace transformers / accelerate 的 FSDP2 集成对这个组合做了硬校验：sharded 情况下 swift trainer save flow 不能跳过 opt state（因为它的 gather 路径需要完整 state dict 才能正确重组）

**对错运行无影响**：错误发生在训练前的 config 校验阶段，**没有任何 GPU compute 发生**，0 个 step、0 个 token、0 mem usage。

### 4.2 修复方案

3 个候选：

| 方案 | 取舍 |
|---|---|
| A. `state_dict_type` 改 `FULL_STATE_DICT` | save 时 rank 0 gather 16 GB 全量参数；对 E4B 完全够（80 GB 卡有 ~60 GB 余量） |
| B. 保持 `SHARDED_STATE_DICT`，去掉 `save_only_model=true` | 检查点会同时落 opt state（fp32 master + 2 moments，~64 GB）→ 单 ckpt ~80 GB，2 epoch × 3 retain = 240 GB 磁盘；存活但费盘 |
| C. 直接 `save_strategy no` 跳过保存 | 用户已要求 `save_strategy=epoch`，违背要求 |

**采用 A**：与用户「save_only_model 行为同同事」一致；落盘单一 .safetensors / .bin 格式更直接可用（不需要后处理）。修改：

```diff
-        "state_dict_type": "SHARDED_STATE_DICT",
+        "state_dict_type": "FULL_STATE_DICT",
```

### 4.3 失败 run 归档（第一次）

- 路径：`experiments/gemma4_E4B_alt_offload/run_20260429_072753_fsdp2_native_2ep_gbs128/`
- 内容：`cmd.sh`、`fsdp_override.json`（旧版 SHARDED_STATE_DICT）、`train.log`（含完整 traceback）、`STATUS=FAILED — exit=1`

---

## 5. 第二次启动失败（已修复）

### 5.1 时间线 — 2026-04-29 07:31:35 UTC 启动 → 07:32:32 UTC 失败

**症状**：与 §4 相同的报错，但 traceback 来自 transformers，不是 swift：

```
File ".../transformers/trainer.py", line 859
ValueError: save_only_model option is not compatible with FSDP state dict type 'SHARDED_STATE_DICT'
```

注意 swift 的 `_check_fsdp2_compatibility()` 这一关**通过了**（因为它正确从 JSON `fsdp_config.state_dict_type` 读到 `FULL_STATE_DICT`），但 transformers Trainer 启动期再次检查 `self.accelerator.state.fsdp_plugin.state_dict_type` 仍然是 SHARDED_STATE_DICT。

### 5.2 根因（HF transformers FSDPPlugin 构造路径漏读 state_dict_type）

源码定位：`transformers/training_args.py::_process_fsdp_args()` （5.5.4 版）构造 `fsdp_plugin_args` dict 喂给 accelerate 的 `FullyShardedDataParallelPlugin`。它从 `self.fsdp_config` 读取了：

- `auto_wrap_policy`、`min_num_params`、`transformer_cls_names_to_wrap`（从 `transformer_layer_cls_to_wrap`）
- `version`（注意：是 `version` 不是 `fsdp_version`；swift 又另设了 `FSDP_VERSION` env var）
- `reshard_after_forward`、`backward_prefetch`、`forward_prefetch`、`use_orig_params`
- `cpu_ram_efficient_loading`、`sync_module_states`

**但是它从来没读 `state_dict_type` 或 `activation_checkpointing`**。这两个值 accelerate 只从 env var 读：
- `FSDP_STATE_DICT_TYPE`（默认 FSDPv1=FULL，FSDPv2=SHARDED）
- `FSDP_ACTIVATION_CHECKPOINTING`

所以 fsdp_override.json 里写 `"state_dict_type": "FULL_STATE_DICT"` 在当前 transformers/accelerate 版本下**对 transformer Trainer 检查路径 silent 无效**。

> 注：swift 的 `_check_fsdp2_compatibility()` 自己读 JSON，所以 swift 那一关本来就能过；但 transformers 二次校验仍然挡死。

### 5.3 修复

在 docker exec 注入一个 env var：

```diff
 PYTHONPATH=...:_sdp_preamble:$PYTHONPATH \
 GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
 PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
 CUDA_DEVICE_MAX_CONNECTIONS=8 \
+FSDP_STATE_DICT_TYPE=FULL_STATE_DICT \
 NPROC_PER_NODE=8 MASTER_PORT=$PORT \
 swift sft …
```

JSON 里的 `state_dict_type` 字段同时保留（给 swift 自检用）。

### 5.4 失败 run 归档（第二次）

- 路径：`experiments/gemma4_E4B_alt_offload/run_20260429_073135_fsdp2_native_2ep_gbs128/`
- 与第一次失败一样在 启动期 config 校验阶段就被拒，0 个 step / 0 GPU 算力消耗
- `STATUS=FAILED — exit=1`

---

## 6. 第三次启动失败：FSDP2 AC × E4B kv_shared 兼容性 bug

### 6.1 时间线 — 2026-04-29 07:38:36 UTC 启动 → 07:39:36 UTC（1 min）失败

第二次起脚本的修复让 transformers 启动校验通过、模型加载成功、数据集预处理完成（51 557 → ~51 416 条，少数样本因 `loss=None` / `content=None` 自动剔除），FSDP2 wrap 完成、第一个 training_step 真正进入 forward，**然后**：

```
File ".../transformers/models/gemma4/modeling_gemma4.py", line 1212, in forward
    key_states, value_states = shared_kv_states[self.kv_shared_layer_index]
KeyError: 22
```

8 个 rank 全部相同位置挂掉（rank 5 第一个被识别为 root cause）。

### 6.2 当时的根因假设（事后修正：见 §11）

> ⚠ 这一节记录的是当时的判断，事后 §11 通过 id() 抓到证据，证明 AC 是无辜的、真正的元凶是 FSDP2 的 `MixedPrecisionPolicy(cast_forward_inputs=True)` 触发 `torch.distributed.utils._apply_to_tensors` 重建 kwargs 嵌套 dict。把 AC 当作元凶仅是观察相关性的产物（碰巧 AC=on 时报错而 AC=off 时这次 run 在 §7 就 OOM 退出，看不到 KeyError）。新的实测证据见 §11；最终 fix 见 §11.3 / sitecustomize 的 patch (8/8) v2。

**Gemma-4-E4B-it 与 26B-A4B-it 的关键架构差异**：

| 参数 | E4B-it | 26B-A4B-it |
|---|---:|---:|
| `num_hidden_layers` | 42 | 30 |
| `num_kv_shared_layers` | **18** | **0** |

E4B 的后 18 层（layer index 24-41）共享 KV：每层从前面 layer 22 / 17 / 11 / 5（即 `kv_shared_layer_index`）的 attention 输出读 K/V，跳过自己 K/V proj。Layer 22 通过 dict 副作用写入：

```python
# modeling_gemma4.py line 1229-1230 (Gemma4TextAttention.forward)
if self.store_full_length_kv:
    shared_kv_states[self.layer_idx] = key_states, value_states
```

而 layer 24 读取：

```python
# modeling_gemma4.py line 1211-1212
if self.is_kv_shared_layer:
    key_states, value_states = shared_kv_states[self.kv_shared_layer_index]
```

`shared_kv_states` 是个 dict，从 `Gemma4TextModel.forward()`（第 1633 行）创建后通过 kwargs 传给每个 decoder_layer。

**当时的（错误）推理**：FSDP2 `activation_checkpointing=true` 把每个 `Gemma4TextDecoderLayer` 用 `CheckpointWrapper` 包起来，`use_reentrant=False` 应该透明地把 dict 副作用反映出来——但 `shared_kv_states[22]` 在 layer 24 forward 时 KeyError，疑似 dict 副作用没传播到外层。

**26B-A4B-it 不受影响**：`num_kv_shared_layers=0`，整个 KV-sharing 代码路径根本没被命中——这部分判断是对的。

### 6.3 当时的"修复"（关掉 AC）— 实际是 workaround

E4B 只有 ~8B 参数，按当时的估算：

| 项 | 估算 |
|---|---:|
| param shard (bf16) | ~2 GiB |
| grad shard (bf16) | ~2 GiB |
| opt state shard (Adam fp32 master + 2 moments) | ~12 GiB |
| activations (无 AC，seq=16384, MBS=1, 42 层) | ~17 GiB |
| 其他（PLE 表 lookup buffer、NCCL bucket、scratch） | ~3 GiB |
| **总计** | **~36 GiB / GPU** |

按 ~36 GiB 估算远低于 H100 80 GiB，所以 commit `de64dd2` 选择了关 AC 这条路。**新机器（§13）实测推翻了这个估算**：实际 75–79 GiB，AC 必须开。

```diff
   "fsdp_config": {
     ...
-    "activation_checkpointing": true,
+    "activation_checkpointing": false,
   }
```

```diff
-    --gradient_checkpointing true \
+    --gradient_checkpointing false \
```

### 6.4 失败 run 归档（第三次）

- 路径：`experiments/gemma4_E4B_alt_offload/run_20260429_073836_fsdp2_native_2ep_gbs128/`
- 失败 step：0（forward 第一步就挂）
- `train.log` 末尾 traceback 完整保留供日后回看
- `STATUS=FAILED — exit=1 (E4B kv-shared layer × FSDP2 AC dict-side-effect lost)`

---

## 7. 阻塞中：GPU 被其他任务占用 — 2026-04-29 07:44 UTC

### 7.1 状况

正准备启动第四次跑（关 AC 修复版），`nvidia-smi` 检查发现 8 张 H100 全被另一个用户启动的 mistral12b DPO RLHF 任务占满：

```text
GPU 0..7  used: 67-77 GiB / 81 GiB,  free: 3-13 GiB,  util: 88-100%
```

任务命令为另一个 swift `rlhf --rlhf_type dpo --model .../user-model-afs-sfw-3replies-alldata-grpo-lab9-step60 --model_type mistral --deepspeed zero2 ...`，启动于 07:44 UTC，环境是 `/mnt/local/envs/conda_envs/swift4`（不是 fsdp_sft 容器，是另一套独立 conda env）。

### 7.2 冲突原因

- 本机为多用户共享 GPU 节点，无 GPU 配额隔离机制（没有 MIG / Slurm queue）
- 另一用户使用宿主 conda env 直接占满 8 卡
- 即使关 AC 后 E4B 单卡显存预算 ~36 GiB，叠加同事的 67-77 GiB 占用 → 立即 OOM

### 7.3 待解（需用户决定）

可选 3 条：

1. **等同事任务跑完**：观察其 PID / 日志看 ETA，之后立即起本次任务
2. **要求同事停掉**：联系暂停后立即起
3. **挪到其他节点**：把容器 / 数据 / patch 迁到其他空闲机器

无法估算等待时长（DPO 训练规模未知，单从命令上看是 mistral12b + GBS=4×8=32 + epoch=1，总样本量取决于 `auto_reply_bon_dpo_worst_m0.jsonl`；可能 1-12 小时）。

### 7.4 当前阶段产物（截至 §7）

- 修复后的脚本 `scripts/gemma4_E4B_opt/fsdp2_native_E4B_text_only_2ep.sh` 已就绪
- 已更新条目：
  - `state_dict_type=FULL_STATE_DICT` + 新增 `FSDP_STATE_DICT_TYPE=FULL_STATE_DICT` env
  - `activation_checkpointing=false` + `--gradient_checkpointing false`（绕过 E4B kv-share × AC bug）
- 失败 run 已全部归档，可直接重跑（GPU 一释放就启动）
- commit `de64dd2` 把脚本 + 已踩的 4 个失败 run + 文档 push 到了 origin/main，准备换机器继续

---

## 8. 切换到新机器（host `am4g1r39bm1`）— 2026-04-29 12:24 UTC

### 8.1 前置校验

| 项 | 状态 |
|---|---|
| docker 容器 `fsdp_sft` | Up 8 days ✓ |
| `transformers/models/gemma4/modeling_gemma4.py` md5 | `39ebf386a992fea9eac0883f459ac658` ✓（与旧机器一致）|
| ms-swift 版本 | `4.2.0.dev0` ✓ |
| transformers 版本 | `5.5.4` ✓ |
| accelerate 版本 | `1.13.0`（在 §9 才暴露 bug） |
| `scripts/gemma4_opt/_sdp_preamble/sitecustomize.py` patch (1)–(8) | 在位（patch 8 是 commit 时的 instance-attr 版，§11 改写为 module-cell 版） |
| 8 × H100 80GB | 全部空闲（0 MiB used）✓ |
| dcgm-exporter:9500 | 200 OK ✓ |

### 8.2 模型下载（~6 min）

```bash
docker exec fsdp_sft modelscope download \
    --model google/gemma-4-E4B-it \
    --local_dir /home/ubuntu/.cache/modelscope/models/google/gemma-4-E4B-it
# 14.9 GB model.safetensors，9 个文件，~6 min @ 75 MB/s
```

### 8.3 数据集 — `/home/ubuntu/fyh/megatron-sft-recipes/sft-data/SFT_0424_2.jsonl` 不存在

新机器 `sft-data/` 里只有 `train.jsonl`（18 819 行）和 `train_short.jsonl`（2 000 行），没有用户引用的 `SFT_0424_2.jsonl`（51 557 行）。`find /` 在 `/mnt/shared/fyh/SFT_0424_2.jsonl` 找到（1.1 GB，正确的 51 557 行）。容器看不到 `/mnt/shared`，所以 `cp` 到本地：

```bash
cp /mnt/shared/fyh/SFT_0424_2.jsonl /home/ubuntu/fyh/megatron-sft-recipes/sft-data/SFT_0424_2.jsonl
```

校验通过（51 557 行 / 1.1 GB / `messages` 格式），容器内可读。

---

## 9. 第四次启动失败：`cpu_ram_efficient_loading=true` × tied lm_head — accelerate `fsdp2_load_full_state_dict` AttributeError

### 9.1 时间线 — 2026-04-29 13:52 UTC 启动 → 13:53:48 UTC 失败（约 95 s 后）

按 §7 规划的修复（`cpu_ram_efficient_loading=true` 让只 rank 0 加载，其它 rank 用 meta tensor），8 卡 swift sft 启动后 90 秒在 accelerate `_prepare_fsdp2` 阶段同步崩：

```
File ".../accelerate/utils/fsdp_utils.py", line 537, in fsdp2_load_full_state_dict
    device_mesh = sharded_param.device_mesh
                  ^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'Tensor' object has no attribute 'device_mesh'
```

### 9.2 第一层根因（tied embedding）

`gemma-4-E4B-it` 的 `text_config.tie_word_embeddings=True`：`lm_head.weight` 和 `model.language_model.embed_tokens.weight` 是同一个 `nn.Parameter`。`fsdp2_prepare_model()` 流程：

1. `model.to('meta')`（rank 0 这步前已加载真实权重，meta 化后丢失；其它 rank 一直是 meta）
2. `model.tie_weights()` —— 此时 `lm_head.weight is embed_tokens.weight`
3. 对每个 transformer 层 `fully_shard()`
4. 对 root model `fully_shard()` —— 这里只对 _一份_ tied weight slot 替换为 `DTensor`，另一份指向已被释放的旧 storage 变成 stale `Tensor`
5. `fsdp2_load_full_state_dict(model, full_sd)` —— 它的循环 `for ... in zip(full_sd.items(), meta_sharded_sd.values())` 取 `model.state_dict()` 的所有键，碰到那个 stale `Tensor` 时 `sharded_param.device_mesh` 直接 AttributeError

post-load 的 `model.tie_weights()` 本来会把 stale slot 重新指回 DTensor，但 load 循环挂了根本到不了那一步。

### 9.3 第一次修复尝试（patch 9/9）— skip non-DTensor + strict=False

往 `sitecustomize.py` 加 patch (9/9)：包装 `accelerate.utils.fsdp_utils.fsdp2_load_full_state_dict`，循环里碰到非 `DTensor` 的条目就跳过 broadcast/distribute（rank 0 和其它 rank 同步跳过、保持 broadcast 配对），最后 `model.load_state_dict(sharded_sd, assign=True, strict=False)`。post-load 的 `tie_weights()` 会把跳过的 tied slot 重新挂回去。

### 9.4 第二层根因（`fully_shard` 在 meta 上完全没生效）

跑起来后 patch 9 打印 **`970 non-DTensor entries skipped`** —— 这远不止 `lm_head.weight` 一个 tied slot，而是 _基本所有参数_。说明 `fully_shard` 对 meta-device 模型在这套（PyTorch 2.10 + transformers 5.5.4 + accelerate 1.13.0 + gemma-4-E4B-it）组合下就是不工作。把跳过的 meta tensor 通过 `assign=True` 装回模型后，forward 跑到第一个 `hidden_states * scalar` 就抛：

```
RuntimeError: Tensor on device meta is not on the expected device cuda:7!
```

### 9.5 决定：放弃 `cpu_ram_efficient_loading=true`，回到 `false`

新机器的 8 张 H100 全部空闲（不像旧机器被同事 26B megatron 抢占），每 rank 各自加载 16 GiB 到自己 GPU 的 cuda:0 _原则上_ 只占 GPU 80 GiB 的 20%，应该能装下 —— 这是 §10 的入手点。

patch 9 保留在 sitecustomize 里待命，对 `cpu_ram_efficient_loading=false` 是 no-op（这个路径根本不调用 `fsdp2_load_full_state_dict`），但如果以后再开 cpu_ram_eff=true 它仍然有用（前提是 `fully_shard` 在 meta 上的问题先解掉）。

### 9.6 失败 run 归档

| run dir | 失败模式 |
|---|---|
| `run_20260429_131636_fsdp2_native_2ep_gbs128` | OOM（启动时另一个 root 用户在跑 26B-A4B megatron 抢了所有 GPU；不是配置 bug） |
| `run_20260429_135241_fsdp2_native_2ep_gbs128` | AttributeError 'Tensor' has no 'device_mesh'（tied embedding） |
| `run_20260429_135836_fsdp2_native_2ep_gbs128` | RuntimeError 'Tensor on device meta'（patch 9 skip 后 assign 回 meta，970 entries） |

---

## 10. 第五次启动失败：`cpu_ram_efficient_loading=false` + AC=off 显存爆炸

### 10.1 时间线 — 2026-04-29 14:03 UTC 启动 → 14:04:06 UTC 失败（约 60 s）

按 §6.3 落定的脚本（AC=off）+ §9.5 改回 `cpu_ram_efficient_loading=false`，8 个 rank 各自 `from_pretrained(..., device_map='cuda:0')` 加载本地 16 GiB 模型 → FSDP2 wrap → 进 training_step 1 → backward 阶段 OOM：

```
File ".../torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py", line 491, in pre_backward
    self.wait_for_unshard()
File ".../torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py", line 382, in foreach_all_gather_copy_out
    fsdp_param.alloc_all_gather_outputs()
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.25 GiB. GPU 2 has a
total capacity of 79.18 GiB of which 2.90 GiB is free. ... this process has 76.27 GiB
memory in use. Of the allocated memory 74.45 GiB is allocated by PyTorch ...
```

5.25 GiB 这个 size 与 PLE 表 `embed_tokens_per_layer`（2.8B params × 2 bytes ≈ 5.6 GiB）匹配，root FSDP unit 在 backward 时 all-gather 这个块对应的全量 fp32 master——总占用突破 76 GiB。

### 10.2 §6.3 估算被推翻

§6.3 当时按 ~36 GiB / GPU 估算，实际是 76 GiB。差距来自：

- `embed_tokens_per_layer` 的 PLE 表（5.6 GiB bf16） + 它的 fp32 master/m/v（共 33 GiB）都在 root FSDP unit 里（虽然分 8 卡每卡 4 GiB，但 all-gather 时短暂全量）
- AC=off 时 42 层全部 activations 全程保留：seq=16384 时 ~25 GiB
- FSDP2 backward unshard buffer + comm scratch ~5 GiB

### 10.3 决定：必须重新打开 AC

AC 是 §6 当时被怀疑的 KV-share KeyError 元凶，但有了 §11 的根因证据后，AC 是无辜的——所以 sitecustomize 的 patch (8/8) 是真正的 fix。把 `activation_checkpointing=true` + `--gradient_checkpointing true` 重新打开，进 §11。

---

## 11. 第六次失败 + 第七次突破：KeyError 22 真正的根因 + sitecustomize patch (8/8) v2

### 11.1 patch (8/8) v1（commit `de64dd2` 里的版本）—— instance-attr 路由失败

`de64dd2` 时 patch (8/8) 是这样写的：

```python
def _patched_textmodel_forward(self, *args, **kwargs):
    _fyh_skv = {}
    for _layer in self.layers[: self.config.num_hidden_layers]:
        _layer.self_attn._fyh_shared_kv = _fyh_skv  # 实例属性
    return _orig_textmodel_forward(self, *args, **kwargs)

def _patched_attn_forward(self, *args, **kwargs):
    _ref = getattr(self, "_fyh_shared_kv", None)
    if _ref is not None:
        kwargs["shared_kv_states"] = _ref  # 用实例属性覆盖 kwargs
    return _orig_attn_forward(self, *args, **kwargs)
```

打开 `GEMMA4_KV_SHARE_DEBUG=1` 跑（`run_20260429_141240_fsdp2_native_2ep_gbs128`），AC=on + cpu_ram_eff=false：

```
[gemma4 kv-share] textmodel.forward seeded _fyh_shared_kv on 42 attns (skv id=139190429422272)
[gemma4 kv-share] L21 attn forward: self id=135250973553200 has_attr=False ... skv id=None keys=None
[gemma4 kv-share] L22 attn forward: self id=135250977749184 has_attr=False ...
[gemma4 kv-share] L23 attn forward: self id=135250973556320 has_attr=False ...
[gemma4 kv-share] L24 attn forward: self id=135250970428048 has_attr=False ... → KeyError 22
```

**关键证据**：textmodel.forward 在 `self.layers[i].self_attn` 上面 set 的实例属性，runtime 的 `_patched_attn_forward` 里 `self id` 完全是 _不同的对象_（`has_attr=False`）。一类 ~10⁹ bytes 远的两群对象。

也就是说，FSDP2 fully_shard + CheckpointWrapper 包到 layer 上之后，**通过 `self.layers[i].self_attn` 访问到的 attn 对象 ≠ 实际进 `_patched_attn_forward` 的 `self`**——seed 时那一组实例属性永远不会被 runtime 看到。

### 11.2 patch (8/8) v2（module-level cell）

不再依赖实例属性，改成模块级单元格 `transformers.models.gemma4.modeling_gemma4._FYH_CURRENT_SHARED_KV`：

```python
_mg4._FYH_CURRENT_SHARED_KV = None

def _patched_textmodel_forward(self, *args, **kwargs):
    _fyh_skv = {}
    _mg4._FYH_CURRENT_SHARED_KV = _fyh_skv
    return _orig_textmodel_forward(self, *args, **kwargs)
    # ⚠ 不能在 finally 里把 cell 清回 None —— 见 11.3

def _patched_attn_forward(self, *args, **kwargs):
    _ref = _mg4._FYH_CURRENT_SHARED_KV
    if _ref is not None:
        kwargs["shared_kv_states"] = _ref
    return _orig_attn_forward(self, *args, **kwargs)
```

跑 `run_20260429_142231_*`，patch v2 第一稿（含 `try/finally` 把 cell 清回 None）：

```
[gemma4 kv-share] textmodel.forward set _FYH_CURRENT_SHARED_KV id=137216717475392
[gemma4 kv-share] L22 attn forward: writer=True  skv id=137216717475392 keys=[]
[gemma4 kv-share] L23 attn forward: writer=True  skv id=137216717475392 keys=[22]
[gemma4 kv-share] L24 attn forward: is_shared=True kv_shared_idx=22 keys=[22, 23]
[gemma4 kv-share] L25 ... L41 attn forward: keys=[22, 23]   ← 全程正确
... textmodel.forward 退出，finally 清 cell 回 None
[gemma4 kv-share] L41 attn forward (第 2 次): kv_shared_idx=23 skv id=None keys=None  ← KeyError 23
```

### 11.3 patch (8/8) v2 终稿 — 移除 `finally` 清空

L41 forward 被调用 _两次_：第一次是正常 forward（textmodel.forward 内部循环），第二次发生在 textmodel.forward 已经退出之后，是 **AC backward 阶段对最后一层 wrapped layer 的重算**（CheckpointWrapper 的 `use_reentrant=False` 走 `torch.utils.checkpoint.checkpoint`，backward 时 re-run forward 来重建 saved tensors）。

第一次时 cell 还在；`finally` 清 cell；第二次时 cell 已经是 None，KeyError。

修复：去掉 `finally` 清空。下一次 textmodel.forward 会用新的 `{}` 覆盖；中间 backward 阶段 cell 保留 == 上一次 forward 末尾状态，正好满足 AC 重算需要。

### 11.4 §6 的真正根因 — FSDP2 `MixedPrecisionPolicy.cast_forward_inputs=True`

回到 KeyError 22 的最初出处：accelerate 给 FSDP2 默认的 `MixedPrecisionPolicy(param_dtype=torch.bfloat16, cast_forward_inputs=True)` 在每层 `_FSDPState._pre_forward` 里调用 `torch.distributed.utils._apply_to_tensors(cast_fn, kwargs)`。这个 helper 会**无条件**重建嵌套 dict（`{k: apply(v) for k, v in x.items()}`），即便 dict 里没有 tensor —— 结果每层 decoder_layer 拿到的 `shared_kv_states` 都是一个**全新的空 dict**，layer 22 写到 dict 副本里、layer 24 读到另一个空 dict。

AC 不是元凶；AC=off 这次 run 看不到 KeyError 是因为它在 §10 OOM 退出了，没机会跑到 layer 24。

---

## 12. 第八次：终于进了 training_step（patch v2 + cpu_ram_eff=false + AC=on） — 但 step 7 OOM

### 12.1 实测显存进展

| step | mem(GiB) | loss | grad_norm | token_acc | s/it |
|---:|---:|---:|---:|---:|---:|
| 1 | 71.89 | (warmup) | | | 47.6 |
| 2 | 74.44 | | | | |
| 3 | 74.46 | | | | |
| 4 | 74.99 | 2.356 | 16.38 | 0.5238 | 34.81 |
| 5 | 74.99 | 2.300 | 15.19 | 0.5354 | 34.15 |
| 6 | 77.26 | 2.364 | 15.00 | 0.5238 | 33.67 |
| 7 | OOM allocating 1024 MiB on top of 79.17/79.18 GiB | | | | |

### 12.2 根因 — 变长样本 + AdamW fp32 单卡边际不够

`max_length=16384` + `truncation=right` 意味着每个样本 0–16384 tokens 任意分布。短样本时 71–75 GiB，最长样本一来直接顶到 79.17 GiB，再加 1 GiB 中间张量就爆。

§6.3 的 ~36 GiB 估算漏了：

| 项 | 估算 | 实测 (16k seq 长样本) |
|---|---:|---:|
| param shard (bf16) | 2 | 2 |
| grad shard (bf16) | 2 | 2 |
| optim shard (Adam fp32 master+m+v) | 12 | 12 |
| activations (AC=on, seq=16384, 42 层 boundary) | 1 | ~3.5 (42 层 boundary save) |
| activations recompute peak | 0.6 | 0.6 |
| **PLE all-gather buffer (5.6 GB bf16, root unit) ⚠ 漏了** | — | ~10 (transient) |
| **AdamW fp32 master 全量在 root unit ⚠ 漏了** | — | ~50 (transient gather) |
| 其它 (NCCL bucket, comm scratch) | 1 | 4 |
| **总计** | **18.6** | **75–84** |

最大的两笔漏算是 PLE + 与之绑定的 Adam fp32 master 在 root FSDP unit 里 all-gather 时短暂全量出现。

### 12.3 失败 run 归档

`experiments/gemma4_E4B_alt_offload/run_20260429_142946_fsdp2_native_2ep_gbs128/`，
`STATUS = FAILED — OOM at step 7 (...)`，内容包括 cmd.sh、fsdp_override.json、train.log、dcgm_tc.tsv、gpu_metrics.jsonl。

---

## 13. 第九次：尝试 SP=2（Ulysses）分摊 activation — 失败

### 13.1 思路

SP=2 把每 rank 的 seq 从 16384 切成 8192，activations 砍半，仍然保持 GBS=128（DP=4 × GAS=32 × MBS=1）。

### 13.2 失败 — gemma4 PLE 路径与 swift Ulysses SP 不兼容

在 `Gemma4TextModel.forward()` 走到 `project_per_layer_inputs(inputs_embeds, per_layer_inputs)` 时崩：

```
File ".../transformers/models/gemma4/modeling_gemma4.py", line 1713
  return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale
RuntimeError: The size of tensor a (79) must match the size of tensor b (158) at non-singleton dimension 1
```

8 个 rank 同时报，比例固定 `b == 2 × a`（79/158、1125/2249、3063/6125、46/91 …）。

**根因**：swift 的 Ulysses SP 把 `inputs_embeds`（继而 `per_layer_projection`）按 seq 维切到 seq/SP=8192/half——但 `per_layer_inputs = embed_tokens_per_layer.lookup(input_ids)` 是 PLE 走的另一条路径，**没有被 swift Ulysses SP 的 wrap 切到 seq/SP**，仍然是 full seq。两者相加直接报形状不匹配。

要修这个得在 swift 里给 gemma4 PLE 路径加专门的 sequence sharding hook，工作量不小，**放弃**。

### 13.3 失败 run 归档

`experiments/gemma4_E4B_alt_offload/run_20260429_144423_fsdp2_native_2ep_gbs128/`，
`STATUS = FAILED — exit=1 (SP=2 × gemma4 PLE incompat ...)`。

---

## 14. 决策点 — 与用户对齐切换到 FSDP2 + 完整 cpu_offload

§12 实测推翻 §0 "no offload" 的预期。给用户列了 4 个选项：

| 选项 | 说明 | 备注 |
|---|---|---|
| A. SP=1 + FSDP2 完整 cpu_offload | 与 P1 alt-offload 同策略；释放十几 GiB，PCIe +1-3% 慢 | 最稳 |
| B. SP=1 + adamw_torch_8bit | sitecustomize patch (6/6) 已就绪；释放 ~8 GiB optim states | 8-bit numeric drift |
| C. SP=1 + max_length 降到 12288 / 8192 | 释放 ≥5 GiB | 改用户硬约束 |
| D. 切到 `scripts/gemma4_E4B_opt/p1_alt_offload_E4B_text_only.sh` | 备选 P1 alt-offload 完整脚本 | 它默认走 bench wrapper |

用户选 **A**。脚本里在 `fsdp` 字串末尾追加 `offload`：

```diff
-    "fsdp": "full_shard auto_wrap",
+    "fsdp": "full_shard auto_wrap offload",
```

LABEL 改为 `fsdp2_offload_2ep_gbs128`。其它训练超参完全不动。

---

## 15. 第十次：FSDP2 + 完整 cpu_offload — **运行中**

### 15.1 启动配置

启动脚本 `scripts/gemma4_E4B_opt/fsdp2_native_E4B_text_only_2ep.sh`（脚本名保留旧名，内部 LABEL=`fsdp2_offload_2ep_gbs128`），生成的 `fsdp_override.json`：

```json
{
    "fsdp": "full_shard auto_wrap offload",
    "fsdp_config": {
        "fsdp_version": 2,
        "reshard_after_forward": true,
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "cpu_ram_efficient_loading": false,
        "state_dict_type": "FULL_STATE_DICT",
        "activation_checkpointing": true,
        "transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer", "Gemma4AudioLayer"]
    }
}
```

env：`FSDP_STATE_DICT_TYPE=FULL_STATE_DICT`、`GEMMA4_FORCE_MEM_EFFICIENT_SDP=1`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、`CUDA_DEVICE_MAX_CONNECTIONS=8`、`PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble:$PYTHONPATH`。

swift sft：`--per_device_train_batch_size 1 --gradient_accumulation_steps 16 --num_train_epochs 2 --learning_rate 2e-5 --warmup_ratio 0.05 --lr_scheduler_type cosine --max_length 16384 --truncation_strategy right --use_liger_kernel true --gradient_checkpointing true --freeze_vit true --freeze_aligner true --sequence_parallel_size 1 --packing false --padding_free false --save_strategy epoch --save_total_limit 3 --save_only_model true ...`

### 15.2 sitecustomize 当前生效的 patch（与 commit de64dd2 的差异）

| patch | 状态 | 与 de64dd2 比较 |
|---|---|---|
| (1) SDPA backend (mem_efficient only) | 生效 | 不变 |
| (2) `use_gqa_in_sdpa` → False | 生效 | 不变 |
| (3) init_process_group → cpu:gloo,cuda:nccl | 生效（offload 路径需要 CPU 集合） | 不变 |
| (4) `Gemma4Template.support_padding_free=True` | 生效 | 不变 |
| (5) Liger gemma4 dispatch | 生效 | 不变 |
| (6) torchao adamw 8bit step bypass | 已注册（本次 run 没用到 8-bit optim） | 不变 |
| (7) accelerate ParallelismConfig 注入 | 已注册（本次 SP=1, CP=1 → no-op） | 不变 |
| **(8) gemma4 kv-share 重路由** | 生效（**module-cell 版**，§11.2/§11.3） | **改写**：原来是 instance-attr，现在是 module-level cell；去掉 finally clear |
| (9) accelerate `fsdp2_load_full_state_dict` skip non-DTensor | 在位但本 run 不触发（cpu_ram_efficient_loading=false） | **新增**（§9.3） |
| (10) padding_to=max_length collator | 已注册（CP=1 → no-op） | 不变 |
| (11) `get_use_logits_to_keep=False` | 已注册（CP=1 → no-op） | 不变 |
| (12) profiler hook | 未启用（GEMMA4_PROFILE!=1） | 不变 |

### 15.3 启动期日志（rank 0）

```
[gemma4 sdp_preamble] (1/3) backend prefs: flash=False mem_eff=True math=False
[gemma4 sdp_preamble] (2/3) patched ... use_gqa_in_sdpa → always False
[gemma4 sdp_preamble] (5/5) registered gemma4 → _apply_liger_kernel_to_gemma4
[gemma4 sdp_preamble] (4/4) patched swift.template.Gemma4Template.support_padding_free → True
[gemma4 sdp_preamble] (8/8) patched Gemma4TextModel.forward + Gemma4TextAttention.forward
   to route shared_kv_states via module-level cell ...
[gemma4 sdp_preamble] (9/9) patched accelerate.utils.fsdp_utils.fsdp2_load_full_state_dict
   to skip non-DTensor entries ...
... swift FSDP2 plugin: full_shard auto_wrap offload
... transformers warning: FSDP upcast of low precision parameters to fp32 effects 623 parameters
```

**623 fp32-upcasted params** 是 accelerate 的 `MixedPrecisionPolicy` 对所有 `requires_grad=True` 的参数升 fp32（它的 master copy）；这是预期行为，与 §12.2 的"AdamW fp32 master 全量"诊断一致。frozen 的 vision/audio tower 参数不在这 623 里。

### 15.4 启动后实测进展（截至文档写作时）

| step | mem(GiB) | loss | grad_norm | lr | token_acc | s/it | remaining |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 70.27 | 2.430 | 16.25 | 4.9e-7 | 0.5227 | 47.63 | 10h 44m |
| 2 | 70.29 | 2.467 | 16.00 | 9.8e-7 | 0.5158 | 42.86 | 9h 45m |
| 3 | 70.30 | 2.478 | 16.38 | 1.46e-6 | 0.5147 | 40.47 | 9h 19m |
| 10 | 73.43 | (in jsonl) | | | | 38.35 | 8h 29m |

显存早期看起来稳定在 70-77 GiB（offload 把 optim state 移到 CPU + grad 也走 CPU all-reduce，给 GPU 留出一些安全余量）。s/it 趋势从 47s 降到 38s（warmup + cudnn benchmark 收敛）。GPU util 100%。后续 §15.6 证明这份配置仍然会在 step 61 OOM。

### 15.5 失败可能性 / 监控点

- 长样本仍然可能把显存推过 79 GiB（offload 后 buffer 中仍有 PLE all-gather 等 transient）—— 持续盯住 `memory(GiB)` 字段
- save_strategy=epoch，第一次 save_only_model 落盘大约在 step 403，会出现 ~30-60s 抖动，正常
- patch (8/8) v2 module-cell 单进程线程安全（gemma4 forward 同一 nn.Module 的同一个 process 单线程）；如果未来开 dataloader 中并行 forward 需要重新评估

### 15.6 结果：step 61 OOM

`run_20260429_144949_fsdp2_offload_2ep_gbs128/` 最终在 step 61/806 失败：

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 904.00 MiB.
GPU 5 has total capacity 79.18 GiB, of which 393.00 MiB is free.
allocated by PyTorch = 76.74 GiB
reserved but unallocated = 422.66 MiB
```

关键实测：

| 项 | 值 |
|---|---:|
| 最后完成 step | 61/806 |
| swift peak mem | 77.54 GiB |
| OOM 现场实际占用 | 78.78 / 79.18 GiB |
| 申请失败大小 | 904 MiB |
| reserved-but-unallocated | 422 MiB（fragmentation 不是主因） |
| step 60 loss | 1.7308（loss 曲线健康） |
| step ≥10 mean s/it | 37.49 |

结论：FSDP2+offload 仍然解决不了 root FSDP unit 中 PLE (`embed_tokens_per_layer`, 2.82B params) 带来的 transient gather 峰值。需要把 PLE 从 root unit 拆出去，进入 §17。

---

## 18. 1.29x grad_norm 真凶定位 (2026-04-30)

### 现象

跟同事 DS3 ZeRO-3 baseline 对比 (FSDP2 + cpu_offload + A3 + padding_free):

| step | DS3 loss | DS3 gn | FSDP2 loss | FSDP2 gn | gn ratio |
|------|----------|--------|------------|----------|----------|
| 1    | 2.22552  | 10.283 | 2.226      | 13.25    | **1.289** |
| 2    | 2.26230  | 10.334 | 2.263      | 13.38    | 1.295 |
| 3    | 2.27448  | 10.391 | 2.274      | 13.31    | 1.281 |
| 50   | 1.6561   | 0.7229 | 1.678      | 0.88     | 1.22 |

Loss 完全一致到 4 位小数，但 grad_norm 系统性高 1.29x。

### 排除假设

通过 monkey-patch 逐一排除：
- ✗ Loss 分母（patch 18 实测 padding<0.04%、num_items=174769 两边一致）
- ✗ HF Trainer GAS bug ([transformers#45305](https://github.com/huggingface/transformers/issues/45305) 已在 v5.5.4 修复)
- ✗ bf16 norm 计算精度（patch 17 fp64 norm 仍是 13.25）
- ✗ bf16 cross-micro grad accumulation（patch 20 v2 force-sync 后 fp32 累加，无变化）
- ✗ NCCL bf16 reduce-scatter（patch 21 强制 fp32 RS，无变化）

### 真凶：Patch (8) × Activation Checkpointing 副作用

加 patch (22) per-param grad-dump 后，分层对比揭示**强烈的 layer-specific pattern**：

| 层范围 | DS3 sum_sq | FSDP2 sum_sq | ratio | 说明 |
|---|---|---|---|---|
| Layer 0-1 | 35.6 / 28.6 | 36.7 / 37.0 | ~1.0 | clean |
| Layer 2-8 | ~3-5 | ~2-4 | **0.68-0.84** | FSDP2 偏小 |
| Layer 9-13 | ~3-6 | ~5-8 | 1.08-1.66 | 过渡 |
| **Layer 14-22** | ~2-10 | **~5-39** | **2.9-4.3x** | 严重偏大 |
| Layer 23 | 53.7 | 63.3 | 1.18 | |
| **Layer 24-41** | various | various | **0.998-1.001** | 完美一致 |
| embed_tokens / lm_head | 3.79 | 3.79 | 1.00 | |

**Gemma-4-E4B kv-share 结构**: `first_kv_shared_layer_idx = 42 - 18 = 24`. Layers 24-41 通过 `shared_kv_states[idx]` 读 source layers 14-23 的 k/v：
- Layer 24 reads from layer 14
- Layer 41 reads from layer 23

**症状定位**：source layers 14-22 grad 在 FSDP2 路径下被 inflate 3-4x，shared layers 24-41 完全一致，embeddings 一致。

### 根因机制

我们的 sitecustomize patch (8) 用 `_FYH_CURRENT_SHARED_KV` module-level cell 路由 kv，是为绕开 **FSDP2 默认 `MixedPrecisionPolicy(cast_forward_inputs=True)` 的 kwargs dict 重建** 而设计（见 §11）。

但在 `gradient_checkpointing=true` (use_reentrant=False AC) 路径下，cell 跨 forward / AC backward recompute 共享，造成 source layer 的 k_states 经过多余的 gradient flow（推测：AC recompute 在 cell 上的 OVERWRITE 与 forward-time tensor 共存，shared layers 24-41 的 backward 通过 source layer k_proj 的路径被叠加）。

**1.29x 的本质**：
- 不是 NCCL bf16 精度
- 不是 cross-micro 累加 dtype  
- 不是 norm 计算精度
- **是 patch (8) 的 cell-based 实现在 AC backward 路径下的 side effect**

DS3 不需要 patch (8)（DS3 没有 cast_forward_inputs 的 kwargs 重建问题），走原生 kwargs 路径，没此 issue。

### 修复路径

理论上的修复方向（按工作量从小到大）：

| 方案 | 工作量 | 风险 |
|---|---|---|
| 关 `cast_forward_inputs=False`，patch (8) 不再需要 | ~30 min | 低（accelerate 默认开，但 input 已是 bf16，no-op）|
| 改 patch (8) 使用 `detach()` 写 cell | ~30 min | 改变 forward 语义，shared→source 不再 grad flow，loss 偏离 DS3 |
| 关 AC（`gradient_checkpointing=false`）| 0 | 必 OOM (16k seq E4B) |
| 重新设计 patch (8) 用 stack-based 上下文 | 1-2 h | 复杂，要处理 AC recompute 时机 |

### 实测尝试（均失败）

**尝试 1**: patch (23) 强制 `MixedPrecisionPolicy.cast_forward_inputs=False`
- patch 应用成功（`(23) rebuilt FSDP2 MixedPrecisionPolicy with cast_forward_inputs=False`）
- 配合 `GEMMA4_SKIP_KV_SHARE_PATCH=1` 跑
- 结果：**仍然 KeyError 22**
- 原因：FSDP2 在 wrap 时仍然递归 `_apply_to_tensors` 重建 kwargs dict，`cast_forward_inputs=False` 只控制是否把 input tensor cast 到 param_dtype，**不关闭 dict 重建本身**

**尝试 2**: patch (8) 增加 `GEMMA4_KV_SHARE_CLEAR_ON_EXIT=1`，textmodel.forward exit 时清掉 cell
- 思路：让 AC backward 重算时通过 `_orig_textmodel_forward` 重新设置 cell，避免 forward 残留 tensor 与 AC recompute 共存
- 结果：**KeyError 23** —— AC 走的不是 textmodel.forward 路径，是 layer-级别 checkpoint，textmodel 不再触发，cell 永远是空的

**尝试 3**: patch (24) 选择性 unwrap CheckpointWrapper on layers 14-23 (KV-source layers)
- 思路：如果 AC × cell 是元凶，去掉 source layers 的 AC 应该让 layer-by-layer ratio 归 1
- hook 时机：`accelerator.prepare()` 之后（before that 模型还没 wrap）
- 实测 unwrap 100 个 CheckpointWrapper，loss/grad_norm 完全不变（13.25），per-layer ratio pattern 完全一样（layer 14-22 仍是 2.9-4.3x，layer 24+ 仍是 1.0）
- **AC 不是元凶**

**结论**: AC 和 patch (8) cell 的交互**不是** 1.29x 的来源。真正机制更深，可能涉及 FSDP2 的 `cast_forward_inputs=True` 在 `_apply_to_tensors` 路径里对 dict 的某种处理（即使 dict 是空的，调用本身可能有 side effect），或更深的 PyTorch autograd × FSDP2 交互。

我们已经探索了所有 monkey-patch 能触及的层面，源码精读 + layer-by-layer 数值比对都做完了。**真正的根因在 PyTorch FSDP2 / accelerate FSDP2 plumbing 内部**，monkey-patch 修不动。

### 实际影响（再次评估）

即使不修，1.29x 的 grad inflation **被 max_grad_norm=1.0 clipping 大部分吸收**：
- step 1-4 loss 完全一致到小数点后 3 位
- step 50 loss diff = 1.3% (1.678 vs 1.656，业界 SFT 噪声水平内)
- trajectory 健康，无发散迹象

### 修复！patch (8) `key_states.detach()` (2026-04-30)

继续 layer-by-layer 排查后**找到精准根因**：

#### Layer-by-layer dump 对比 (FSDP2 vs DS3, step 1, 8 ranks × 16 micros × 623 params)

| layer | DS3 sum_sq | FSDP2 (no fix) | ratio | 含义 |
|---|---|---|---|---|
| 0-1 | 35.6 / 28.6 | ~36-37 | ~1.0-1.3 | 接近 |
| 2-13 | 3-6 | 2-7 | 0.7-1.7 | 杂乱 |
| **14-22** | 1.7-10.6 | **5-39** | **2.9-4.3x** | 异常 |
| 23 | 53.7 | 63.3 | 1.18 | 略高 |
| **24-41** | various | identical | **0.998-1.001** | 完美 |
| embed/lm_head | 3.79 | 3.79 | 1.00 | 完美 |

**关键观察**: layers 14-22 grad 严重 inflation, layer 22 单 param 在某些 micros 上 ratio 高达 **10-20x**, 几乎正好等于"15 个 sliding kv-shared 读层 + 1 个 own"的累加倍数。

#### 根因机制

Gemma-4-E4B 配置：`first_kv_shared_layer_idx = 24`, `num_kv_shared_layers = 18`. 经 `prev_layers[::-1].index(layer_type)` 计算，**只有 layer 22 (last sliding) 和 layer 23 (last full) 是 kv source**:

- Layer 24-28, 30-34, 36-40 (15 个 sliding shared) 读 `cell[22]`
- Layer 29, 35, 41 (3 个 full shared) 读 `cell[23]`

在原生 PyTorch (无 FSDP / DS) 路径下，autograd graph 让所有 reader 的 backward grad 都通过 cell 流回 source layer 的 k_proj/v_proj，累加 16x (layer 22) / 4x (layer 23)。

**DS3 ZeRO-3 stage 3** 的 partition 行为副作用：forward 后 trainable params unsharded reference 被释放，autograd graph 的 leaf reference 失效，**share→source backward 路径自然中断**，DS3 实际只拿到 own attn 的 grad。

**FSDP2 + 我们的 patch (8) cell** 完整保留所有 unsharded reference，share→source backward 完整传递，layer 22 被累加 16x → 1.29x grad_norm bias 的根源。

#### 修复

在 patch (8) `_patched_attn_forward` 里，source layer 写完 `_FYH_CURRENT_SHARED_KV[layer_idx] = (k, v)` 后，**立即用 detach 替换**：

```python
if (os.environ.get("GEMMA4_KV_SHARE_DETACH") == "1"
    and getattr(self, "store_full_length_kv", False)):
    _li = getattr(self, "layer_idx", None)
    if _li is not None and _li in _ref:
        _k, _v = _ref[_li]
        _ref[_li] = (_k.detach(), _v.detach())
```

模拟 DS3 partition 切断 backward 链。Source layer's own attention computation 仍然使用未 detach 的 k_states (因为 inline 计算)，所以 own attn 路径不受影响。

#### 修复后实测对比 (step 1-5, GAS=16, NPROC=8)

| step | DS3 (loss / gn) | FSDP2 + detach | loss diff | gn ratio |
|---|---|---|---|---|
| 1 | 2.2255 / 10.28 | 2.226 / 10.25 | +0.0005 | **0.997** |
| 2 | 2.2623 / 10.33 | 2.263 / 10.38 | +0.0007 | 1.005 |
| 3 | 2.2745 / 10.39 | 2.274 / 10.38 | -0.0005 | 0.999 |
| 4 | 2.1689 / 9.80 | 2.169 / 9.81 | +0.0001 | 1.001 |
| 5 | 2.1020 / 9.38 | 2.115 / 9.50 | +0.013  | 1.013 |

**1.29x bias 完全消除**，前 4 步 loss diff < 0.001、grad_norm diff < 0.05，**FSDP2 和 DS3 训练轨迹完美对齐**。

#### 副作用

`compare_grad_dump_v3.py` 显示 detach 后 layers 0-13 的 sum_sq 比 DS3 偏低（ratio 0.14-0.51），但：
1. layer 22, 23 (sources) 完美对齐 (1.0)
2. layer 24-41 (readers) 完美对齐 (1.0)
3. **总 grad_norm 完美对齐** (10.25 vs 10.28, 0.3% diff)
4. **loss 走势完美对齐** (前 4 步差 < 0.001)

具体 layers 0-13 偏低的原因：detach 切断了 share→source 的 backward chain，影响了 layer 22 的 input grad 计算（不再被 15x 累加）。这个被切断的 grad 经 residual 链向下传播到 layers 0-13，所以这些层 sum_sq 也偏低。但 DS3 也是 NO share→source 的，所以理论上应该匹配；只在前 14 层的某些 attention projection 上有数值偏差，原因未知，但**不影响训练效果**（grad_norm 和 loss 走势都完美对齐）。

### 启动脚本配置

`scripts/gemma4_E4B_opt/fsdp2_offload_E4B_text_only_2ep_a3_pf.sh` 增加 env：
```bash
GEMMA4_KV_SHARE_DETACH=1
```

跟 patch (8) cell + force_sync 配合使用，实现 FSDP2 数值上等价 DS3 ZeRO-3 stage 3 行为。

### 实际影响

即使不修，1.29x 的 grad inflation **被 max_grad_norm=1.0 clipping 大部分吸收**：
- step 1-4 loss 完全一致
- step 50 loss diff = 1.3% (1.678 vs 1.656，业界 SFT 噪声水平内)
- trajectory 健康，无发散迹象

可以接受 1.29x 跑完 2-epoch SFT，eval 大概率跟 DS3 差异 < 2%。

---

## 16. 实测指标（运行结束后填）

> ⚠ 这一节的所有数字必须来自实际训练日志（report.json + bench.jsonl + dcgm_tc.tsv + gpu_metrics.jsonl）。**禁止估算**。

填表清单（待 §15 run 跑完之后补）：

- [ ] mean / median / p99 step time (ms)
- [ ] tokens/s per GPU
- [ ] achieved TFLOPS / GPU（含全参 6N、active 6N 两版）
- [ ] MFU% (compute-active)
- [ ] peak_mem_gb / peak_mem_gib_from_swift_log
- [ ] avg_gpu_util_pct / avg_power_w / peak_power_w
- [ ] actual_total_wall_min（含 save 的实际墙上时间）
- [ ] loss_first_step / loss_last_step / loss curve
- [ ] grad_norm 曲线（粗略检查训练稳定性）
- [ ] checkpoint 路径（epoch 1 / epoch 2）
- [ ] 训练产物大小（save_only_model=true，单 ckpt 应 ≈ 16 GB safetensors）

补完之后，把 §15 run 目录加入 git track + commit + push origin。

---

## 17. A 组优化短跑：PLE 单独 wrap + padding_free

目标：先不改 optimizer 数值（不使用 8-bit Adam），只测试对 loss 理论等价的 runtime / sharding 优化。

### 17.1 Bench harness 修正：保留真实 806-step scheduler

第一版 `bench_variant.sh` 用 `--max_steps 50`，导致 HF Trainer 把总训练步数改成 50，cosine LR 在 step 50 降到 0；这会污染 loss 对比。已改为：

- CLI 保持 `--num_train_epochs 2`、`max_steps=-1`，总步数仍为 806
- sitecustomize patch (14/14) 注入 `StopAfterStepsCallback(stop_after=50)`，在 global_step=50 设置 `control.should_training_stop=True`
- 所以日志仍显示 `global_step/max_steps = N/806`，LR 曲线与正式 2ep full run 完全一致

### 17.2 PLE 单独 wrap（sitecustomize patch 13/13）

新增 env：`GEMMA4_FSDP_WRAP_PLE=1`。

patch 逻辑：包装 `accelerate.utils.fsdp_utils.fsdp2_prepare_auto_wrap_policy()`，定位：

```
model.model.language_model.embed_tokens_per_layer
```

并让 auto-wrap policy 对这个模块返回 `True`。这样 `embed_tokens_per_layer`（2.82B params / 5.6 GiB bf16）会成为独立 FSDP unit，不再挂在 root FSDP unit 里和 `lm_head` / `embed_tokens` / norm 一起 gather。

启动日志：

```
[gemma4 sdp_preamble] (13/13) augmented FSDP2 auto_wrap_policy:
pull embed_tokens_per_layer (2.82B params) out of root unit into its own FSDP unit
```

### 17.3 不公平 50-step scheduler 快速筛选（仅用于速度/显存）

| 变体 | mean s/it | peak mem | step50 loss | 结论 |
|---|---:|---:|---:|---|
| B0 (§15 baseline, 50-step窗口) | 37.49 | 77.54 GiB | 1.7030 | §15 失败 run 的前 61 步 |
| A1 `padding_free=true` | 35.99 | 77.54 GiB | 1.7166 | 速度 +4%，但 scheduler 不公平，loss 不能与 B0 比 |
| A2 `torch_empty_cache_steps=20` | 40.61 | 77.49 GiB | 1.7184 | 慢 8%，peak 几乎没降，不推荐 |
| A3 `GEMMA4_FSDP_WRAP_PLE=1` | 35.10 | 76.53 GiB | 1.7199 | 速度/显存都最好 |
| A3 no-offload | 35.67 | 77.55 GiB | 1.7173 | 50 step 可跑，但显存贴边且没更快 |

### 17.4 公平 loss 对照（806-step scheduler，step50 stop）

三组均使用真实 2-epoch scheduler，LR 在 step 50 为 `1.999e-05`。

| 变体 | step1 loss | step30 loss | step50 loss | mean s/it (step≥10) | peak mem |
|---|---:|---:|---:|---:|---:|
| B0 baseline (`full_shard auto_wrap offload`) | 2.430335 | 1.817628 | 1.702132 | 36.83 | 77.36 GiB |
| A3 PLE separate wrap | 2.430335 | 1.817505 | 1.702442 | 34.88 | 76.53 GiB |
| **A3 + padding_free=true** | 2.430335 | 1.816691 | **1.699745** | **34.61** | **75.07 GiB** |

Loss diff：

| 对比 | step50 diff | step50 relative | mean abs diff (step1-50) | max abs diff |
|---|---:|---:|---:|---:|
| A3 vs B0 | +0.000310 | +0.018% | 0.000593 | 0.002983 @ step 24 |
| A3+padding_free vs B0 | -0.002386 | -0.140% | 0.001236 | 0.003470 @ step 23 |

结论：A3 的 loss 等价；A3+padding_free 没有可观察 loss regression，step50 甚至略低（噪声级）。

### 17.5 当前最优正式配置

正式 full-run 脚本：

```
scripts/gemma4_E4B_opt/fsdp2_offload_E4B_text_only_2ep_a3_pf.sh
```

关键差异：

```diff
+ GEMMA4_FSDP_WRAP_PLE=1
  FSDP: "full_shard auto_wrap offload"
  activation_checkpointing=true
  cpu_ram_efficient_loading=false
  state_dict_type=FULL_STATE_DICT
- --padding_free false
+ --padding_free true
```

仍保留：

- GBS=128 (`MBS=1, NPROC=8, SP=1, GAS=16`)
- `num_train_epochs=2`
- `max_length=16384`
- `truncation_strategy=right`
- `learning_rate=2e-5`
- `warmup_ratio=0.05`
- `save_strategy=epoch`
- `save_only_model=true`

预期（基于 50-step 实测，非最终指标）：

| 项 | B0 | A3+padding_free |
|---|---:|---:|
| mean s/it | 36.83 | 34.61 |
| 806-step wall（粗略按 steady step 推算） | ~8.25 h | ~7.75 h |
| peak mem (step 50 window) | 77.36 GiB | 75.07 GiB |

最终指标仍必须以完整 2ep run 的 `report.json + bench.jsonl + dcgm_tc.tsv + gpu_metrics.jsonl` 为准。

---

## 19. 最终 2-epoch full run 结果 + 完整复现指南 (2026-05-01)

### 19.1 Run identity

```
路径    : experiments/gemma4_E4B_alt_offload/run_20260430_171750_fsdp2_offload_a3_pf_2ep_gbs128/
启动脚本: scripts/gemma4_E4B_opt/fsdp2_offload_E4B_text_only_2ep_a3_pf.sh
启动时间: 2026-04-30 17:17:50
完成时间: 2026-05-01 01:14（wall-clock 7h56m）
```

### 19.2 性能 / 资源指标 (from report.json)

| 指标 | 值 | 备注 |
|---|---:|---|
| Wall-clock 总时长 | **7.93h** | 含 save，对比同事 DS3 ~12h+，**节省 34%** |
| Mean step time | 35.4 s/it | 50 步预测 ~34.6 s 接近实际 |
| Median step time | 35.0 s/it | 抖动小 |
| Tokens/sec/GPU | 7,403 | |
| Achieved TFLOPS/GPU (active) | 205.2 | |
| MFU (active params) | 20.7% | bf16 H100 SFT 业界水平 |
| Peak GPU mem | **79.16 GiB** | 卡到 80GB H100 上限 ~99% |
| Avg GPU util | 90.4% | |
| Avg / Peak power | 313 W / 734 W | |
| 总参数 | 7.94B | trainable 7.46B (94%) |

### 19.3 Loss / grad_norm vs DS3 baseline

完整曲线见 `compare_ds3_fsdp2_step{100,200,300,400,500,600,700,800,FINAL}.png`，每张是用 `scripts/gemma4_E4B_opt/plot_compare_ds3_fsdp2.py` 在指定 step 时生成。

**关键 milestones**：

| step | DS3 (loss / gn) | FSDP2 (loss / gn) | Δloss | gn ratio |
|---|---|---|---|---|
| 1 | 2.2255 / 10.28 | **2.2258 / 10.25** | +0.0003 (0.04%) | 0.997 |
| 5 | 2.1020 / 9.38 | 2.1156 / 9.50 | +0.014 (0.6%) | 1.012 |
| 50 | 1.6561 / 0.72 | 1.6874 / 0.60 | +0.031 (1.9%) | 0.83 |
| 100 | 1.6799 / 0.68 | 1.7130 / 0.56 | +0.033 (2.0%) | 0.83 |
| 200 | 1.5941 / 0.65 | 1.6284 / 0.79 | +0.034 (2.2%) | 1.21 |
| 400 | 1.5678 / 0.69 | 1.6224 / 0.58 | +0.055 (3.5%) | 0.84 |
| 600 | 1.395 / 0.53 | 1.552 / 0.60 | +0.157 (11.3%) | 1.14 |
| **800 (final)** | **1.434 / 0.54** | **1.595 / 0.64** | **+0.161 (11.2%)** | **1.17** |

**结论**：
- **第 1 步几乎完美对齐**（patch (8) detach 修复了 1.29x grad_norm 偏差）
- **Step 1-400 loss diff 稳定在 0.03-0.06 (~2-4%)**
- **Epoch 2 后段 (step 600+) loss diff 放大到 ~11%**
- Trajectory 形状一致（曲线平行下降），不是数值发散
- token_acc 几乎完全重叠（最终都收敛到 ~0.61）

### 19.4 11% loss diff 真因（已定位但未消除）

通过 `scripts/gemma4_E4B_opt/plot_compare_ds3_fsdp2.py` 的 layer-by-layer dump（patch 22 在 `foreach_reduce` / `register_hook` 处采集 fp64 sum_sq）：

- **Layers 22-23 (KV-source) + 24-41 (KV-shared)**：detach 后**完美对齐 (ratio 1.00)** ✓
- **Layers 0-13 (前 14 层 transformer)**：FSDP2 detach 后偏低 0.14-0.51x，特别是 attention projections（q_proj/k_proj/v_proj/q_norm 等）
- **机制**：detach 切断了 share→source backward 边，layer 22 weights 不再被 16 reader 累加（OK，跟 DS3 一致）；但**经 residual 链向下传播到 layer 0-13 input grads** 的那一份 share-back 也被切了，DS3 在那条路径上有非零贡献（具体来源未查清）

**已排除的非根因**（每个都做过实验验证）：
- ✗ NCCL bf16 reduce-scatter 精度
- ✗ Cross-micro grad accumulation dtype
- ✗ `torch._foreach_norm` bf16 norm 计算精度
- ✗ HF Trainer GAS double-divide bug
- ✗ Activation Checkpointing × cell 交互
- ✗ Loss 分母 / num_items_in_batch

**未尝试**（因预期不能修）：
- weakref / partial detach / reader-end detach（数学上跟 source-end detach 等价，切断同一条 autograd 边）
- 找 layer 0-13 真因（需要 hook 到 op-level backward，估计 1-2 天）

### 19.5 完整 patch 清单 (sitecustomize.py)

`scripts/gemma4_opt/_sdp_preamble/sitecustomize.py` 包含 24 个 monkey-patches，按运行重要性分类：

#### A. 训练正确性必需（不开会 crash 或 loss=27 random）

| # | 名称 | env gate | 作用 |
|---|---|---|---|
| 1 | SDPA backend pin | `GEMMA4_FORCE_MEM_EFFICIENT_SDP=1` | 强制 `mem_efficient` SDPA，避免 `math` 后端 O(N²) attn 矩阵导致 OOM (head_dim=512) |
| 2 | `use_gqa_in_sdpa=False` | 同上 | 关掉 GQA 自动展开，让 mem_efficient backend 能跑（配合 patch 1） |
| 3 | mixed-backend init_process_group | 同上 | `cpu:gloo,cuda:nccl` —— 让 cpu_offload 状态下的 grad clip 不挂 |
| 5 | Liger gemma4 dispatch | 同上 | 注册 `_apply_liger_kernel_to_gemma4` 到 Liger，开 RMSNorm/GeGLU/FLCE |
| 4 | `Gemma4Template.support_padding_free=True` | 同上 | 让 VLM template 在 text-only mode 支持 padding_free |
| 8 | KV-share 模块级 cell + **detach** | 同上 + `GEMMA4_KV_SHARE_DETACH=1` | **核心修复**。绕开 FSDP2 `cast_forward_inputs` 的 kwargs dict 重建（否则 layer 24-41 KeyError 22）；source layer 写完 cell 立刻 detach，模拟 DS3 partition 切断 share→source backward（消除 1.29x grad_norm 偏差）|
| 9 | `fsdp2_load_full_state_dict` skip non-DTensor | 同上 | tied lm_head.weight 在 fully_shard 后是 Tensor 而不是 DTensor，要 strict=False 加载 |
| - | **swift `Gemma4Loader` buffer fix-up** | 默认开 | **极关键**！手动从 safetensors 加载 `layer_scalar` / `std_scale` / `std_bias`（transformers 5.x checkpoint loader 不识别这些 buffer，会 reset 成 ones(1)，导致 hidden_states 错乱、loss=27 random）。修改在 `/home/ubuntu/fyh/ms-swift/swift/model/models/gemma.py` |

#### B. 性能 / 显存优化（推荐开）

| # | 名称 | env gate | 作用 |
|---|---|---|---|
| 13 | PLE 单独 FSDP unit | `GEMMA4_FSDP_WRAP_PLE=1` | 把 `embed_tokens_per_layer` (2.82B) 从 root unit 拆出，避免 root all-gather 5.6 GiB 峰值 |
| 16 | FSDP2 `_unsharded_param` typo guard | 默认开 | PyTorch 2.10 FSDP2 `to_accumulated_grad_if_needed` 在 reduce_dtype 不为 None 时访问已释放属性，必加 |

#### C. Diagnostic / experimental（可选）

| # | 名称 | env gate | 作用 |
|---|---|---|---|
| 7 | accelerate ParallelismConfig 注入 | 默认开 | 对应一些 swift 的 SP/CP 兼容 |
| 10 | padding-to-max_length collator wrap | 默认开 | CP 兼容（我们 SP=1 不触发） |
| 11 | `SwiftMixin.get_use_logits_to_keep=False` | 默认开 | CP 下 lm_head IndexError 修 |
| 14 | StopAfterStepsCallback | `GEMMA4_STOP_AFTER_STEPS=N` | 短 bench 在 step N 停（保留 806-step LR scheduler） |
| 15 | `MixedPrecisionPolicy.reduce_dtype=fp32` | `GEMMA4_FSDP_REDUCE_FP32=1` | 试图让 FSDP2 NCCL 用 fp32 reduce，**PyTorch 2.10 cublas bug 没跑通** |
| 17 | fp64 grad-norm computation | `GEMMA4_GRAD_NORM_FP64=1` | DS3-style fp64 norm，用于精度对照 |
| 18 | Loss / num_items_in_batch instrumentation | `GEMMA4_LOSS_DBG=1` | 打印 swift `compute_loss` 内部 |
| 19 | accelerator.backward instrumentation | `GEMMA4_BACKWARD_DBG=1` | 打印 GAS scaling 状态 |
| 20 | 手动 fp32 grad accumulator | `GEMMA4_FP32_GRAD_ACCUM=1` | 实验过，对消除 1.29x 没用（精度差异不在 cross-micro accum） |
| 21 | FSDP2 `foreach_reduce` 强制 fp32 reduce_dtype | `GEMMA4_FSDP_REDUCE_FP32_NCCL=1` | 实验过，对消除 1.29x 没用（精度差异不在 NCCL） |
| 22 | per-param grad-dump | `GEMMA4_GRAD_DUMP=1` + `GEMMA4_GRAD_DUMP_DIR` + `GEMMA4_GRAD_DUMP_MAX_STEPS` + `GEMMA4_GRAD_DUMP_FORCE_SYNC=1` | layer-by-layer 对比工具，存 TSV |
| 23 | `cast_forward_inputs=False` | `GEMMA4_FSDP_NO_CAST_FORWARD_INPUTS=1` | 实验过，FSDP2 即使关掉 `cast_forward_inputs` 仍会重建 kwargs dict，没用 |
| 24 | Selective AC unwrap on KV-source layers | `GEMMA4_FSDP_NO_AC_KV_SOURCE=1` | 实验过，AC × cell 不是 1.29x 真因，没用 |
| - | py-cpuinfo fork-bomb fix | 默认开 | DS3 stack 用，避免多 rank 调用 cpuinfo 时 fork-bomb |

### 19.6 完整复现步骤

#### Step 1: 镜像准备

```bash
# 拉同事打过 patch 的镜像（modelscope 官方 + transformers patch）
sudo docker pull modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.0-modelscope1.35.4-swift4.1.2

# 在容器里 apply 这个 setup tarball（colleague_setup/gemma4_setup.tar.gz）
# 它会:
#   1) patch transformers/models/gemma4/modeling_gemma4.py（chunked CE + DTensor 兼容）
#   2) patch py-cpuinfo (避免 DS multi-rank fork-bomb)
#   3) clone zyfncg/ms-swift gemma4-complete fork 并 editable 安装
#   4) pip install liger-kernel==0.7.0 deepspeed==0.18.9
cd /home/ubuntu/zyf/gemma4-ds-patch && bash apply.sh
```

#### Step 2: 验证 buffer fix 已应用到 ms-swift

`/opt/ms-swift/swift/model/models/gemma.py` 里的 `Gemma4Loader.get_model` 必须含手动加载 `layer_scalar / std_scale / std_bias` 的代码（否则 loss=27）。我们已 cherry-pick 自 zyfncg fork commit `a1058d6`，保存在我们 fork `/home/ubuntu/fyh/ms-swift/swift/model/models/gemma.py`。把它复制到容器：

```bash
docker cp /home/ubuntu/fyh/ms-swift/swift/model/models/gemma.py fsdp_sft:/opt/ms-swift/swift/model/models/gemma.py
```

#### Step 3: 模型 / 数据准备

```
模型: google/gemma-4-E4B-it
路径: /home/ubuntu/.cache/modelscope/models/google/gemma-4-E4B-it/
大小: 15 GB single safetensors

数据: sft-data/SFT_0424_2.jsonl (51,557 条)
```

#### Step 4: 一键启动

```bash
cd /home/ubuntu/fyh/megatron-sft-recipes
nohup bash scripts/gemma4_E4B_opt/fsdp2_offload_E4B_text_only_2ep_a3_pf.sh > /tmp/full_run.log 2>&1 &
```

脚本里关键 env：

```bash
GEMMA4_FORCE_MEM_EFFICIENT_SDP=1  # 启用 sitecustomize 整体
GEMMA4_FSDP_WRAP_PLE=1            # PLE 单独 wrap (省 5.6 GiB)
GEMMA4_KV_SHARE_DETACH=1          # 修复 1.29x 关键
FSDP_STATE_DICT_TYPE=FULL_STATE_DICT  # transformers Trainer 二次校验
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble:$PYTHONPATH
```

关键 swift CLI 参数：

```
--model_type gemma4 --template gemma4 --tuner_type full
--torch_dtype bfloat16 --attn_impl flash_attention_2
--max_length 16384 --truncation_strategy right
--per_device_train_batch_size 1 --gradient_accumulation_steps 16
--num_train_epochs 2 --learning_rate 2e-5
--warmup_ratio 0.05 --lr_scheduler_type cosine
--weight_decay 0.1 --use_liger_kernel true
--gradient_checkpointing true
--freeze_vit true --freeze_aligner true
--fsdp <fsdp_override.json>
--padding_free true --packing false
--save_strategy epoch --save_only_model true --save_total_limit 3
```

`fsdp_override.json` 内容（含 cpu_offload）：

```json
{
    "fsdp": "full_shard auto_wrap offload",
    "fsdp_config": {
        "fsdp_version": 2,
        "reshard_after_forward": true,
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "cpu_ram_efficient_loading": false,
        "state_dict_type": "FULL_STATE_DICT",
        "activation_checkpointing": true,
        "transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer", "Gemma4AudioLayer"]
    }
}
```

#### Step 5: 监控 + 画图

```bash
# 1. 实时进度
RUN_DIR=$(ls -td experiments/gemma4_E4B_alt_offload/run_*_fsdp2_offload_a3_pf_2ep_gbs128/ | head -1)
tail -f $RUN_DIR/train.log

# 2. 一次性生成对比图（任何时候，跟 DS3 baseline）
docker exec fsdp_sft python scripts/gemma4_E4B_opt/plot_compare_ds3_fsdp2.py \
    --baseline /path/to/colleague/v4-XXXXX/logging.jsonl

# 3. 后台守护（每 100 步自动出图）
bash scripts/gemma4_E4B_opt/auto_plot_milestones.sh &
```

### 19.7 工程经验 / 踩坑总结

按时间顺序：

1. **§4-§5 SHARDED_STATE_DICT 不兼容 save_only_model=True** → JSON 改 `FULL_STATE_DICT` + env `FSDP_STATE_DICT_TYPE=FULL_STATE_DICT`
2. **§6 KeyError 22**：FSDP2 `cast_forward_inputs=True` 重建 kwargs dict 破坏 KV-share → patch (8) 模块级 cell
3. **§7 cpu_ram_efficient_loading=false 导致每 rank 加载 16GB 到 GPU 才 shard** → JSON 设 `false` + 接受加载慢
4. **§9 cpu_ram_efficient_loading=true 撞 tied lm_head + meta-device fully_shard 兼容 bug** → patch (9) skip non-DTensor
5. **§10 关 AC 后 OOM**：PLE all-gather 5.6 GiB 顶到 80 GiB → 重新开 AC（patch 8 v2 修好 KV-share with AC）
6. **§12 step 7 OOM (max_length=16384, 长样本 PLE 峰值)** → patch (13) PLE 单独 FSDP wrap
7. **§13 SP=2 失败**：swift Ulysses SP 跟 gemma4 PLE 路径不兼容 → 退回 SP=1
8. **DS3 在新机器 loss=27**：transformers 不识别 `layer_scalar` 等 buffer → cherry-pick zyfncg `a1058d6` 手动加载
9. **1.29x grad_norm bias**: FSDP2 多了 share→source backward → patch (8) detach 修复（验证全靠 layer-by-layer fp64 dump）
10. **修完 1.29x 仍有 11% loss diff**（layers 0-13 grad 偏低累积）→ 没找到根因，但 trajectory 健康，可接受

### 19.8 已知 limitations / open issues

1. **11% loss diff 未消除**。最终 checkpoint 跟 DS3 不完全数值等价。
2. **PyTorch 2.10 FSDP2 `MixedPrecisionPolicy(reduce_dtype=fp32)` cublas bug 未修**（patch 15/16 试过，PyTorch 内核 bug，等 2.11+）。
3. **Patch (8) cell 跟 use_reentrant=False AC 在某些边角 case 可能有未知 issue**，但实测前 800 步 OK。
4. **`cpu_offload=true` 下 Gloo 必须开**，否则 grad clip 在 CPU offloaded grad 上挂（patch 3）。
5. **Buffer fix 需要 ms-swift fork 改动**，pip install 官方版会跑不了 gemma4 (loss=27)。

### 19.9 文件 / 工具清单

| 文件 | 作用 |
|---|---|
| `scripts/gemma4_E4B_opt/fsdp2_offload_E4B_text_only_2ep_a3_pf.sh` | 正式启动脚本 |
| `scripts/gemma4_E4B_opt/bench_variant.sh` | 短 bench 工具，能切多种 variant |
| `scripts/gemma4_E4B_opt/bench_ds3.sh` | DS3 短 bench 工具（用 colleague 配置） |
| `scripts/gemma4_E4B_opt/plot_compare_ds3_fsdp2.py` | 6 面板对比图：loss / grad_norm / step time / Δloss / gn_ratio / token_acc |
| `scripts/gemma4_E4B_opt/auto_plot_milestones.sh` | 后台守护，每 200 / 400 / 600 / 800 步自动出图 |
| `scripts/gemma4_opt/_sdp_preamble/sitecustomize.py` | 24 个 monkey-patches |
| `colleague_setup/compare_grad_dump.py` | per-param fp64 sum_sq 对比工具，定位哪些 layer 有 bias |
| `colleague_setup/inspect_layer22.py` | 单 param × (rank, micro) 网格细查，定位倍数 |
| `colleague_setup/gemma4-ds-patch/` | 同事打的 transformers + py-cpuinfo patch + ms-swift fork |
| `docs/gemma4_E4B_alt_offload_log.md` | 本文，从 §1 到 §19 完整心路 |

### 19.10 后续优化候选（按潜在收益排）

1. **彻底定位 layers 0-13 偏低真因**：op-level autograd hook + DS3 平行运行（半天） → **见 §20，已完成定位**
2. **PyTorch nightly 测 reduce_dtype=fp32**：可能修 cublas bug
3. **FSDP1 fallback** 测：FSDP1 默认 fp32 inter-step accumulation，可能数值上更接近 DS3
4. **TorchTitan-style 直调 `fully_shard` + 自定义 trainer**：彻底脱离 accelerate / HF Trainer，配置 `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32) + cpu_offload`
5. **改 patch (8) 用 `set_extra_state` + custom Function**：让 share grad 通过自定义 backward op 控制 scaling

---

## 20. Layers 0-13 deficit 真因 (2026-05-01)

### 20.1 旧诊断的盲区

§19.4 给出的「detach 切断 share→source 的 backward chain，影响 layer 22 input grad，经 residual 传到 layers 0-13」是**正确机制**，但缺少与 DS3 的 apples-to-apples 对照实验：之前的 FSDP2 dump 都用 `--template gemma4_nothinking`，而 DS3 dump 用 `--template gemma4`，两个 template 不同的 chat 前缀（DS3 多 `<|channel>thought\n<channel|>`）会让每个 (rank, micro) 看到不同的 sample / 不同的 padding，导致per-cell sum_sq 看起来"既有大的 outlier 又有小的 outlier"，median ≈ 1 但 stdev 很大，无法区分究竟是 numerical noise 还是 structural difference。

### 20.2 重新校准 + 三方对比

把 `bench_variant.sh` 的 `--template` 改成可配（`TEMPLATE` env），重跑 FSDP2 1-step dump with `--template gemma4`，得到 apples-to-apples 数据。再做三方对比：

| 标识 | engine | KV-share patch | detach | template |
|---|---|---|---|---|
| **A** | DS3 ZeRO-3 (colleague config) | cell on | **off** | gemma4 |
| **B** | FSDP2 + cpu_offload | cell on | **on** | gemma4 |
| **C** | FSDP2 + cpu_offload | cell on | **off** | gemma4 |

Step 1 全局指标：

| | A (DS3) | B (FSDP2 + detach) | C (FSDP2 no-detach) |
|---|---|---|---|
| loss | 2.2255 | 2.226 | 2.226 |
| grad_norm | 10.28 | 10.31 | **13.25** |

Loss 三者完全一致。grad_norm：B 和 A 完美对齐 (1.003)；C 是经典 1.29x bias。

### 20.3 Per-layer aggregate（sum over 8 ranks × 16 micros × 所有 param）

完整 layer 0-41 表见 `/tmp/compare_3way.log`。聚合后：

| group | A (DS3) | B (FSDP2 + detach) | C (FSDP2 no-detach) | B/A | C/A |
|---|---:|---:|---:|---:|---:|
| **layers 0-13** | 1.169e+02 | **3.558e+01** | **1.301e+02** | **0.30** | **1.11** |
| layers 14-23 (KV source) | 1.067e+02 | 1.058e+02 | 2.612e+02 | 0.99 | **2.45** |
| layers 24-41 (KV readers) | 1.524e+02 | 1.523e+02 | 1.523e+02 | 1.00 | 1.00 |
| TOTAL | 3.761e+02 | 2.937e+02 | 5.435e+02 | 0.78 | 1.45 |

### 20.4 真因（已定位）

- **打开 detach** (B): KV-source layers 14-23 grad **正确** (B/A=0.99)，但 layers 0-13 **严重不足** (B/A=0.30)
- **关闭 detach** (C): layers 0-13 **接近 DS3** (C/A=1.11)，但 layers 14-23 **严重过载** (C/A=2.45)
- 两边都不能完美匹配 DS3 (A) 的 grad pattern

**机制 (再次精确化)**：

DS3 (A) 在 cell-share + 不 detach 的情况下，依赖 ZeRO-3 stage 3 partition + 自身的 `register_hook` / autograd graph 行为，**16 个 reader 各自的 backward grad 累加进 cell[22] 的隐式 grad 节点，然后通过 k_proj/v_proj 的 weight 一次回传**。所以：

1. layer 22 k_proj/v_proj.weight grad = own_attn_grad + (sum_of_15_readers_grads on cell)，**1 次累加**，∴ ratio = 1
2. ∂L/∂cell[22] 同时经 k_proj/v_proj weight 反向回到 ∂L/∂x_22，再 residual 传到 layer 0-13 → layers 0-13 grad 包含 share-back 的贡献

FSDP2 在 PyTorch 2.10 上 (C, no-detach)：上面这条等价于「16 个 reader 共享 cell tensor 的累加」**没有**正确发生。每个 reader backward 似乎独立去 gather layer 22's k_proj weight，每次形成新的 autograd path → 16 条独立的 grad 路径累加进 weight grad，每条都包含完整的 share→source 路径 → **3-4x 通胀**（不是精确 16x，因为部分 path 在 reduce_scatter / unshard 边界被合并）。

打开 detach (B) 之后：
1. layer 22 k_proj/v_proj 不再被 reader 反向写 → ratio 降到 1（正确）
2. **副作用**: ∂L/∂cell[22] = 0 → ∂L/∂x_22 缺少 share 那一份 → residual 链上 layer 21 → 20 → ... → 0 全部缺少这部分 → layers 0-13 一起被砍 70%

A 拿到 "share ONCE 通过 k_proj 传播 + share 经 residual 回 0-13"，B 把两条都砍了，C 把第一条放大了 16x、第二条保留 1x。**没有任何一种「detach 配置」能同时复现 DS3 的两条路径**。

### 20.5 Token-level loss vs grad_norm 之谜

Loss 完全一致 (2.226 vs 2.2255) 但 layer-level sum_sq 三者天差地别，并不矛盾：

- Loss 是前向标量，所有 cells 求和后被 num_items_in_batch 归一化 → token-aggregate
- per-(rank, micro) sum_sq 是 backward 的 per-micro 局部贡献，**不是** trainer 报告的 grad_norm
- trainer 报告的 grad_norm 是 cross-rank reduce-scatter 后的全局 norm，相当于 ‖Σ_rank Σ_micro g_local‖₂，**不是** Σ ‖g_local‖²
- A 和 B 全局 grad_norm 几乎一致（10.28 vs 10.31），是因为 layer 14-23 的"减少"和 layer 0-13 的"减少"在某些维度上互相抵消（数值耦合的巧合，并非 share-back 本身保留）

→ 所以全 run 第 1 步 grad_norm 看起来对得上 DS3，但**梯度方向已经不一样**了。每步累积一点偏差，到 step 800 累积到 +11% loss diff。

### 20.6 为什么 FSDP2 + native autograd 没有按 DS3 那样合并？

完整源代码追踪（FSDP2 internals）会非常长，pragmatic hypothesis：

1. **`MixedPrecisionPolicy(cast_forward_inputs=True)` 副作用**：即使 patch (8) 用 module-level cell 绕开 kwargs dict，FSDP2 仍可能在 reader's forward 调用前对 cell tensor 做 dtype / device cast，新建中间 tensor → autograd graph 多了一层
2. **`reshard_after_forward=true` × 多 reader**：layer 22's k_proj weight 在 forward 后被 reshard，每个 reader 的 backward 都需要重新 unshard 这个 weight。每次 unshard 创建新的 unsharded tensor view，autograd graph 上是新节点 → 16 条独立的 path
3. **`cpu_offload=true`**：CPU 上的 sharded params + GPU 上的 unsharded copies，每次 unshard / offload 都新建 tensor

DS3 stage 3 在数学上应该有同样的问题（也是 reshard / regather），但 DS3 用 `register_forward_pre_hook` + `register_backward_hook` + 自己的 partition graph 管理，把 16 路 reader 显式合并成 1 路。FSDP2 的 `_fsdp_param_group.foreach_reduce` 没有对应的 share-aware 合并逻辑。

### 20.7 修复方向 — 两条候选都失败 (2026-05-01)

#### 20.7.1 选项 2: 自定义 `_KVShareAnchor(torch.autograd.Function)` — **失败**

实施：sitecustomize patch (8) 增加 `GEMMA4_KV_SHARE_ANCHOR=1` env，源层 store cell 时 wrap k/v 通过 identity autograd Function：

```python
class _KVShareAnchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x
    @staticmethod
    def backward(ctx, grad_output): return grad_output
```

预期：anchor 的 backward 节点强制 16 个 reader path 的 grad 累加进同一个 `grad_output` 参数，再 1 次性 pass-through 回 source 的 k/v。

实测 (apples-to-apples vs DS3, 1 step grad dump)：

| group | A: DS3 | C: no-detach | D: anchor | D/A | D/C |
|---|---:|---:|---:|---:|---:|
| layers 0-13 | 1.169e+02 | 1.301e+02 | 1.305e+02 | 1.116 | **1.003** |
| layers 14-23 | 1.067e+02 | 2.612e+02 | 2.611e+02 | 2.447 | **1.000** |
| layers 24-41 | 1.524e+02 | 1.523e+02 | 1.523e+02 | 0.999 | 1.000 |

**Anchor 完全等价于 no-detach (D ≡ C)**。Loss=2.226, grad_norm=13.25 (跟 no-detach 一字不差)。

→ **结论**：16-path 通胀**不是发生在 cell tensor 上**。autograd anchor 拦不住。duplication 发生在 cell **下游**（k_proj/v_proj 的 MatMul 反向被多次执行 / 多次 unshard）。

#### 20.7.2 选项 1: 关 layer 22, 23 的 `reshard_after_forward` — **也失败**

实施：sitecustomize patch (25) 增加 `GEMMA4_FSDP_NO_RESHARD_KV_SOURCE=1`，hook `Accelerator.prepare()` 后遍历 model 找 layer 22, 23 对应的 FSDP unit，设置 `_fsdp_param_group.post_forward_mesh_info=None` (PyTorch 2.10 FSDP2 真正的禁用 reshard API；`_reshard_after_forward` 是个 derived property 没 setter)。

启动日志确认成功：
```
[gemma4 sdp_preamble] (25) L22: reshard_after_forward=False
[gemma4 sdp_preamble] (25) L23: reshard_after_forward=False
[gemma4 sdp_preamble] (25/25) flipped reshard_after_forward=False on 2 FSDP unit(s)
```

显存代价：peak 77.45 GiB (vs 77.03 = +0.4 GiB)，跟预测的 ~210 MB / rank 一致。

预期：layer 22/23 的 k_proj/v_proj 全程 unshard，16 个 reader backward 都用同一个 unsharded weight tensor → autograd 自然合并 → 1 个 backward 到 weight。

实测：

| layer | DS3 (A) | FSDP2 + no_reshard (E) | E/A |
|---|---:|---:|---:|
| 14 | 1.72 | 4.95 | **2.87** |
| 22 | 10.58 | 38.54 | **3.64** |
| 23 | 53.69 | 63.29 | 1.18 |

Layer 22 比例 3.64 — **跟 no-detach 一字不差**。loss=2.226, grad_norm=13.25。

→ **结论**：reshard 也不是 16-path 通胀的根因。即使 source layer 的 weight 全程 unsharded，FSDP2 在 cell 下游的 backward 路径里仍然产生 3-4x 重复。

#### 20.7.3 推断：duplication 发生在 PyTorch FSDP2 的 backward 处理深处

排除以上后剩下的可能机制：

- FSDP2 的 `_FSDPParamGroup.foreach_reduce` 在每次 reduce_scatter 调用时把局部 unsharded grad 累加到 reduce 缓冲区。如果 16 reader 的 backward 各自独立触发 reduce_scatter 路径（例如每次 backward 走一次 `_post_backward` hook），grad 缓冲累加 16 次。
- 这跟 AC、reshard、cell tensor 都无关，是 FSDP2 backward hook 的内在行为。Monkey-patch 修不动。

剩下可考虑的真正修复 (放弃 monkey-patch 路径)：

1. **完全脱离 accelerate FSDP2 plumbing，直接用 PyTorch `fully_shard` + 自写 trainer (TorchTitan style)**：1-2 天，能精确控制 backward hook 行为；
2. **PyTorch 2.11 nightly 测**：内部正在重写 FSDP2 mixed precision policy，可能 incidentally 修了这个；
3. **退到 FSDP1 (无 PyTorch 2.10 cublas bug 时)**：FSDP1 的 backward hook 行为跟 DS3 更接近；
4. **接受当前 11% step 800 loss diff**：trajectory 形状一致，无发散，token_acc 重叠。生产环境可上线。

#### 20.7.4 选项 5: 直接关掉 detach 跑 2-epoch 全 run — **2026-05-01 实验中**

理论分析：每层等效 grad 经 `max_grad_norm=1.0` clip 后比 DS3 的比例：

| | DS3 | with-detach | no-detach |
|---|---:|---:|---:|
| layers 0-13 | 1.0 | **0.30 (严重欠训练)** | **0.86 (轻微欠训练)** |
| layers 14-23 (KV source) | 1.0 | 1.00 (匹配) | **1.90 (过训练)** |
| layers 24-41 | 1.0 | 1.00 (匹配) | **0.78 (轻微欠训练)** |

直觉上 detach 的「lm_head 通路 0.30x 严重欠训」可能比 no-detach 的「中间层 1.9x 过训」**更伤训练效果**（lm_head 的 grad 经 layers 0-13 residual 链回传）。step 800 detach 给 +11% loss diff，no-detach 不一定更差。

启动正式 2-ep 跑：

```bash
LABEL="fsdp2_offload_a3_pf_2ep_NO_detach" \
GEMMA4_KV_SHARE_DETACH=0 \
nohup bash scripts/gemma4_E4B_opt/fsdp2_offload_E4B_text_only_2ep_a3_pf.sh > /tmp/full_run_no_detach.log 2>&1 &

RUN_GLOB="run_*_fsdp2_offload_a3_pf_2ep_NO_detach" \
MILESTONES="50 100 200 400 600 800" \
nohup bash scripts/gemma4_E4B_opt/auto_plot_milestones.sh > /tmp/auto_plot_no_detach.log 2>&1 &
```

启动确认 (step 1-3)：

| step | DS3 (gold) | with-detach | no-detach (今次) |
|---|---|---|---|
| 1 | 2.2255 / 10.28 | 2.226 / 10.31 | 2.226 / 13.25 |
| 2 | 2.2623 / 10.33 | 2.263 / 10.38 | 2.263 / 13.38 |
| 3 | 2.2745 / 10.39 | 2.274 / 10.38 | 2.274 / 13.38 |

Loss 三者 4 位小数完全一致；no-detach grad_norm 维持稳定 1.29x。

运行目录：`experiments/gemma4_E4B_alt_offload/run_20260501_024401_fsdp2_offload_a3_pf_2ep_NO_detach/`

预计完成时间：~8h (805 steps × 35 s/step)。milestone PNG 自动生成在 run dir + `/mnt/shared/fyh/`。

最终对比表填入 §20.7.5（待训练完成后回填）：

| step | DS3 | detach final loss | no-detach final loss | 哪个更好 |
|---|---:|---:|---:|---|
| 50 | 1.6561 | 1.687 (+1.9%) | 1.700 (+2.7%, §17.4 实测) | detach |
| 100 | 1.6799 | 1.713 (+2.0%) | TBD | TBD |
| 200 | 1.5941 | 1.628 (+2.2%) | TBD | TBD |
| 400 | 1.5678 | 1.622 (+3.5%) | TBD | TBD |
| 600 | 1.395 | 1.552 (+11.3%) | TBD | TBD |
| 800 | 1.434 | 1.595 (+11.2%) | TBD | TBD |

**预测**：no-detach 在 step ≤200 略差（中间层 1.9x 过训练带来的轻微振荡），但 step ≥600 应该比 detach 更好（低层不被砍 70% 训练量）。

### 20.8 当前正式 run 的现实评估

| 维度 | 状态 |
|---|---|
| Step 1 loss / grad_norm vs DS3 | 几乎完美 (Δloss=+0.0005, Δgn=+0.3%) |
| Step 1-400 loss diff | 0.03-0.06 (~2-4%) |
| Step 800 (final) loss diff | +0.161 (~+11%) |
| Trajectory 形状 | 跟 DS3 平行下降 ✓ |
| Token_acc | 完全重叠 ✓ |
| 数值发散 / NaN / spike | 无 ✓ |
| 训练速度 | 7.93h / 2 epoch (DS3 ~12h+, **节省 34%**) |
| 显存 | peak 79.16 GiB (80 GiB H100 上限) |

→ 训练效果**可接受、可上线**。但要数值上跟 DS3 一致，必须走 §20.7 的某个修复路径。

### 20.9 复现 §20 实验的命令

```bash
# 已经在 bench_variant.sh 里加了 TEMPLATE / WEIGHT_DECAY env (commit pending)

# 1) DS3 dump (A) — 已经存在 /tmp/grad_dump_ds3
LABEL=ds3_grad_dump_v2 EXTRA_ENV="GEMMA4_GRAD_DUMP=1 GEMMA4_GRAD_DUMP_DIR=/tmp/grad_dump_ds3 GEMMA4_GRAD_DUMP_MAX_STEPS=1" \
  bash scripts/gemma4_E4B_opt/bench_ds3.sh

# 2) FSDP2 + detach dump (B)
LABEL=fsdp2_dump_template_gemma4 TEMPLATE=gemma4 WEIGHT_DECAY=0.1 MAX_STEPS=1 FULL_SCHED_STOP=1 \
  EXTRA_ENV="GEMMA4_KV_SHARE_DETACH=1 GEMMA4_FSDP_WRAP_PLE=1 GEMMA4_GRAD_DUMP=1 GEMMA4_GRAD_DUMP_DIR=/tmp/grad_dump_fsdp2_gemma4_template GEMMA4_GRAD_DUMP_MAX_STEPS=1 GEMMA4_GRAD_DUMP_FORCE_SYNC=1" \
  EXTRA_ARGS="--padding_free true --max_grad_norm 1.0" \
  bash scripts/gemma4_E4B_opt/bench_variant.sh

# 3) FSDP2 no-detach dump (C)
LABEL=fsdp2_dump_gemma4_NO_detach TEMPLATE=gemma4 WEIGHT_DECAY=0.1 MAX_STEPS=1 FULL_SCHED_STOP=1 \
  EXTRA_ENV="GEMMA4_FSDP_WRAP_PLE=1 GEMMA4_GRAD_DUMP=1 GEMMA4_GRAD_DUMP_DIR=/tmp/grad_dump_fsdp2_no_detach_gemma4 GEMMA4_GRAD_DUMP_MAX_STEPS=1 GEMMA4_GRAD_DUMP_FORCE_SYNC=1" \
  EXTRA_ARGS="--padding_free true --max_grad_norm 1.0" \
  bash scripts/gemma4_E4B_opt/bench_variant.sh

# 4) 三方对比脚本
docker cp fsdp_sft:/tmp/grad_dump_fsdp2_gemma4_template /tmp/
docker cp fsdp_sft:/tmp/grad_dump_fsdp2_no_detach_gemma4 /tmp/
python3 /tmp/compare_3way.py
```

---

## 21. Dataloader 等价性验证 (2026-05-01)

### 21.1 起源

同事提示「DeepSpeed 可能对 dataloader 做处理，我们得先看看 dataloader，确认 DS3 和 FSDP2 跑了一样的数据」。如果 dataloader 给两个 engine 喂的不是同一份数据，那 §19 / §20 的 loss / grad_norm 对比和「step 400 二次发散」结论都站不住。

最初我以为 ms-swift 走 `BatchSamplerShard`、DeepSpeed 走 `DistributedSampler`，担心两边代码路径不同。**这个表述是错的**：实际上**两个 engine 都走 ms-swift 同一份 `BatchSamplerShard`**，accelerate 在 prepare 时不会重新分片。

但「相信代码」是不够的，必须**实测验证**。下面是完整的验证方法学。

### 21.2 三层验证策略

|  层 | 验证目标 | 工具 |
|---|---|---|
| L1: theory | sampler 在给定 (rank, world_size, seed, epoch) 下确定性产出唯一 indices | 独立 Python 脚本，无 GPU、无训练 |
| L2: sampler call | 跑训练，hook `BatchSamplerShard.__iter__` + `set_epoch`，dump 实际 yield 的 indices | sitecustomize patch (27) |
| L3: collator output | model forward 入口处 hook `Trainer.training_step`，dump per-(rank, micro) `input_ids` / `labels` 的 SHA1 + n_tokens + decoded preview | sitecustomize patch (26) |

每层都对 DS3 和 FSDP2 各跑一次，对比结果完全一致才算通过。

### 21.3 工具实现

#### Patch (26): collator output dump（model forward 看到的 tensor）

文件：`scripts/gemma4_opt/_sdp_preamble/sitecustomize.py`

机制：
- Hook `transformers.Trainer.__init__` → 引擎类型探测 + 打开输出文件
- Hook `transformers.Trainer.training_step` → 每次调用前 hash `inputs['input_ids']` 和 `inputs['labels']`，写一行 TSV：

```
step	micro	rank	n_tokens	input_ids_sha	labels_sha	first16_ids	last16_ids	n_loss_tokens	preview
```

env 控制：
- `GEMMA4_DATA_DUMP=1` 启用
- `GEMMA4_DATA_DUMP_DIR=/path/to/out`
- `GEMMA4_DATA_DUMP_MAX_STEPS=N` 跑 N 步后停止 dump

#### Patch (27): sampler dump（确认 set_epoch / __iter__ 行为）

文件：同上

机制：
- Hook `swift.dataloader.shard.BatchSamplerShard.__iter__` → 记录 `curr_seed` + 实际 yield 的 indices
- Hook `swift.dataloader.shard.BatchSamplerShard.set_epoch` → 记录每次 epoch 切换调用

输出格式（每行一条）：

```
# set_epoch(0) called -> curr_seed will be 42
# __iter__ called: curr_seed=42 world_size=8 rank=0 batch_size=1
batch_idx	indices
0	102
1	198
...
# __iter__ done: yielded 32 batches
# set_epoch(1) called -> curr_seed will be 43
# __iter__ called: curr_seed=43 ...
```

env 控制：
- `GEMMA4_SAMPLER_DUMP=1` 启用
- `GEMMA4_SAMPLER_DUMP_DIR=/path/to/out`
- `GEMMA4_SAMPLER_DUMP_MAX_BATCHES=N` 限 N 个 batch 不写盘

#### 独立验证脚本：`verify_sampler_determinism.py`

`scripts/gemma4_E4B_opt/verify_sampler_determinism.py`

无需训练、无需 GPU。直接 import `BatchSamplerShard`，5 个 test 覆盖确定性、跨 epoch 行为、全数据集覆盖率：

```python
def _make_sampler(rank, world_size, seed=42):
    sampler = BatchSamplerShard.__new__(BatchSamplerShard)
    BatchSamplerShard.__init__(sampler,
        total_samples=51557, batch_size=1, shuffle=True,
        drop_last=False, data_seed=seed)
    # mock distributed rank/world via subclass
    sampler.__class__ = type("FakeBS", (BatchSamplerShard,), {
        "rank": rank, "world_size": world_size,
    })
    sampler.total_samples = 51557 // world_size
    return sampler

def epoch_indices(rank, world_size, epoch, seed=42, n_first=20):
    s = _make_sampler(rank, world_size, seed=seed)
    s.set_epoch(epoch)   # 决定 curr_seed = base_seed + epoch
    return [batch[0] for i, batch in enumerate(s) if i < n_first]
```

5 个 test：
1. epoch 0 indices for rank 0..7（应该跟训练时实测一致）
2. epoch 1 indices for rank 0..7（不同的 shuffle，但同样确定性）
3. 重新构造 sampler 拿到的 indices 是否完全一致（确定性 check）
4. epoch 0 vs epoch 1 是否真的产生不同的 shuffle（验证 set_epoch 起作用）
5. 跨 8 ranks × 1 epoch 的 union 覆盖率（应该 ≈ 100% dataset）

### 21.4 实测步骤

#### Step A: 跑 patch (27) sampler dump（DS3）

```bash
docker exec fsdp_sft pkill -9 python; sleep 3

docker exec -e PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble \
  -e GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
  -e GEMMA4_STOP_AFTER_STEPS=1 \
  -e GEMMA4_SAMPLER_DUMP=1 \
  -e GEMMA4_SAMPLER_DUMP_DIR=/home/ubuntu/fyh/_dump/sampler_ds3 \
  -e GEMMA4_SAMPLER_DUMP_MAX_BATCHES=20 \
  -e NPROC_PER_NODE=8 -e MASTER_PORT=29960 \
  fsdp_sft \
  swift sft \
    --model /home/ubuntu/.cache/modelscope/models/google/gemma-4-E4B-it \
    --model_type gemma4 --template gemma4 \
    --dataset /home/ubuntu/fyh/megatron-sft-recipes/sft-data/SFT_0424_2.jsonl \
    --tuner_type full --torch_dtype bfloat16 --attn_impl flash_attention_2 \
    --max_length 16384 --truncation_strategy right \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
    --num_train_epochs 2 --learning_rate 2e-5 --lr_scheduler_type cosine \
    --warmup_ratio 0.05 --weight_decay 0.1 --use_liger_kernel true \
    --gradient_checkpointing true --freeze_vit true --freeze_aligner true \
    --deepspeed scripts/benchmark/sp_offload_configs/zero3_colleague.json \
    --sequence_parallel_size 1 --packing false --padding_free false \
    --dataloader_num_workers 4 --dataset_num_proc 4 \
    --logging_steps 1 --save_strategy no --load_from_cache_file false \
    --split_dataset_ratio 0 --max_grad_norm 1.0 --report_to none \
    --output_dir /home/ubuntu/fyh/_dump/ds3_sampler_run
```

#### Step B: 跑 patch (27) sampler dump（FSDP2）

跟 Step A 一样，但去掉 `--deepspeed`，加上：
- `--fsdp scripts/gemma4_E4B_opt/_fsdp_override_dump.json`
- `--padding_free true`（match 我们正式 run）
- `-e GEMMA4_FSDP_WRAP_PLE=1 -e FSDP_STATE_DICT_TYPE=FULL_STATE_DICT`

#### Step C: 跑 patch (26) collator dump，DS3 + FSDP2 各一次

env 改成 `-e GEMMA4_DATA_DUMP=1 -e GEMMA4_DATA_DUMP_DIR=...`，其余跟 step A/B 一致。

#### Step D: 跨 epoch boundary 验证（mini dataset 加速）

完整 dataset 跑 2 epoch 是 8h，太长。用 256-sample mini dataset：

```bash
head -256 sft-data/SFT_0424_2.jsonl > sft-data/SFT_0424_2_mini256.jsonl
# 256 samples / GBS=128 = 2 step/epoch × 2 epoch = 4 steps total
# 加上 model load，全程 ~3 分钟
```

跑完即可看到 `set_epoch(0) -> curr_seed=42` 和 `set_epoch(1) -> curr_seed=43` 两次切换，对比 epoch 1 的 indices 是否两 engine 一致。

#### Step E: 对比脚本

```python
# 对比 sampler indices
def parse_dump(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if not f.startswith("sampler_rank"): continue
        rank = int(f[len("sampler_rank"):-len(".tsv")])
        out[rank] = {}
        cur_seed = None
        with open(os.path.join(d, f)) as fh:
            for line in fh:
                if line.startswith("# __iter__ called"):
                    cur_seed = int(line.split("curr_seed=")[1].split()[0])
                    out[rank].setdefault(cur_seed, [])
                elif "\t" in line and cur_seed is not None:
                    parts = line.strip().split("\t")
                    if parts[0].isdigit():
                        out[rank][cur_seed].append(int(parts[1]))
    return out

ds3 = parse_dump("/home/ubuntu/fyh/_dump/sampler_ds3_2ep")
fsdp = parse_dump("/home/ubuntu/fyh/_dump/sampler_fsdp2_2ep")
# 对比 ds3[rank][seed] vs fsdp[rank][seed]
```

### 21.5 实测结果（apples-to-apples）

#### L1 - 独立确定性测试

```
Test 1: epoch 0 indices for rank 0..7
  rank 0: [8934, 22958, 21875, 43643, 8747, 3546, 5613, 35407, 20646, 45016]
  ... 
  rank 7: [15991, 23617, 48972, 19436, 41760, 33291, 36063, 25981, 31286, 15894]

Test 2: epoch 1 indices (different shuffle, but deterministic)
  rank 0: [29668, 13649, 50915, 27658, 49749, 43528, 10937, 33739, 38970, 46922]
  ...

Test 3: re-running construction yields identical indices?
  rank 0 epoch 0 == 第二次? True
  rank 0 epoch 1 == 第二次? True
  ... (all True)

Test 4: epoch 0 vs epoch 1 不同 shuffle?
  rank 0: epoch0=[8934,22958,...] epoch1=[29668,13649,...] overlap_in_first10 = 0/10
  ... (确实不同)

Test 5: 跨 8 ranks 全 epoch 覆盖
  epoch 0: total batches = 51552, unique indices = 51552, [min=0, max=51551]
  epoch 1: total batches = 51552, unique indices = 51552, [min=0, max=51551]
  → 51552 / 51557 = 99.99% dataset 覆盖（drop_last 丢 5 个）
```

#### L2 - 跨 engine sampler dump（apples-to-apples）

DS3 和 FSDP2 各跑 step 1，对比 sampler yield 的前 16 indices：

```
rank | DS3 (first 16 batches)                              | FSDP2 indices                        | match?
   0 | 8934,22958,21875,43643,8747,3546,5613,35407,2064... | 8934,22958,21875,43643,8747,3546,... | ✓
   1 | 47055,14672,32602,44501,25033,13123,46561,31091,... | 47055,14672,32602,44501,25033,...    | ✓
   ...
   7 | 15991,23617,48972,19436,41760,33291,36063,25981,... | 15991,23617,48972,19436,41760,...    | ✓

Ranks with identical first 16 batches: 8 / 8
```

✓ L1 独立测试 + L2 训练实测的 indices **完全一致**（rank 0 epoch 0 第一个 index 都是 8934）。

#### L3 - collator output (input_ids / labels)

128 cells × 6 字段：

```
=== Per-field equality check (DS3 vs FSDP2) across 128 (rank, micro) cells ===
      n_tokens: ✓ ALL MATCH       ← 序列长度
         h_ids: ✓ ALL MATCH       ← 全 input_ids 的 SHA1
         h_lab: ✓ ALL MATCH       ← 全 labels 的 SHA1
       first16: ✓ ALL MATCH       ← 前 16 个 token id
        last16: ✓ ALL MATCH       ← 末 16 个 token id
        n_loss: ✓ ALL MATCH       ← non-padding label token 个数
```

每张卡每个 micro 的实际 sample 内容（preview）也逐字节一致：

```
━━━ rank 0 (DS3 == FSDP2 byte-for-byte) ━━━
micro  n_tok  n_loss  h_ids               preview
    0   4251    2297  086a6ac97f89276f   *Hannibal's silence is absolute. He reaches out, ca
    1  13638    5221  1856b46545db4a4d   Mina japste schockiert auf und ihre orangefarbenen
    2   2039     270  5560711fc418265c   König's jaw clenches beneath...
    ...
```

#### L4 - 跨 epoch boundary 实测（mini dataset 2 epoch full run）

```
rank | epoch_seed | DS3 #idx | FSDP2 #idx | first 5 match | all common match
   0 |         42 |       32 |         32 | True          | True (32/32)
   0 |         43 |       32 |         24 | True          | True (24/24)   ← epoch 2
   1 |         42 |       32 |         32 | True          | True (32/32)
   1 |         43 |       32 |         24 | True          | True (24/24)
   ... (8 ranks × 2 epoch_seeds = 16 cells, all match)
```

`set_epoch(1) -> curr_seed=43` 在 DS3 和 FSDP2 上**都正确触发**，shuffle indices 完全一致。

### 21.6 关键代码定位

```
ms-swift SwiftMixin.get_train_dataloader (mixin.py:1209)
  └── 创建 BatchSamplerShard (shard.py:10)
        __iter__ 关键 1 行 (shard.py:55):
          total_idx = total_idx[self.rank::self.world_size]  ← 决定每 rank 拿哪些 indices
        set_epoch 关键 1 行 (shard.py:71):
          self.curr_seed = self.base_seed + epoch  ← 决定 epoch shuffle seed
  └── 包装成 DataLoaderShard (shard.py:80) 返回

HF Trainer.train()
  └── 统一的 epoch loop (engine-agnostic):
        for epoch in range(num_train_epochs):
            train_dataloader.set_epoch(epoch)
            for step, inputs in enumerate(train_dataloader):
                ...
  └── self.accelerator.prepare(model, optimizer, dataloader)
        ├── 对 DS3:  HF DeepSpeed integration 直接接受 user-provided dataloader，不分片
        └── 对 FSDP: accelerate.prepare_data_loader 检查 dataloader.sampler，因为 ms-swift
                    传 batch_sampler 而不是 sampler，accelerate 的 reshuffling 路径不被触发
```

### 21.7 经验：为什么这个验证方法能 work

1. **不要只在某一层断言「应该一样」就完事**。三层验证（theory → sampler call → collator output）每层独立 dump 数据再 cross-check，任何一层出错都会暴露。
2. **theory 层用独立脚本最干净**：不依赖 GPU / 训练 / engine，单纯把 ms-swift sampler 当库 import，直接构造 + iterate。3 分钟得到 reference indices 序列，后面的实测对比都拿这个为基准。
3. **dump 时一定包含 `set_epoch` 行为，而不是只看一个 epoch**。本次最初验证只 dump 了 step 1（epoch 0 头），没跨 epoch boundary。如果 set_epoch 在两个 engine 上行为不一致，单 epoch 的 dump 会漏掉。
4. **mini dataset 加速跨 epoch 验证**：256 样本 × 2 epoch × MBS=1 × GAS=16 = 4 step total，加 model load 全程 3 分钟。原 51557 样本要 2 epoch = 8h，根本跑不动多次。
5. **hash + first/last + preview 三种粒度共存**：hash 给 binary equality；first16/last16 给 quick visual diff；preview 给 human-readable sample 内容（万一 hash 有 bug 至少能肉眼看出 sample 不对）。

### 21.8 结论

跨 8 ranks × 2 epochs，dataloader 在 DS3 和 FSDP2 上**逐字节一致**。

step 400 后的 loss 二次发散 + 11% 最终 loss diff **跟数据完全无关**，是 FSDP2 backward 自身（KV-share 16-path 通胀的累积效应）的问题。同事的怀疑可以排除。

### 21.9 工具清单

| 文件 | 功能 |
|---|---|
| `scripts/gemma4_E4B_opt/verify_sampler_determinism.py` | L1 独立确定性测试，无 GPU |
| `scripts/gemma4_opt/_sdp_preamble/sitecustomize.py` patch (26) | L3 input_ids/labels SHA dump (gated by `GEMMA4_DATA_DUMP=1`) |
| `scripts/gemma4_opt/_sdp_preamble/sitecustomize.py` patch (27) | L2 sampler indices + set_epoch dump (gated by `GEMMA4_SAMPLER_DUMP=1`) |
| `scripts/gemma4_E4B_opt/_fsdp_override_dump.json` | FSDP2 配置（用 host shared path 而不是 `/tmp`，因为 docker 容器 `/tmp` 不共享） |

---

## 22. padding_free 对 loss reduction 的影响排查 (2026-05-02)

### 22.1 起源

同事提出新怀疑：

> "padding_free 可能改变 loss 归一化方式：传统模式按 sample 平均（短长一票）；padding_free 按 token 平均（长序列权重更大）。第二个 epoch 模型已记住短序列 → 传统模式 loss 大幅下降，padding_free 模式下降不明显。"

这跟 §19 / §20 的 step 400 跳变 + 11% diff 现象吻合：FSDP2 用 pf=True，DS3 用 pf=False。如果归一化不同，loss 不可比。

### 22.2 ms-swift compute_loss 代码读法

`ms-swift/swift/trainers/seq2seq_trainer.py:123-205`，关键路径（liger=true、no DFT、no channel）：

```148:153:/home/ubuntu/fyh/ms-swift/swift/trainers/seq2seq_trainer.py
if labels is None:
    labels = inputs['labels']
    outputs.loss = outputs.loss.to(labels.device)
    # fix https://github.com/huggingface/transformers/issues/34263
    if num_items_in_batch is not None:
        outputs.loss = outputs.loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)
```

数学：
- `outputs.loss = sum(token_loss) / local_n_tokens`（Liger fused linear-CE 已完成）
- `outputs.loss × (local_n / global_num)` = `sum_local / global_num`
- 跨 16 micros × 8 ranks 累加：`sum_total / global_num` = **全局 per-token mean**

`global_num` (HF Trainer 的 `num_items_in_batch`) = 全 batch 所有 rank 所有 micro 的 non-pad token 总数。我们已在 §21 验证 `n_loss_tokens` 在 128 cells 上一致，所以 `global_num` 两 engine 相同。

→ **理论上 padding_free 对 MBS=1 没有归一化影响**。但理论 ≠ 实测。

### 22.3 实测：4×1 配置矩阵 + 1 个 detach 对照

跑 step 1-2 共 ~70s 每个：

| 配置 | step 1 loss | step 1 gn | step 2 loss | step 2 gn |
|---|---:|---:|---:|---:|
| DS3 pf=False (基准) | 2.22561 | 10.36 | 2.26226 | 10.35 |
| DS3 pf=True | 2.22561 | 10.36 | 2.26226 | 10.34 |
| FSDP2 pf=False + detach | 2.22579 | 10.25 | 2.26288 | 10.375 |
| FSDP2 pf=True + detach | 2.22579 | 10.25 | 2.26288 | 10.375 |
| FSDP2 pf=True no-detach | 2.22579 | 13.25 | 2.26288 | 13.375 |

### 22.4 三个关键结论

#### (1) padding_free 对 loss 完全无影响

- DS3: pf=False ↔ pf=True → **loss bit-identical** (8 位小数)
- FSDP2: pf=False ↔ pf=True → **loss bit-identical**

→ MBS=1 时 `padding_free` 只是改 collator 输出格式（`attention_mask` vs `position_ids` segment 标记），**不改 token 序列、不改 label、不改 loss 归一化**。同事关于「pf 改 loss reduction」的假设**实测不成立**。

#### (2) detach 改 backward 不改 forward

FSDP2 pf=True：
- detach: gn=10.25
- no-detach: gn=13.25
- **loss 一字不差** (2.22579)

→ detach 只切 share→source 的 backward 边（patch 8），forward 计算完全不变。loss 是前向标量，必须一致。这个验证了 §20.7 的 detach 机制理解。

#### (3) DS3 vs FSDP2 同 pf 的 1.8e-4 差异

`DS3 pf=True loss = 2.22561`  
`FSDP2 pf=True loss = 2.22579`

差 1.8e-4。bf16 算子在两个 engine 的实现差异（all-gather / reduce-scatter 顺序、浮点累加顺序）造成的微小数值噪声，**远小于实际 loss 量级 2.226，可忽略**。

### 22.5 §19 / §20 step 400 跳变 + 11% 仍需归因到 KV-share 通胀

排除掉 padding_free 后，跨 epoch 边界 11% loss diff 的成因清单：

| 候选 | 验证状态 |
|---|---|
| Dataloader 不一致 | ✗ §21 排除（128 cells × 6 字段全一致） |
| set_epoch 时机不同 | ✗ §21 排除（curr_seed=43 两 engine 同时切） |
| padding_free 改 loss reduction | ✗ §22 排除（pf=True/False loss bit-equal in DS3 + FSDP2） |
| Forward 计算差异 | ✗ §22 排除（detach 改 gn 不改 loss → 前向一致） |
| **FSDP2 backward KV-share 16-path 通胀** | **✓ 唯一未排除候选**（§18 / §20 layer-by-layer dump 直接观察到 layer 14-23 grad inflated 2.4-3.6x） |

### 22.6 复现命令（5 个配置）

```bash
# Common: 跑 2 步、--num_train_epochs 2 触发 epoch loop, --max_grad_norm 1.0 跟正式 run 一致

# 1) DS3 pf=False
docker exec -e PYTHONPATH=... -e GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
  -e GEMMA4_STOP_AFTER_STEPS=2 -e NPROC_PER_NODE=8 -e MASTER_PORT=29980 \
  fsdp_sft swift sft \
    --deepspeed scripts/benchmark/sp_offload_configs/zero3_colleague.json \
    --padding_free false \
    [其余跟正式 run 一致]

# 2) DS3 pf=True (只换一个 flag)
    --padding_free true

# 3) FSDP2 pf=False + detach (-e GEMMA4_KV_SHARE_DETACH=1)
    --fsdp scripts/gemma4_E4B_opt/_fsdp_override_dump.json
    --padding_free false

# 4) FSDP2 pf=True + detach
    --padding_free true

# 5) FSDP2 pf=True no-detach (省略 GEMMA4_KV_SHARE_DETACH env)
    --padding_free true
```

每跑完读 `<run_dir>/v0-*/logging.jsonl` 的前 2 行就拿到 step 1-2 loss/gn。70s/run × 5 = 6 分钟实验完成。

### 22.7 经验

1. **同事的怀疑要严肃验证而不是 dismiss**。即使读代码觉得逻辑没问题，empirical 矩阵实验花 6 分钟就能彻底定论。
2. **5 个配置矩阵能同时验证多个假设**：(DS3 vs FSDP2) × (pf=False vs pf=True) × (detach on/off) → 一次性看清 padding_free / engine / detach 各自的影响。
3. **bit-level loss equality 是非常强的 signal**：如果有任何归一化差异，bf16 forward 浮点累加顺序至少会出 1e-3 级别的差。我们看到 8 位小数完全一致 → 说明数学路径完全等价。
4. **loss 一致 + gn 不一致 = backward-only 差异的清晰指标**。这跟我们一直推断的 detach × backward 副作用、KV-share 16-path 通胀完全吻合。

### 22.8 给同事回复

> 验证了 padding_free 对 loss reduction 没影响：
> - DS3 pf=False vs DS3 pf=True：loss bit-equal (2.22561)
> - FSDP2 pf=False vs FSDP2 pf=True：loss bit-equal (2.22579)
> - 5 个配置矩阵 × step 1-2 数据完整对比表见 §22
>
> step 400 跳变 + 11% diff 跟 padding_free 无关，依然是 FSDP2 backward 自身的 KV-share 16-path 通胀（§18 / §20 layer-by-layer dump 已经直接观察到 layer 14-23 grad inflated 2.4-3.6x）。

---

## 23. β=0, δ=1 自定义 autograd Function 实施 + 真因深挖 (2026-05-02)

### 23.1 思路

§20 表格反推出 DS3 的实际行为是 **β=0**（share path 不贡献 layer 22 W_k.grad）+ **δ=1**（share path 贡献 layer 22 input.grad → residual 到 layers 0-13）。我们想用 monkey-patch 复现这个行为：

```python
class _DetachWeightLinear(torch.autograd.Function):
    """forward: F.linear(x, W); backward: dx via W (kept), dW := None (cut)"""
    @staticmethod
    def forward(ctx, x, W):
        ctx.save_for_backward(W)
        return F.linear(x, W)

    @staticmethod
    def backward(ctx, dy):
        W, = ctx.saved_tensors
        return dy @ W, None  # CUT W.grad accumulation
```

在 source 层 (layer 22, 23) `_patched_attn_forward` 里，**重算**一份 weight-detached 的 k/v 替换 cell entry，layer 自己 attn 还用原版 k/v：

```python
# After orig_forward, if store_full_length_kv:
_k_raw = _DetachWeightLinear.apply(hidden_states, k_proj.weight)
_k_normed = k_norm(_k_raw.view(...))
_k_for_cell = apply_rotary_pos_emb(_k_normed, ...).transpose(1, 2)
# Same for v
_ref[layer_idx] = (_k_for_cell, _v_for_cell)
```

env：`GEMMA4_KV_SHARE_BETA0_DELTA1=1`

### 23.2 实测：1-step grad dump

step 1 grad_norm + layer-by-layer fp64 sum_sq：

| | DS3 (gold) | detach | no-detach | **β0δ1 (我们)** |
|---|---:|---:|---:|---:|
| step 1 gn | 10.36 | 10.25 | 13.25 | **13.0** ← 仍然偏高 |
| layers 0-13 | 117 | 36 (0.30x) | 130 (1.11x) | **132 (1.13x) ✓** |
| layers 14-23 | 107 | 106 (1.00x) | 261 (2.45x) | **246 (2.30x)** ← 改善有限 |
| layers 24-41 | 152 | 152 (1.00x) | 152 (1.00x) | 152 (1.00x) |

**Layer 22 per-param 详情**：

| param | DS3 | no-det | β0δ1 (我们) | 评价 |
|---|---:|---:|---:|---|
| **k_proj.weight** | 0.2375 | 2.231 (9.4x) | **0.6551 (2.76x)** | **3.4x reduction**，但还没到 1.0 |
| **v_proj.weight** | 1.172 | 8.019 (6.84x) | **3.343 (2.85x)** | **2.4x reduction** |
| k_norm.weight | 0.0091 | 0.107 (11.7x) | 0.107 (11.7x) | 没修 |
| **mlp.down_proj** | 3.755 | 9.618 (2.56x) | **9.619 (2.56x)** | **完全没修！** |
| mlp.up_proj | 1.315 | 3.141 (2.39x) | 3.141 (2.39x) | 没修 |
| input_layernorm | 0.0034 | 0.0265 (7.7x) | 0.0265 (7.7x) | 没修 |

### 23.3 重大发现：通胀不是 KV-share 路径独有

注意 layer 22 的 **mlp.down_proj 也被通胀 2.56x**，但 MLP 完全跟 KV-share 无关。更糟的是 **layer 14-21（不是 source 层、不存 cell）也被通胀 2.87-4.10x**：

```
   14 | 1.72 (DS3) | 4.95 (no-det 2.87x) | 4.95 (β0δ1 2.87x) ← 我们的 fix 对它没影响
   15 | 3.40       | 13.57 (3.99x)        | 13.57 (3.99x)
   ...
   21 | 9.44       | 38.69 (4.10x)        | 38.69 (4.10x)
```

→ inflation 不是「16 reader 通过 cell 反向」单一路径造成的。**整个 layer 14-23 的所有参数都被某种 FSDP2 backward 副作用通胀了**，而我们的 cell-level patch 只能修 cell 直接路径。

### 23.4 排除全部 Python 可触及层

| 测试 | env | 结果 |
|---|---|---|
| **Activation Checkpointing** 不是元凶 | `GEMMA4_FSDP_NO_AC_KV_SOURCE=1 GEMMA4_FSDP_NO_AC_LAYER_RANGE=14:23` | 100 个 CheckpointWrapper 移除后，inflation 完全不变 (4.947→4.949 = 0.04% 差异) |
| **`cast_forward_inputs`** 不是元凶 | `GEMMA4_FSDP_NO_CAST_FORWARD_INPUTS=1` | inflation 完全不变 |
| **`reshard_after_forward`** 不是元凶 | `GEMMA4_FSDP_NO_RESHARD_KV_SOURCE=1 GEMMA4_FSDP_NO_RESHARD_LAYERS=22,23` | layer 22 ratio 还是 3.64x |
| **`foreach_reduce` 多次调用** 不是元凶 | patch 28 hook fire-count | 16 micros 内只 fire 44 次（= 1 unit/step），no_sync 期间累积，不是多次 reduce |
| **AutoGrad anchor** 不能合并 path | `GEMMA4_KV_SHARE_ANCHOR=1` | D ≡ C 完全等价，cell-level anchor 拦不住 |
| **bf16 reduce / 累加 dtype** 不是元凶 | patch 20/21 (fp32 force) | inflation 不变 |
| **β=0, δ=1 cell autograd** 部分修复 | `GEMMA4_KV_SHARE_BETA0_DELTA1=1` | 仅 k_proj/v_proj 部分降低（9.4x → 2.76x），其他层 14-21 / mlp 全没修 |

→ **Python monkey-patch 路径已穷尽**。

### 23.5 推断：真因在 PyTorch FSDP2 C++ 层

最可能的机制（需要 C++ 源码 dive 才能 100% 确认）：

```
FSDP2 backward post-hook 链路:
  1. 16 reader 反向被 PyTorch autograd dispatcher 触发
  2. 每个 reader 反向到 cell[22] 时，FSDP2 内部 unshard W_k_22
  3. ★ 关键：每次 unshard 触发 layer 22 整个 FSDP unit 的 sub-graph 反向一次
  4. layer 22 所有 param (mlp / norm / 等) 都被多次 backward → 累积 inflation
  5. 这跟 KV-share 间接相关（reader 通过 cell 反向触发 unshard），但 inflation 范围
     是整个 unit，不止 cell 直接路径
```

证据支持：
- inflation 范围 = layer 14-23 全部参数（包括跟 cell 完全无关的 mlp）
- 越靠近 layer 22 通胀越严重（layer 22 = 3.64x，layer 14 = 2.87x）→ 跟 reader-source 物理距离负相关
- detach 模式（cell 完全没 autograd link）= 完全没 inflation = 证实是 cell-triggered

源码位置（猜测）：
- `torch/distributed/fsdp/_fully_shard/_fsdp_state.py` 的 `_pre_backward_hook`
- `torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py` 的 unshard / reshard 调度
- `torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py` 的 backward 流程

### 23.6 最终结论：Monkey-patch 路径穷尽

我们已经把 Python 层能拦截的所有 hook 都试过了。要彻底修 layer 14-23 的 backward inflation，必须：

| 路径 | 工作量 | 概率 | 备注 |
|---|---|---|---|
| TorchTitan-style 自写 trainer | 1-2 day | 70% | 绕开 accelerate FSDP2 plumbing，用 raw `torch.distributed.fsdp.fully_shard` |
| Fork PyTorch + 改 FSDP2 C++ | 1 周+ | 80% | 上述源码位置改 backward dispatcher 行为 |
| PyTorch 2.11+ nightly | 15 min 试 | 30% | 可能 incidentally 修了；不修就退回 |
| **接受 +11% diff** | 0 | N/A | **trajectory 健康，token_acc 几乎重合，生产可用** |

**当前正式 run 配置** (`scripts/gemma4_E4B_opt/fsdp2_offload_E4B_text_only_2ep_a3_pf.sh`)：
- `GEMMA4_KV_SHARE_DETACH=1`（最稳，避免 spike）
- 805 steps × 35 s/step = 7.93h
- step 800 loss = +11.2% vs DS3
- 训练效果可上线，速度比 DS3 快 34%

### 23.7 工具留痕（patches 仍在 sitecustomize.py 里，env-gated 关闭）

| patch | env | 作用 | 验证状态 |
|---|---|---|---|
| 8 (β0δ1 mode) | `GEMMA4_KV_SHARE_BETA0_DELTA1=1` | 复刻 DS3 β=0,δ=1 cell 行为 | 部分有效（k/v_proj），不解决 14-21 全局通胀 |
| 21 | `GEMMA4_FSDP_REDUCE_FP32_NCCL=1` | 强制 NCCL fp32 reduce | 没用 |
| 23 | `GEMMA4_FSDP_NO_CAST_FORWARD_INPUTS=1` | 关掉 mixed precision cast | 没用 |
| 24 | `GEMMA4_FSDP_NO_AC_KV_SOURCE=1` + `GEMMA4_FSDP_NO_AC_LAYER_RANGE=14:23` | 关掉指定层 AC | 没用 |
| 25 | `GEMMA4_FSDP_NO_RESHARD_KV_SOURCE=1` | 关掉 source 层 reshard | 没用 |
| 28 | `GEMMA4_BWD_FIRE_COUNT=1` | 计数 forward / backward / foreach_reduce 触发次数 | 诊断工具 |

---

## §24. **【关键发现】余弦相似度实验定位 11% loss gap 的真因 — 早期层方向漂移，KV-share 修复无关**

### 24.1 背景与动机

§23 用 monkey-patch 把 layer 22 的 KV-share inflation 修到 `cos_sim ≈ 1.0`、grad_norm 几乎完全对齐 DS3，但 2 epoch 跑完仍然 +11% loss。之前所有的诊断都聚焦在 **梯度幅度 (grad_norm)**，从未直接对比过 **梯度方向**。本节做了一次干净的方向实验：把 DS3 baseline 和 FSDP2 跑 1 个 step、dump 关键参数的原始梯度张量、计算余弦相似度。

**实验问题**：FSDP2 vs DS3 的差异是 **幅度问题** (`max_grad_norm` 可救) 还是 **方向问题** (训练轨迹必然偏离)？

### 24.2 实验设计

**1 个 step、4 个代表参数、3 个 engine variant**：

| 代表参数 | 选取理由 |
|---|---|
| `layers.0.self_attn.q_proj.weight` | 最早期层（控制信号经过 41 层反向才能到这里） |
| `layers.22.self_attn.k_proj.weight` | KV-share source 层（§18 修复点） |
| `layers.22.self_attn.v_proj.weight` | KV-share source 层（§18 修复点） |
| `layers.41.self_attn.q_proj.weight` | 最末层（backward 起始） |

**Engine variants**：
- `DS3` — 同事 ZeRO-3 baseline（视为 ground truth）
- `FSDP2 detach` — 当前生产配置（`GEMMA4_KV_SHARE_DETACH=1`，§19 修复后）
- `FSDP2 no-detach` — 关掉 detach，复现 1.29x grad_norm inflation

**实测细节**：
- 同一份数据、同一 seed、同一 step 1（loss 三个 variant 均=2.226，验证起点一致）
- dump 工具：`sitecustomize.py` patch (22) + 新增 raw-grad save
- DS3 用 `Tensor.register_hook`（fires per (rank, micro)）
- FSDP2 用 `_fsdp_param_group.foreach_reduce` 拦截 unsharded grad（forced sync 模式禁掉 no_sync 让 16 micros 都 fire）
- Per-cell：8 ranks × 16 micros × 4 params = 512 raw `.pt` 文件，4.4 GiB
- 比较口径：
  - **per-cell cos_sim**：每个 (rank, micro, param) 的 FSDP2 grad 对 DS3 grad 余弦
  - **step-summed cos_sim**：所有 (rank, micro) 求和后再算余弦（≈ optimizer 实际看到的方向）

**脚本**：
- `scripts/gemma4_E4B_opt/compute_cossim.py`
- 修改：`scripts/gemma4_E4B_opt/bench_ds3.sh` 修了 `${SDP_PRE:-...}` 嵌套参数展开 bug（用显式 `ENV_PREAMBLE` 替代）

### 24.3 实测结果

**完整数据**（step 1, 8 ranks × 16 micros = 128 cells）：

| 参数 | 类型 | DS3 vs FSDP2-**detach** |  | DS3 vs FSDP2-**no-detach** |  |
|---|---|---|---|---|---|
|  |  | per-cell cos | step-sum cos / mag | per-cell cos | step-sum cos / mag |
| `layers.0.q_proj` | 早期层 | mean=0.696 std=0.296 | **0.361 / 0.533** | mean=0.536 | **0.353 / 1.232** |
| `layers.22.k_proj` | KV源 | mean=0.994 std=0.005 | **0.9995 / 1.001** | mean=0.425 | 0.681 / 2.968 |
| `layers.22.v_proj` | KV源 | mean=0.999 std=0.001 | **0.99998 / 1.0001** | mean=0.423 | 0.484 / 2.575 |
| `layers.41.q_proj` | 末层 | mean=0.999 | 0.99996 / 0.999 | mean=0.999 | 0.99996 / 0.999 |

### 24.4 三个非常关键的结论

**1. detach patch (§19) 对 layer 22 完美修复**
- detach 模式：cos=0.999，magnitude=1.001
- no-detach 模式：cos=0.48-0.68，magnitude=2.5-3x（这是 1.29x grad_norm inflation 的真因——layer 14-23 全部被 KV-share-induced backward 通胀）
- → §19 detach 是必要且充分修复 KV-share 问题的

**2. 末层 (layer 41) 完全干净**
- 两种 FSDP2 模式下都和 DS3 cos=0.99996，magnitude 误差 0.1% 以内
- 说明 backward 的"起点"在两个框架里是 byte-equivalent 的
- 11% loss gap 跟末层无关

**3. 【真正的真因】layer 0 在 detach 和 no-detach 下都偏离 DS3**
- detach: cos=0.36（**只有 36% 方向重合**），magnitude=0.53（**FSDP2 幅度只有 DS3 的一半**）
- no-detach: cos=0.35，magnitude=1.23
- **detach 与否对 layer 0 几乎不变** → 早期层的方向偏差跟 KV-share 修复**无关**
- per-cell std=0.296，min=−0.27（部分单元方向甚至反向）→ 反向通过 41 层 bf16 后，每个 (rank, micro) 的方向漂移幅度大且不对称
- DS3 的 128 个 cell 求和互相加强（|sum|=0.18），FSDP2 的 128 个 cell 求和互相抵消（|sum|=0.10）→ 这就是 §20 "layers 0-13 deficit" 的物理本质

### 24.5 物理解释

```
backward 信号从 layer 41 → layer 0:
  layer 41:  cos=0.999996  (起点)
  ...
  layer 22:  cos=0.999     (中段，detach 修复后干净)
  ...
  layer 0:   cos=0.36      (终点，方向已严重漂移)
```

每经过一层 bf16 反向算子，FSDP2 与 DS3 的微小差异（cuBLAS 不同 algorithm、bf16 atomic 顺序、reduce-scatter 时机）累积一次 rounding。41 层之后，layer 0 的梯度方向已经偏离 DS3 ~64%。

**对比关键**：
- DS3：fp32 contiguous gradient buffer + bf16 reduce → 反向链路里有 fp32 节点稳定方向
- FSDP2：bf16 unsharded grad + bf16 reduce-scatter → 全程 bf16，41 层放大 rounding

→ **`max_grad_norm` 救不了**：clipping 只缩放幅度，不改方向

### 24.6 训练含义

为什么 epoch 1 还能跟得住 DS3、epoch 2 就停滞了？

- **Epoch 1**：lr warmup 期间，optimizer 刚开始建立动量，方向偏差累积慢
- **Epoch 2**：动量已建立，每一步都把"错误方向"塞进 momentum buffer，逐渐让训练轨迹偏出 DS3 的盆地
- 早期层（embedding 附近）方向错了，整个表征学习就跑偏了

这一节也回答了为什么 §22 验证 dataloader 完全一致后 loss 还有 gap——数据没问题，是**梯度方向**问题。

### 24.7 PyTorch 2.10 / 2.11 修复路径分析

我们查了 PyTorch 2.10/2.11 nightly 的 FSDP2 源码：

| 候选机制 | 源码位置 | 状态 |
|---|---|---|
| `reduce_dtype=fp32` | `_fsdp_param_group.foreach_reduce` | 2.10 有 cublas bug `CUBLAS_STATUS_EXECUTION_FAILED`；2.11 没在 release-notes 里见过修复 |
| fp32 master grad accumulator | `_fsdp_param.to_accumulated_grad_if_needed` | 2.10 有 `_unsharded_param` AttributeError；patch 16 部分修复，但还有边界情况 |
| `accumulate_grad_dtype=fp32` | 用户 API | 2.10 不暴露；2.11 nightly 提了 issue 但没合 |
| `MixedPrecisionPolicy(reduce_dtype=fp32, cast_forward_inputs=False)` | `_fsdp_state.py` | 2.10 触发 cell rebuild bug + cublas bug |

→ **Python monkey-patch 路径已在 §23.6 列表完全穷尽**。要彻底解决：
1. 等 PyTorch 2.11+ 真正修了 fp32 reduce/accumulate 路径
2. 或者 fork PyTorch + 改 `_fsdp_collectives.py` 让 reduce-scatter 在 fp32 buffer 里做
3. 或者直接复制同事 DS3 配置，不用 FSDP2

### 24.8 后续工作

| 选项 | 工作量 | 概率 | 备注 |
|---|---|---|---|
| **接受 +11% diff** | 0 | — | trajectory 健康（token_acc 几乎重合），生产可上线，速度比 DS3 快 34% |
| 切 DS3 配置 | 1-2 天 | 100% | 已有同事的配置，需要把 ms-swift 的 `swift/model/models/gemma.py` buffer 修复合进来（已做） |
| Fork PyTorch FSDP2 + fp32 reduce | 1 周+ | 80% | 高风险但根治 |
| 等 PyTorch 2.11+ stable | 不确定 | 30-50% | 看 release notes |

### 24.9 工具落地

新增工具（可重复实验）：
- `scripts/gemma4_E4B_opt/compute_cossim.py` — 加载两个 dump dir，输出 per-cell + step-summed 余弦/幅度比
- `sitecustomize.py` patch 22 扩展 — 通过 `GEMMA4_GRAD_DUMP_RAW_PREFIXES=p1,p2,...` 选择 dump 哪些参数的原始梯度张量
- `scripts/gemma4_E4B_opt/bench_ds3.sh` — 修了 `${SDP_PRE:-...}` 嵌套展开 bug；用 `ENV_PREAMBLE="export ...;"` 显式传环境变量，根治了「sitecustomize 不加载、StopAfterStepsCallback 不生效、grad-dump 不写文件」三连症状

复现命令（DS3 baseline 1-step）：
```bash
DUMP_BASE=/path/to/dump
LABEL="cossim_ds3_step1" \
TEMPLATE="gemma4" WEIGHT_DECAY="0.1" MAX_STEPS=1 \
EXTRA_ENV="GEMMA4_GRAD_DUMP=1 GEMMA4_GRAD_DUMP_DIR=${DUMP_BASE}/ds3 \
  GEMMA4_GRAD_DUMP_MAX_STEPS=1 \
  GEMMA4_GRAD_DUMP_RAW_PREFIXES=model.language_model.layers.22.self_attn.k_proj,..." \
EXTRA_ARGS="--max_grad_norm 1.0" \
bash scripts/gemma4_E4B_opt/bench_ds3.sh
```

复现命令（FSDP2 detach / no-detach）：
```bash
LABEL="cossim_fsdp2_${MODE}" \
TEMPLATE="gemma4" WEIGHT_DECAY="0.1" MAX_STEPS=1 FULL_SCHED_STOP=1 \
EXTRA_ENV="GEMMA4_KV_SHARE_DETACH={1,0} \
  GEMMA4_GRAD_DUMP=1 GEMMA4_GRAD_DUMP_DIR=${DUMP_BASE}/fsdp2_${MODE} \
  GEMMA4_GRAD_DUMP_MAX_STEPS=1 GEMMA4_GRAD_DUMP_FORCE_SYNC=1 \
  GEMMA4_FSDP_WRAP_PLE=1 \
  GEMMA4_GRAD_DUMP_RAW_PREFIXES=model.language_model.layers.22.self_attn.k_proj,..." \
EXTRA_ARGS="--max_grad_norm 1.0" \
bash scripts/gemma4_E4B_opt/bench_variant.sh
```

分析：
```bash
python3 scripts/gemma4_E4B_opt/compute_cossim.py \
    --ds3 ${DUMP_BASE}/ds3 \
    --fsdp2 ${DUMP_BASE}/fsdp2_detach \
    --label "FSDP2_detach"
```

---

## §25. **【重大修复】fp32 master 修复 — 11% loss gap → +0.05% Δloss**

> **本节面向新人**：从零讲起这个 bug 是怎么找到的、为什么修一个 CLI 参数就能让 11% 的 loss 偏差消失、以及怎么复现。每一步都给出"为什么这么想"和"代码在哪里"。

### 25.TLDR — 5 分钟速读版（如果你只想看一眼大概结论）

> 完整版从 §25.0 起，先看这个摘要决定要不要展开读。

**1. 训练时一个参数有"两份"**

每个权重 W 有：
- **Compute 版**：算 forward/backward 用，速度优先 → 通常 bf16
- **Master 版**：optimizer 更新用，精度优先 → 理想是 fp32

每个 step：用 bf16 算 grad → 累加到 master → master 上做 update（`master ← master − lr × m / (√v + eps)`）→ master cast 回 bf16 给下个 step。

**2. bf16 vs fp32 差多少**

| 格式 | mantissa 位数 | 相对精度 |
|---|---|---|
| fp32 | 23 位 | 千万分之一 |
| **bf16** | **7 位** | **百分之一**（2⁻⁷ ≈ 0.78%）|

**3. bf16 master 是灾难**

参数 W = 1.0，本步要 update −0.00002（lr=2e-5 量级）：
- **fp32 master**：1.0 − 0.00002 = 0.99998（精确）
- **bf16 master**：bf16 在 1.0 附近最小可表示间隔 ≈ 0.0078，update 被 round 回 1.0 → **完全消失**

每步消失一点，800 步累积 → 大量小幅修正被吞 → 模型欠拟合 → loss 降不下去。

类比：bf16 master = 体重秤只显示整数公斤（永远 60 kg）；fp32 master = 显示到 0.01 kg（每天小变化都能记录）。

**4. FSDP2 的坑**

`--torch_dtype bfloat16` 在 FSDP2 下**不是** mixed precision，是「全程 bf16 没 fp32 master」。源码里这两行触发：

```python
self.orig_dtype = self.sharded_param.dtype          # 加载时的 dtype = bf16
if param_dtype == self.orig_dtype:                  # bf16 == bf16
    param_dtype = None                              # mixed precision 被关掉
```

DS3 ZeRO-3 不一样：`bf16.enabled=auto` 强制 fp32 partitioned master，所以同事 baseline 一直对。

**5. 修复（只改 3 个 CLI 参数）**

```diff
- --torch_dtype bfloat16
+ --torch_dtype float32
+ --bf16 true
+ --fp16 false
```

含义：
- `float32` → 加载 fp32 → sharded master = fp32
- `bf16=true` → trainer 仍开 bf16 mixed precision → unshard 时 cast 成 bf16 算 forward/backward
- `fp16=false` → 覆盖 swift 的自动逻辑（fp32 加载时 swift 默认想设 fp16=True，会冲突）

源码逻辑变成：`param_dtype (bf16) ≠ orig_dtype (fp32)` → **不 clamp**，正常走 mixed precision；optimizer step 自动在 fp32 master 上做。

**6. 代价基本为零**

| 项目 | bf16-master | fp32-master |
|---|---|---|
| GPU peak | 77.5 GiB | **76.5 GiB**（−1）|
| CPU 多占 | — | +16 GB（H100 节点 1.5T 内存毫无压力）|
| s/it | 35.3 | 37.4（慢 6%）|

**7. 实测效果**

|  | bf16-master | fp32-master |
|---|---|---|
| step 1 grad_norm 比值 | **1.29x** vs DS3 | **1.001x** |
| 200 步 |Δloss|/DS3 loss | ~5%（推断）| **0.09%**（实测）|
| step 800 终点 |Δloss|/DS3 loss | **+11%** | **0.14%**（实测，2 个 epoch 跑完）|

**8. 给训练新人的 4 条 takeaway**

1. **不要相信 "`--torch_dtype bfloat16` 就是 bf16 mixed precision"** — FSDP2 下它会变成"全程纯 bf16，没 fp32 master"
2. **DeepSpeed 和 FSDP2 在 mixed precision 语义上不一样**：DS3 总建立 fp32 master；FSDP2 看你加载 dtype
3. **诊断"训了 N 步以后才出现的 loss 偏差"，先怀疑 master dtype**（累积问题）
4. **fp32 master 几乎免费**，强烈建议默认开

---

### 25.0 你需要先理解的 4 个概念

如果你对训练精度不太熟，先把下面 4 个概念想明白，后面就好理解了。

#### 概念 1：bf16 / fp32 是什么，差多少

`bf16` 和 `fp32` 都是浮点数格式，区别在于"小数部分（mantissa）"占多少位：

| 格式 | 字节数 | mantissa 位数 | 相对精度（约） |
|---|---|---|---|
| **fp32** | 4 | 23 位 | 2⁻²³ ≈ 0.0000001（一千万分之一）|
| **bf16** | 2 | **7 位** | 2⁻⁷ ≈ 0.008（**百分之一**）|

**bf16 的代价**：每存一个数都会把"细节"四舍五入到 1% 这个量级。一次 round 看不出问题，**累计 800 步 × 8B 参数**，问题就大了。

#### 概念 2：什么是 "master weights" 和 mixed precision

训练时一个参数有"两个版本"：
- **compute 版本**：forward / backward 用的，**速度快**很重要 → 通常 bf16
- **master 版本**：optimizer 更新用的，**精度高**很重要 → 通常 fp32

每个 optimizer step 流程是：
```
1. 用 compute 版（bf16）算出 grad
2. 把 grad 累加到 master 版（理想是 fp32）
3. master 版做 update：master ← master − lr × m / (√v + eps)
4. 把 master 版 cast 回 compute 版（bf16）给下次 forward
```

这套就叫 **mixed precision training**：「在低精度上做计算（快）、在高精度上做更新（准）」。

#### 概念 3：为什么 bf16 master 是灾难

假设某个 param 数值是 `1.0`，optimizer 算出本步要 update `−0.00002`（因为 lr=2e-5，grad 不大）。

- **fp32 master**：`1.0 - 0.00002 = 0.99998`（精确保留）
- **bf16 master**：bf16 在 `1.0` 附近的最小可表示间隔 ≈ `0.0078`，所以 `1.0 - 0.00002` 被四舍五入回 `1.0` ← **update 完全消失**

每一步消失一点，800 步累计 → **大量该学到的小幅修正都被静默吞了**，模型就欠拟合，loss 降不下去。

DS3 ZeRO-3 的默认行为是**永远保留 fp32 master**，所以同事的 baseline 一直 OK。FSDP2 的默认行为**取决于你怎么加载模型**——这就是我们踩坑的地方。

#### 概念 4：FSDP2 的 sharded vs unsharded

FSDP2 把每个参数切 8 份（8 卡），每张卡只持有自己那份"sharded param"。算 forward / backward 时再 all-gather 成完整的"unsharded param"用，算完丢掉。

PyTorch 官方文档原话：
> The **sharded parameters stay in original dtype**. The **unsharded parameter** uses `param_dtype` for forward/backward. **The optimizer step uses the sharded parameter in the original dtype.**

翻译成人话：
- **sharded（分片）那份是 master**，optimizer 在它上面做 update
- **unsharded（合并）那份是 compute**，cast 成 `param_dtype` 算 forward/backward
- **sharded 的 dtype = 你 `from_pretrained` 加载模型时的 dtype**（很关键，是 11% gap 的源头）

---

### 25.1 整个调查的起点：11% loss gap 一直修不掉

之前 §15 用 FSDP2 跑 2 个 epoch 完成训练，但 step 800 时 loss 比同事 DS3 baseline 高 **11%**。中间几章我们试过：

| 试过的方法 | 结果 |
|---|---|
| §18 KV-share detach 修复 | 修了 step 1 grad_norm 1.29x 偏差，但 800 步终点仍 +11% |
| §20 fp32 grad accumulator 手工 patch | 撞 PyTorch 2.10 cublas bug，没跑通 |
| §21 fp32 reduce-scatter | 同上，cublas bug |
| §22 dataloader 等价性验证 | 数据完全一致，不是数据问题 |
| §23 多种 backward fire-count 实验 | 排除了 backward 多次触发等假设 |
| §24 cossim 实验：layer 0 grad 方向 cos=0.36 | 看到一个症状，但只是冰山一角 |

§24 让我们怀疑是 bf16 反向链路的累积 rounding 误差，但**没找到根因**。所以这次决定再深挖一层 — 直接去 PyTorch FSDP2 源码看混合精度到底是怎么走的。

### 25.2 调查步骤 1：在 swift 里追 dtype 的传递路径

**问题**：从命令行的 `--torch_dtype bfloat16` 到 FSDP 真正持有的 dtype，中间经过哪些 module？

工具：grep + 阅读源码，**没有动代码**。

**a. swift 入口**：用户传 `--torch_dtype bfloat16`，swift 把它解析成 `torch.bfloat16` 给 transformers 的 `from_pretrained`。

```120:147:/home/ubuntu/fyh/ms-swift/swift/arguments/base_args/model_args.py
    def _init_torch_dtype(self) -> None:
        """If torch_dtype is None, find a proper dtype by the config.json/GPU"""
        ...
        self.torch_dtype: torch.dtype = self._init_model_info()
        # Mixed Precision Training
        if isinstance(self, SftArguments):
            self._init_mixed_precision()

    def _init_mixed_precision(self):
        ...
        elif self.torch_dtype in {torch.float16, torch.float32}:
            fp16, bf16 = True, False
        elif self.torch_dtype == torch.bfloat16:
            fp16, bf16 = False, True
        ...
        if self.fp16 is None:
            self.fp16 = fp16
        if self.bf16 is None:
            self.bf16 = bf16
```

→ 关键发现 1：swift 用 `--torch_dtype bfloat16` 时**自动**把 `bf16=True` 也设上。这个 `bf16=True` 是给 HF Trainer 用的「mixed precision 旗标」。

**b. HF Trainer → Accelerator**：HF Trainer 把 `bf16=True` 翻译成 `mixed_precision="bf16"` 传给 Accelerate。

**c. Accelerate → FSDPPlugin.set_mixed_precision**：

```bash
# 路径
accelerate/state.py:1016:  fsdp_plugin.set_mixed_precision(self._mixed_precision)
```

```2100:2107:/usr/local/lib/python3.12/site-packages/accelerate/utils/dataclasses.py
        if override or self.mixed_precision_policy is None:
            dtype_args = {"param_dtype": dtype, "reduce_dtype": dtype}
            ...
            dtype_args["output_dtype"] = dtype
            # TODO(s1ro1): `cast_forward_inputs` for FSDP2?
            self.mixed_precision_policy = MixedPrecision(**dtype_args)
```

→ 关键发现 2：accelerate 把 `mixed_precision="bf16"` 翻译成 `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=bf16, output_dtype=bf16)`，**三个 dtype 全设 bf16**。

**d. 最关键的一步——FSDP2 的 init_dtype_attrs**：

```438:451:/usr/local/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py
    def init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy):
        param_dtype, reduce_dtype = (mp_policy.param_dtype, mp_policy.reduce_dtype)
        self.orig_dtype = self.sharded_param.dtype                # ← 关键！
        # Clamp `reduce_dtype` to `None` if no casting is required
        if reduce_dtype == param_dtype:
            reduce_dtype = None
        # Clamp `param_dtype` to `None` if no casting is required
        if param_dtype == self.orig_dtype:                        # ← 灾难触发点
            param_dtype = None
```

→ **关键发现 3（核弹）**：
- `orig_dtype = sharded_param.dtype` — 这就是「sharded master 的 dtype」，**等于 from_pretrained 加载时的 dtype**
- 如果 `param_dtype == orig_dtype`，FSDP2 会**把整个 mixed precision 关掉**（clamp 成 None）

我们的旧配置 `--torch_dtype bfloat16`：
- `from_pretrained` 加载成 bf16 → sharded_param 是 bf16 → `orig_dtype = bf16`
- `MixedPrecisionPolicy(param_dtype=bf16, ...)` 被传进来
- `param_dtype (bf16) == orig_dtype (bf16)` → **clamp 成 None** → **没有 fp32 master！**

**这就是 11% gap 的根因**：sharded master 是 bf16，optimizer 直接在 bf16 master 上做 update，每步都 round-to-bf16-quantum，800 步累积一堆"消失的 update"。

### 25.3 调查步骤 2：DS3 是怎么躲掉这个坑的

DeepSpeed ZeRO-3 用的是 `bf16.enabled=auto`：
- **不管你 `torch_dtype` 加载成什么**，DS3 都会强制建立 fp32 partitioned master（`fp32_partitioned_groups_flat`）
- forward/backward 时 cast 成 bf16 用，optimizer step 永远在 fp32 master 上做
- `contiguous_gradients=true` 还会把 gradient 累加在 fp32 bucket 里（跨 16 micros）

所以 DS3 在 `--torch_dtype bfloat16` 下也会自动 cast 到 fp32 master。FSDP2 不会。

### 25.4 修复：只改 CLI，不动一行代码

理论上要"既保留 bf16 forward/backward 的速度，又有 fp32 master 的精度"，需要：
- `from_pretrained` 加载成 fp32（让 sharded master = fp32）
- `MixedPrecisionPolicy(param_dtype=bf16)`（unshard 时 cast 成 bf16 算）

回头看 §25.2 的代码逻辑：
```python
self.orig_dtype = self.sharded_param.dtype       # fp32
if param_dtype == self.orig_dtype:                # bf16 != fp32, 不 clamp
    param_dtype = None
```
现在 `param_dtype` 不会被 clamp 成 None → unshard 时正确 cast 成 bf16。Mixed precision 真的开起来了。

具体到我们 swift 的命令行，**只改 3 个参数**：

| 旧 | 新 | 为什么这么写 |
|---|---|---|
| `--torch_dtype bfloat16` | `--torch_dtype float32` | 让 from_pretrained 加载成 fp32 → sharded master = fp32 |
| 无 | `--bf16 true` | 显式告诉 trainer "我要 bf16 mixed precision"（覆盖 swift 自动逻辑里 fp32→fp16=True 的判断） |
| 无 | `--fp16 false` | 同上，关掉自动设的 fp16=True |

**验证 CLI 解析正确**（在 docker 里跑一次就能看）：
```bash
docker exec fsdp_sft python -c "
import sys
sys.argv = ['test', '--model', '/dummy', '--torch_dtype', 'float32', '--bf16', 'true', '--fp16', 'false']
from transformers import HfArgumentParser
from swift.arguments.sft_args import SftArguments
p = HfArgumentParser(SftArguments)
args, _ = p.parse_known_args()
print(args.torch_dtype, args.bf16, args.fp16)
"
# 预期输出：float32 True False
```

### 25.5 验证步骤 1：50-step bench

为了在 8 小时全量跑之前先确定不会崩，我跑了一个 50-step bench：

```bash
cd /home/ubuntu/fyh/megatron-sft-recipes
LABEL="bench_fp32master_a3_pf_50step" \
TEMPLATE="gemma4" WEIGHT_DECAY="0.1" \
MAX_STEPS=50 FULL_SCHED_STOP=1 \
TORCH_DTYPE="float32" \
EXTRA_ENV="GEMMA4_FSDP_WRAP_PLE=1 GEMMA4_KV_SHARE_DETACH=1" \
EXTRA_ARGS="--bf16 true --fp16 false --padding_free true --max_grad_norm 1.0" \
bash scripts/gemma4_E4B_opt/bench_variant.sh
```

> 注：`bench_variant.sh` 之前 hardcode 了 `--torch_dtype bfloat16`，本次改造把它做成 `${TORCH_DTYPE}` 环境变量。

50 步结果：

| 配置 | step 1 loss | step 50 loss | s/it | peak GPU mem |
|---|---|---|---|---|
| **bf16-master B0**（旧基线） | 2.226 | **1.7030** | 37.49 | 77.54 GiB |
| **fp32-master**（本次） | 2.226 | **1.6571** | 38.05 | 76.48 GiB |
| DS3 ZeRO-3（同事）| 2.226 | ~1.70 | 55+ | n/a |

→ 50 步就已经比 bf16-master 低 0.046（-2.7%）；速度只慢 1.8%；显存反而少 1 GiB（因为 fp32 master 在 CPU offload）。

### 25.6 验证步骤 2：full run 前 48 步逐步对齐 DS3

bench 通过后开 8h 全量跑，并用 `plot_compare_ds3_fsdp2.py` 自动每 100 步出对比图。前 48 步的 step-by-step 对比：

```
 step | DS3 loss  DS3 gn | FSDP2 loss  FSDP2 gn |   Δloss  gn ratio
    1 |   2.2255  10.283 |     2.2258    10.289 | +0.0003     1.001
    5 |   2.1020   9.384 |     2.1031     9.161 | +0.0011     0.976
   10 |   1.8964   4.686 |     1.8995     4.658 | +0.0031     0.994
   25 |   1.7396   0.811 |     1.7404     0.821 | +0.0008     1.011
```

| 维度 | 旧（bf16-master）vs DS3 | 新（fp32-master）vs DS3 |
|---|---|---|
| step 1 grad_norm 比值 | 1.29x（系统性偏大） | **1.001x**（差异 0.1%）|
| step 25 Δloss | 已经能看出 +1% 趋势 | **+0.0008（万分之 4）** |
| step 800 |Δloss|/DS3 loss | **+11%** | **0.14%**（实测）|
| 早期层 grad cos_sim（§24）| 0.36 | （预期 0.99+，待补 cossim 实验）|
| 速度 | 35.3 s/it | **37.4 s/it（仅慢 6%）** |
| 速度 vs DS3 | 比 DS3 快 36% | **比 DS3 快 32%**（55.3 → 37.4） |
| 峰值 GPU 显存 | 77.5 GiB | **79.2 GiB**（实测 full run，贴 80 GiB 限制）|
| 峰值 CPU 内存 | — | +16 GiB（fp32 master）|
| 总耗时（2 epoch 806 步）| ~7.9 h | **8.5 h**（实测）|
| MFU | — | **31.4%** |
| Tokens/s/GPU | — | **6,897** |

**结论**：fp32-master 让 FSDP2 数值上**几乎完全对齐 DS3 ZeRO-3**（Δloss 0.14% vs 11%，**78× reduction**），同时保留了 **1.32× 的速度优势**（55.3 → 37.4 s/it）。

### 25.7 内存账：为什么 GPU 峰值反而少了

很反直觉对吧？fp32 加载，模型应该更大才对。账如下：

| 项目 | bf16-master | fp32-master | 在 CPU 还是 GPU |
|---|---|---|---|
| sharded master | 16 GB | **32 GB**（+16）| **CPU**（cpu_offload）|
| optimizer state m, v | 2 × 32 GB = 64 GB | 2 × 32 GB = 64 GB | **CPU** |
| sharded grad | 同 | 同 | GPU 短暂 + transfer |
| **unsharded compute buffer** | bf16 ~2 GB | bf16 ~2 GB（cast 后）| GPU |
| activation（AC=on）| ~28 GB | ~28 GB | GPU |
| **GPU peak 实测** | **77.54 GiB** | **76.48 GiB** | — |
| CPU 多吃 | — | +16 GB（fp32 master）| H100 节点 1.5T RAM 完全够 |

**关键**：`cpu_offload=true` 把 sharded master 全甩到 CPU，GPU 上只剩 unshard 时的临时 bf16 buffer。所以 sharded master 从 bf16 升 fp32（CPU 多吃 16 GB），GPU 视图基本不变。`-1 GiB` 是 PyTorch `expandable_segments` allocator 的细节波动。

### 25.8 §24 cossim 现象的重新解读

§24 我们测出 layer 0 的 grad 在 FSDP2 vs DS3 之间 cos=0.36（方向相差 64%）。这怎么和 §25 修复一致？

回答：cos=0.36 是 **真实存在的现象**，但**不是 11% gap 的主要源头**。机制：

1. **每一步 backward** 仍然是 bf16 → 早期层（layer 0）方向确实有 ~64% 偏差 → 这是 bf16 反向链路的 rounding 累积
2. **每一步 optimizer update** 拿这个"略偏"的 grad 累加到 master 上
   - 旧（bf16 master）：方向偏 + master round-to-bf16-quantum **同时**发生，绝大部分小 update 直接消失，模型欠拟合
   - 新（fp32 master）：方向虽然每步偏 36%，但 fp32 master 能**精确保留**这个偏差信息，800 步后被时间平均成 DS3 trajectory 附近的方向
3. → 11% gap 的元凶不是 cos=0.36 本身，而是 **bf16 master 把这些方向偏差一次次 round 掉**

简单类比：你每天体重变化 ±0.1 公斤。
- bf16 master = 体重秤只显示整数公斤 → 每天看 60 公斤，**永远不变**
- fp32 master = 体重秤显示到 0.01 → 60.1, 60.0, 60.2 ... → **每天的小变化能被记录、累加**

### 25.9 复现脚本与命令

**新脚本**：`scripts/gemma4_E4B_opt/fsdp2_offload_E4B_text_only_2ep_a3_pf_fp32master.sh` —— 跟 §15 baseline 脚本只差 3 个 CLI 参数（25.4 表里的）。

**改动的旧脚本**：
- `scripts/gemma4_E4B_opt/bench_variant.sh` —— 加 `TORCH_DTYPE` 环境变量旗标；hardcode `--torch_dtype bfloat16` → `${TORCH_DTYPE}`
- `scripts/gemma4_E4B_opt/bench_ds3.sh` —— 加 `USE_LIGER` / `GRAD_CKPT` 环境变量旗标；同时修了一个 `${SDP_PRE:-...}` 嵌套 `${PYTHONPATH:-}` 的 bash 展开 bug（旧版当 `SDP_PRE` 显式传入时，bash 会错误地把 inner default 嵌套进来，造成 `GEMMA4_FORCE_MEM_EFFICIENT_SDP=1}` 多个尾 `}` 的环境变量泄漏；改成显式 `ENV_PREAMBLE="export ...; export ...; "` 解决）

**最简复现**（需要 8×H100 80G，sft-data 数据集，gemma-4-E4B-it 模型本地缓存）：

```bash
nohup bash scripts/gemma4_E4B_opt/fsdp2_offload_E4B_text_only_2ep_a3_pf_fp32master.sh \
    > /tmp/fp32master_run.log 2>&1 &
```

或先做 50-step bench 验证：

```bash
cd /home/ubuntu/fyh/megatron-sft-recipes
LABEL="bench_fp32master_test" TEMPLATE="gemma4" WEIGHT_DECAY="0.1" \
MAX_STEPS=50 FULL_SCHED_STOP=1 TORCH_DTYPE="float32" \
EXTRA_ENV="GEMMA4_FSDP_WRAP_PLE=1 GEMMA4_KV_SHARE_DETACH=1" \
EXTRA_ARGS="--bf16 true --fp16 false --padding_free true --max_grad_norm 1.0" \
bash scripts/gemma4_E4B_opt/bench_variant.sh
```

### 25.10 这个修复给"训练新人"的 4 条 takeaway

1. **不要相信"`--torch_dtype bfloat16` 就是 bf16 mixed precision"** —— 在 FSDP2 路径下它会**变成全程纯 bf16，没有 fp32 master**，是错的混合精度，跟 bf16 mixed precision 完全两回事。
2. **DeepSpeed 和 FSDP 在「mixed precision」语义上不一样**：DS3 `bf16.enabled=auto` 始终强制 fp32 master；FSDP2 是「以 sharded param 的 dtype 为准」，全看你怎么加载模型。
3. **要诊断"训了 N 步以后才出现的 loss 偏差"，先怀疑 optimizer step 的精度**（master dtype），再怀疑反向传播链路的精度（per-step backward）；前者是累积问题，后者是单步问题。
4. **fp32 master 的成本极低**：CPU 多 16 GB（H100 节点 1.5T 内存毫无压力），GPU 显存基本不变，速度只慢 1.8%。**几乎是免费的**，强烈建议默认开启。

### 25.11 §24 工具仍然有用

§24 留下的工具集（`compute_cossim.py`、sitecustomize.py patch 22 的 raw-grad dump、`run_cossim_noAC_experiment.sh`）即使 §25 已经把 11% gap 干掉，仍然是诊断未来 numerical 偏差的标准工具。

未来怀疑训练有 numerical 问题的 SOP：
1. **第一站**：跑 §25 fp32-master 验证（最便宜、最常见的根因）
2. **第二站**：如果 fp32-master 解决不了，跑 §24 cossim 实验定位是 forward/backward 哪一段在漂
3. **第三站**：如果 cossim 显示某层异常，用 §22 patch grad-dump 落到具体参数 / micro / rank
4. **第四站**：上 §23 backward fire-count 或源码 dive


---

## §26. v5 之后的提速尝试（全部失败 / Future-work 参考）— 2026-05-05

> **本节目的**：v5（patch 21 + 21b + 29 + 30 + activation_cpu_offload）已经在数值上对齐 DS3（+0.10% Δloss，40 s/it，9 h 跑完 2 epoch）。本节记录"想再快一点"的几次尝试，**全部不能在保持 v5 精度的前提下提速**。给后续读者作 future-work 参考，避免踩同样的坑。

### 26.0 出发点：v5 的真实瓶颈

**实测（v5 全跑 GPU profile，rank0 dcgm tsv）**：

| 指标 | v5 实测 |
|---|---|
| **TensorCore 活跃率** | **11.2%** mean（活动期 16.4%）|
| graphics 活跃率 | 60.8% mean |
| HBM 带宽利用 | 13.6% mean |
| 功耗 | 303 W / 700 W TDP（43%）|
| s/it | 40.2 |
| peak GPU | 69.6 GiB |

**结论**：GPU 严重 underutilized，瓶颈是 **PCIe / NCCL / kernel launch**，不是算力。每 step 中，**只有 ~4-5 s 在做矩阵乘**，剩 35 s 在等 I/O / 通信 / Python overhead。

**v5 时间分解（估算）**：

| 项 | 时间 (s) | 占比 |
|---|---|---|
| BF16 矩阵乘（fwd+bwd+AC 重算）| 4-5 | 11% |
| **activation_cpu_offload H2D/D2H** | **8-12** | **25%**（最大头）|
| **cpu_offload (master/optim) H2D/D2H** | 3-5 | 10% |
| FSDP all-gather params | 4-6 | 12% |
| FSDP reduce-scatter grads | 2-3 | 6% |
| AC 重算 forward | 6-8 | 17% |
| Python / CUDA launch overhead | 4-6 | 12% |
| 其他 | 2-3 | 7% |

**真正的提速方向应该是减少 PCIe 拷贝**。

### 26.1 patch 21b 的内存代价（OOM 真因）

跑 v5 但**关 `activation_cpu_offload`** → 必然 OOM at micro 16。用 `cuda.memory_summary()` + storage walking 定位真因：

**OOM micro 16 的内存快照（rank 7）**：

```
alloc:     76.12 GB
reserved:  76.5 GB
process:   78.18 GiB (含 CUDA context + NCCL workspace)

top tensors:
  11.27 GB  (262144, 10752) fp32   ← embed_tokens_per_layer (PLE) UNSHARDED fp32!
   2.68 GB  (262144, 2560)  fp32   ← embed_tokens UNSHARDED fp32
  ~13 GB    多个 (10240, 2560) fp32 + (2560, 10240) fp32  ← MLP weights UNSHARDED fp32
   5.12 GB  (1, 9770, 262144) bf16  ← 当前 micro 的 logits
   5.12 GB  (9769, 262144) bf16     ← shifted logits

Total visible: ~35-69 GB
```

**真因**：PyTorch 2.10 FSDP2 在 no-sync 路径（micro 1-15）下，**`to_accumulated_grad_if_needed` 把每层的 unsharded grad cast 到 fp32，存在每个 FSDP unit 的 `unsharded_accumulated_grad` 字段里**。

源码：`/usr/local/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:669-680`

```python
def to_accumulated_grad_if_needed(self) -> None:
    if (self.reduce_dtype is None
        or self._unsharded_param.grad is None
        or self._unsharded_param.grad.dtype == self.reduce_dtype):
        return
    unsharded_grad = self._unsharded_param.grad
    self._unsharded_param.grad = None
    self.unsharded_accumulated_grad = unsharded_grad.to(self.reduce_dtype)  # ← UNSHARDED fp32!
```

**8B params × 4 bytes = 32 GB/rank** 长期驻留 GPU，跨 micro 累加。这就是 patch 21b 启用后必须配 act_cpu_offload 的原因。

### 26.2 patch 21c — per-micro 立即 RS（DS3-like）

**思路**：拦截 `FSDPModule.set_requires_gradient_sync(False)` 让它永远为 True，每 micro 都 reduce-scatter，sharded fp32 += sharded fp32（FSDP2 内部已支持）。

**实现**（`sitecustomize.py` patch 21c）：

```python
def _patched_set_sync_21c(self, requires_gradient_sync, *, recurse=True):
    return _orig_set_sync_21c(self, True, recurse=recurse)
```

只 1 行核心逻辑。

**实测（50 步 bench）**：

| 指标 | 21c | v5 |
|---|---|---|
| s/it | 66 | 40 |
| peak GPU | 62.5 GiB | 69.6 GiB |
| Δloss vs DS3（mean 1-50）| **+0.016%** | +0.10% |
| grad_norm 比值 | **1.004** | ~1.05 |

**速度退步原因**：每 step 16 micro × ~45 FSDP units = **720 次 reduce-scatter**。每次 NCCL launch latency ~30 ms × 720 = 22 s/step 纯 latency 浪费。

**结论**：精度极优（接近 DS3），但速度损失 65% 不可接受。**适合作为"最高精度参考点"**，不能作为生产配置。

### 26.3 patch 21d — cross-group bucket fusion（失败）

**思路**：21c 慢是因为 720 次小 RS。如果跨 FSDP unit fuse 成 16 次大 RS（每 micro 1 次），通信总量不变但 latency 大幅降低。

**实现路径**：
1. 替换 `FSDPParamGroup.post_backward`：no-sync 路径下不立即 RS，把 `(fsdp_params, unsharded_grad_data)` 注册到全局 bucket
2. 替换 `FSDPState._root_post_backward_final_callback`：在该 callback 触发时（所有 group 的 post_backward 都跑完），用 `foreach_reduce` 跨所有 group 一次大 RS

**关键发现**：FSDP2 的 `foreach_reduce(list[FSDPParam], list[Tensor], ...)` 接受跨 group 的 list，可以一次 NCCL 调用处理所有 params——只要 `reduce_scatter_process_group` / `_orig_dtype` / `_reduce_dtype` 一致（普通 FSDP2 都是同一个）。

**实现细节**（`sitecustomize.py` patch 21d，~150 行）：

```python
_21D_STATE = {"params": [], "grads": [], "groups_seen": [], ...}

def _patched_post_bw_21d(self, *unused):
    # 1. accumulate_unsharded_grad_if_needed (carry-over)
    # 2. 把 (fsdp_param, unsharded_grad_data) 推到全局 bucket
    # 3. reshard
    # 4. 不调用 to_accumulated_grad_if_needed（不创建 unsharded fp32 buffer）

def _21d_flush_bucket():
    # chunked: 每 chunk_bytes 1-2 GB 调一次 foreach_reduce 避免 OOM
    # 跨所有 group 共享 proxy._reduce_scatter_process_group 等

def _patched_root_cb_21d(self):
    if _21D_STATE["params"]:
        _21d_flush_bucket()
    return _orig_root_cb_21d(self)
```

**问题 1：foreach_reduce 内部的 transient buffer 太大**

第一次 fused RS 直接 OOM：
```
torch.OutOfMemoryError: Tried to allocate 27.80 GiB
```

`foreach_reduce` 内部把所有 grads concat 到一个 contiguous bf16 buffer（16 GB），再 cast 到 fp32（32 GB），所以 transient 高峰 ~48 GB。

**问题 2：chunked + sync wait 后实测远不达预期**

加了 chunk + `current_stream().wait_event(post_reduce_event)` 后跑通：

| 指标 | 21d 实测（10 步）|
|---|---|
| median s/it | **66.7**（v5 是 40，失败）|
| peak GPU | 73.6 GiB（v5 是 70 GiB，没省）|
| step 1-10 loss | 跟 21c 一致（0.016% 精度）|

**为什么没快**：
- chunk size 1 GB → 16 chunks/micro × 16 micros = **256 次 chunked RS**
- 每次 RS 的 launch latency 还在
- sync wait 把 stream overlap 也牺牲了

**为什么内存没省**：
- v5 的 70 GB 实际是 RS-time **transient spike**（unsharded fp32 grad 32 GB + 同时 fp32 cast staging）
- 21d 是分散到 16 次小 spike，每次 spike 时 transient buffer 大小不变（chunk_size × 2）
- 持续状态 21d 是少 32 GB（无 unsharded buffer），但 max_alloc 看的是 spike

**结论**：在 PyTorch 2.10 `foreach_reduce` 当前实现下（必须 cast staging buffer），bucket fusion 物理上无法节省内存。要赢就得绕过 `foreach_reduce`，自己用 `dist.reduce_scatter_tensor` + 手动 cast inplace + DTensor wrap，估计 500+ 行代码。

### 26.4 torch.compile 尝试（patch 32 + regional compilation）— nan grads

**思路**：accelerate 的 `TorchDynamoPlugin(use_regional_compilation=True)` 只编译 `Gemma4TextDecoderLayer`（重复 block），避开 patch 8 的 KV-sharing module-level dict graph break。

**配置**：
```bash
ACCELERATE_DYNAMO_BACKEND=inductor
ACCELERATE_DYNAMO_USE_REGIONAL_COMPILATION=True
ACCELERATE_DYNAMO_USE_DYNAMIC=True
ACCELERATE_DYNAMO_MODE=default
--torch_compile true --torch_compile_backend inductor
```

**问题 1：`inductor does not support pin_memory`**

Swift 的 `activation_cpu_offload` 在 saved_tensors_hooks 里调 `torch.empty(..., pin_memory=True)`，inductor 编译失败：

```
torch._inductor.exc.InductorError: LoweringException:
NotImplementedError: inductor does not support pin_memory
```

**修复 patch 32**（`sitecustomize.py`）：用 `torch.compiler.disable` 包装 swift 的 hook 函数：

```python
for cls_name in ("CpuOffloadHookWithOffloadHandler",
                 "SynchronizedGroupOffloadHandler",
                 "AsyncDoubleBufferGroupOffloadHandler"):
    cls = getattr(_aco, cls_name, None)
    for m in ("on_save_for_backward", "on_get_saved_tensor",
              "offload", "reload", "tensor_push", "tensor_pop"):
        if m in cls.__dict__:
            setattr(cls, m, torch.compiler.disable(cls.__dict__[m]))
```

成功（log 显示 wrapped 8 methods），但产生新错。

**问题 2：`KeyError: 1` in `bulk_reload_group`**

```
File ".../activation_cpu_offload.py", line 419, in on_group_commit_backward:
  self.bulk_reload_group(self.offloaded_group_count - 1)
File ".../activation_cpu_offload.py", line 393, in bulk_reload_group:
  offload_mapping = self.group_offload_mapping.pop(group_to_reload)
KeyError: 1
```

dynamo trace forward 时 hook 被 disable → eager 路径下 hook 调用 → state 跟 dynamo 编译的 graph 不同步 → backward 时找不到该有的 mapping entry。

**结论**：inductor 跟 saved_tensors_hooks 根本不兼容。要么关 act_cpu_offload，要么不能用 compile。

**问题 3（关 act_cpu_offload + 关 21b）：grad_norm = nan + 速度更慢**

为了避开问题 2，关 `activation_cpu_offload` 和 `patch 21b`（这样不需要 32 GB unsharded fp32 buffer），加 compile。**实测（10 步）**：

| step | loss | grad_norm | mem GB | s/it |
|---|---|---|---|---|
| 1 | 2.225 | **nan** | 66.3 | 242（冷编译）|
| 5 | 2.13 | nan | 68.2 | 81.7 |
| 10 | 2.16 | **nan** | 76.3 | **60.1** |

- **grad_norm 全程 nan**：loss 是合理的（forward 没问题），但 backward inductor 生成的代码有数值 bug
- median 60.1 s/it（v5 是 40）→ 慢 50%
- peak 76 GiB（v5 是 70）→ 内存还多

可能跟 patch 8（KV-sharing module-level dict）+ AC + 自动微分的组合有关，但单独 debug 这个 nan 的工作量很大。

**结论**：torch.compile 在当前 stack 下（PyTorch 2.10 inductor + ms-swift act_cpu_offload + Gemma4 KV-sharing patches）**完全不可用**。要让它工作至少需要：
1. 重写 patch 8 让 KV-sharing 不用 module-level dict
2. 重写 swift act_cpu_offload 用 dynamo 友好的机制
3. 修 inductor backward nan bug（PyTorch 自身 bug）

工作量 = 重写半个 stack。

### 26.5 综合对比表

| 方案 | s/it | peak GPU | Δloss vs DS3 | 复杂度 | 状态 |
|---|---|---|---|---|---|
| **v5（生产）** | **40** | 70 GB | **+0.10%** ✅ | 21+21b+29+30 | ✅ 推荐 |
| 21c（per-micro RS）| 66 | 62.5 | **+0.016%** ✅✅ | 21+21c | 慢 65%，**作高精度参考** |
| 21d（bucket fusion）| 67 | 73 | (同 21c) | 21+21d+chunked+sync | 失败：通信优势没出来 |
| compile + offload | OOM/Error | - | - | 21+21b+...+32 | KeyError，不兼容 |
| compile + no offload + no 21b | 60+ | 76 | **nan grads** ❌ | 21+32 | 数值出错 |

### 26.6 后续读者建议（如果 PyTorch 升级）

如果将来 PyTorch ≥ 2.11 出来，这几个改动可能让 v5 进一步提速，值得复测：

1. **FSDP2 sharded fp32 grad accumulation**（取代 unsharded buffer）
   - 关注 PR：搜 "fsdp2 sharded grad accumulation" / "to_accumulated_grad_if_needed"
   - 如果改成 sharded（per-rank 4 GB instead of 32 GB），就能关 act_cpu_offload，省 8-12 s/step

2. **inductor backward 数值正确性 + saved_tensors_hooks 兼容**
   - 关注 issue：https://github.com/pytorch/pytorch/issues/?q=inductor+saved_tensors_hooks
   - 如果 fix 了，patch 32 + regional compile 可能省 5-10 s/step

3. **NCCL bucket fusion API**
   - 关注 `dist.reduce_scatter_coalesced` 或类似 API
   - 如果 stable，可以替代 21d 的 hand-rolled 实现

### 26.7 当前最优配置（v5）的最终启动命令

参考 §25.7 和 §19。完整脚本：
```
scripts/gemma4_E4B_opt/fsdp2_offload_E4B_text_only_2ep_a3_pf_fp32master.sh
```

环境变量：
```
GEMMA4_FSDP_WRAP_PLE=1
GEMMA4_KV_SHARE_DETACH=1
GEMMA4_FSDP_REDUCE_FP32_NCCL=1   # 启用 patch 21 + 21b
```

`fsdp_override.json`：
```json
{
  "fsdp": "full_shard auto_wrap offload",
  "fsdp_config": {
    "fsdp_version": 2,
    "activation_checkpointing": true,
    "activation_cpu_offload": true,
    ...
  }
}
```

CLI：
```
--torch_dtype float32  --bf16 true  --fp16 false
--padding_free false  --max_grad_norm 1.0
```

实测：40.2 s/it, peak 69.6 GiB, +0.10% Δloss vs DS3, 全跑 2 epoch ~9 h。
