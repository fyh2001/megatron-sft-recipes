# Gemma-4-E4B-it FSDP2 — text-only SFT 实测日志

> **目标**：参考 [`gemma4_phase_delta_summary.md`](gemma4_phase_delta_summary.md) P1 alt-offload 策略，将 GBS=64 / seq_len=16384 / truncation=right 的核心约束平移到 `google/gemma-4-E4B-it` 上，进行 **text-only SFT**。
>
> **最终选定的配置**：FSDP2 native（无 offload）+ GBS=128 + 2 epoch（用户在比较同事 DS 命令后改定，详见 §3.0）。
>
> **数据集**：`/home/ubuntu/fyh/megatron-sft-recipes/sft-data/SFT_0424_2.jsonl`（51 557 条）
>
> **要求**：所有性能指标必须为真实测试数据，不允许任何估算。
>
> **本文档自更新**：每次跑出新的状态、报错、解决方案，都直接落到这里。

---

## 0. 最终配置基线

| 维度 | 值 | 备注 |
|---|---|---|
| 模型 | `google/gemma-4-E4B-it` | 4B effective param dense + PLE，原生支持 text/image/audio |
| 任务 | text-only SFT | freeze ViT、aligner、audio tower |
| 数据集 | `sft-data/SFT_0424_2.jsonl` | 51 557 条 messages 格式样本 |
| Sequence length | 16384 | — |
| Truncation | **right** | 长样本截断到 16k |
| 引擎 | **FSDP2 native（无 offload）** | E4B 8B 模型 H100 80G 完全装得下，offload 是 PCIe 浪费 |
| MBS / GAS / SP | 1 / 16 / 1 | — |
| DP / GBS | 8 / **128** | NPROC / SP；GBS = MBS × DP × GAS |
| Epochs | **2** | — |
| LR / Warmup | **2e-5** / **0.05** | — |
| LR scheduler | cosine | swift 默认 |
| Save strategy | **epoch** | save_only_model=true，save_total_limit=3 |
| AC | on（FSDP2 自管 + `--gradient_checkpointing true`） | — |
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

### 6.2 根因（KV-sharing × FSDP2 activation_checkpointing 交互失败）

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

**FSDP2 `activation_checkpointing=true` 把每个 `Gemma4TextDecoderLayer` 用 `torch.distributed.algorithms._checkpoint.checkpoint_wrapper.CheckpointWrapper` 包裹，CheckpointWrapper.forward 调用 `torch.utils.checkpoint.checkpoint(layer, *args, **kwargs, use_reentrant=False)`**。`use_reentrant=False` 模式下，dict 副作用对 forward 应当是透明的，但实测**不**透明 —— 证据是 `shared_kv_states[22]` 在 layer 24 forward 时 KeyError，说明 layer 22 的 forward 写入没有传播到外层 dict 引用。

**26B-A4B-it 不受影响**：`num_kv_shared_layers=0`，整个 KV-sharing 代码路径根本没被命中。所以原 P1 alt-offload 在 26B-A4B 上的 `activation_checkpointing=true` 工作正常。

### 6.3 修复（关掉 AC）

E4B 只有 ~8B 参数，FSDP2 ZeRO-3 + 8 卡分片 + activations 全保留下，单卡显存预算：

| 项 | 估算 |
|---|---:|
| param shard (bf16) | ~2 GiB |
| grad shard (bf16) | ~2 GiB |
| opt state shard (Adam fp32 master + 2 moments) | ~12 GiB |
| activations (无 AC，seq=16384, MBS=1, 42 层) | ~17 GiB |
| 其他（PLE 表 lookup buffer、NCCL bucket、scratch） | ~3 GiB |
| **总计** | **~36 GiB / GPU** |

远低于 H100 80 GiB，**关掉 AC 完全没问题**，还省 33% recompute FLOPs。修改：

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







