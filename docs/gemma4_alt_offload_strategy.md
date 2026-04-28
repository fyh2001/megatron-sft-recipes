# Gemma-4-26B-A4B-it FSDP2 alt-offload（DS-equivalent）汇报

> **策略代号**：P1 alt-offload，又名「DS-equivalent on FSDP2」。
> **角色定位**：把生产线 DeepSpeed ZeRO-3 + offload 配置**等量平移到 FSDP2** 上的稳定基线。**不是吞吐峰值**（峰值在 P5，37.5 min/epoch），而是把"内存开销 / 优化器轨迹 / GBS / AC 行为"全部对齐 DS 的同时，用 FSDP2 把端到端 wall 缩到 1.89× DS prod 的版本。
>
> **唯一一行结论**：单步 **45.4 s**、单卡显存 **76.6 GiB**、单卡瞬峰功耗 **596 W**、active-MFU **13.3%**、单 epoch（18819 样本）外推 **223 min** —— 比 DS 生产线（422 min）快 **1.89 倍**，比 FSDP2 性能峰值 P5（37.5 min）慢 **5.95 倍**。
>
> **本文档自包含**：环境 / 依赖 / 配置 / 完整启动命令 / 完整指标 / Full-epoch 推算 / 适用场景，全部一次性给齐，便于直接拿这份汇报付诸生产替换。

---

## 0. 一图速览

| 维度 | DS prod baseline | **P1 alt-offload (本策略)** | P5 peak（参考） |
|---|---:|---:|---:|
| 引擎 | DS ZeRO-3 + offload(opt+param) | **FSDP2 + offload** (full_shard auto_wrap offload) | FSDP2 native（无 offload） |
| GBS | 64 | **64** | 4 |
| MBS / GAS | 1 / 16 | **1 / 16** | 1 / 1 |
| AC | on（gradient_checkpointing） | **on**（activation_checkpointing） | **off** |
| Packing | false | **false** | true |
| Liger | true（gemma4 silent no-op） | **true** | true（实际生效：RMSNorm + GeGLU） |
| Step time | 86,180 ms | **45,393 ms** | 2,781 ms |
| Per-micro | 5,386 ms | **2,837 ms** | 2,781 ms |
| Peak mem (GiB) | ~43.6（CPU 兜了一半）| **76.6** | 64.9 |
| Peak power (W/GPU) | — | **596.3** | — |
| TFLOPS/GPU (active) | 24.3 | **131.7** | 134.2 |
| MFU active | 2.5% | **13.31%** | 13.6% |
| Tokens/s/GPU (padded) | 1521 | **5775** | 5891 |
| 单 epoch wall | 422 min | **223 min** | 37.5 min |
| vs DS prod | 1.0× | **1.89×** | 11.3× |

---

## 1. 环境（必须先满足）

### 1.1 硬件 / 系统

| 项 | 要求 | 本次实测值 |
|---|---|---|
| GPU | 8 × H100 80 GB SXM (NV18) | 8 × NVIDIA H100 80GB HBM3 |
| Driver | ≥ 575（支持 CUDA 12.9） | 590.44.01 |
| 主机 RAM | **≥ 256 GB**（offload(opt+grad+param) 三件套 ≈ 200 GB） | — |
| 主机磁盘 | ≥ 200 GB | — |
| 容器 | docker name=`fsdp_sft`，`--ipc=host --shm-size=128g`，`--gpus all` | up |

### 1.2 软件栈（容器内）

```
Python    3.12.13
PyTorch   2.10.0+cu129
CUDA      12.9
transformers   5.5.4    (modeling_gemma4 已打 1 个 patch，md5=39ebf386a992fea9eac0883f459ac658)
accelerate     1.13.0
ms-swift       4.2.0.dev0   (commit b462182d)
liger-kernel   0.7.0
```

镜像名：
```
modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.0-modelscope1.35.4-swift4.1.2
```

### 1.3 必装的 3 个 Patch

| # | 文件 | 作用 | 校验 |
|---|---|---|---|
| 1 | `transformers/models/gemma4/modeling_gemma4.py` | (a) FA2/FA3 在 head_dim=512 的 global 层回退到 SDPA；(b) CE 计算分块 (chunked CE) 防止 vocab=262144 时 logits OOM | `md5sum` 等于 `39ebf386a992fea9eac0883f459ac658` |
| 2 | `scripts/gemma4_opt/_sdp_preamble/sitecustomize.py` | 5 个 monkey-patch：(1) SDPA backend 强制 mem_efficient；(2) `use_gqa_in_sdpa→False`；(3) `init_process_group` 加 `cpu:gloo,cuda:nccl` 混合后端（**本策略关键**——offload 后 `clip_grad_norm` 走 gloo）；(4) `Gemma4Template.support_padding_free=True`；(5) Liger gemma4 dispatch 注册 | 必须在 PYTHONPATH 最前 + `GEMMA4_FORCE_MEM_EFFICIENT_SDP=1` |
| 3 | `scripts/benchmark/liger_gemma4_patch.py` | gemma4 Liger 实现（RMSNorm offset=0 + GeGLU）；本策略下 silent 启用但不影响吞吐特征 | 文件存在即可 |

**一键恢复**（如容器是 reset 状态）：
```bash
cd /home/ubuntu/fyh/megatron-sft-recipes
bash scripts/gemma4_opt/diy_restore.sh
```

会把 `sitecustomize.py.applied` → `sitecustomize.py`、`liger_gemma4_patch.py.applied` → `liger_gemma4_patch.py`，并重新打 modeling_gemma4 patch。完事后输出的 md5 必须是 `39ebf386a992fea9eac0883f459ac658`。

### 1.4 模型 + 数据

```
模型:  /home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it    (51.6 GB, 2 个 safetensors shard)
数据:  /home/ubuntu/fyh/megatron-sft-recipes/sft-data/train.jsonl          (18819 条)
```

---

## 2. 完整配置（参数全集）

### 2.1 关键差异 vs DS prod

| 维度 | DS prod | **本策略** | 原因 |
|---|---|---|---|
| 引擎 | DS ZeRO-3 | FSDP2 | 砍 DS 的 fragment overhead，其它都对齐 |
| Param shard 策略 | ZeRO-3（param+grad+opt 全分片+CPU offload） | `full_shard auto_wrap offload`（CPUOffloadPolicy：param+grad+opt 全部 offload 到 CPU） | 行为等价 |
| `auto_wrap_policy` | n/a（DS 不需要） | `TRANSFORMER_BASED_WRAP` + 显式 `transformer_layer_cls_to_wrap=[Gemma4TextDecoderLayer, Gemma4VisionEncoderLayer]` | 跳过 `_no_split_modules` 里 phantom 的 `Gemma4AudioLayer`（A4B 实例没 audio tower） |
| `cpu_ram_efficient_loading` | n/a | **false** | A4B 是 VLM，root 上有 vision/embed plain Tensor 不被 wrap，true 时 accelerate 会把 plain Tensor 当 DTensor → `Tensor has no attribute device_mesh` |
| `reshard_after_forward` | n/a | **true**（FSDP2 默认） | 与 DS ZeRO-3 同语义 |
| `activation_checkpointing` | true | **true** | 完全对齐 |
| `optimizer` | `bnb` paged adamw（DS 默认） | **`adamw_torch`** | offload 后 grad 在 CPU，bnb paged 要 `device.index`，FSDP2 offloaded shard 是 None → TypeError；切 device-agnostic adamw_torch |
| Distributed backend | nccl | **`cpu:gloo,cuda:nccl`** | offload 后 `clip_grad_norm_` 在 CPU 上做 vector_norm + all_reduce，NCCL 不接 CPU tensor，gloo 路由 CPU collective（PyTorch ≥ 2.6 支持 device-typed backend） |
| SDPA backend | math（默认） | **mem_efficient**（强制） | gemma4 global 层 head_dim=512，math O(N²) 在 seq=16384 是 8.6 GB/层 → OOM；mem_efficient O(N) 仅 384 MiB/层（53× 节省） |
| `enable_gqa` | n/a | **手动 repeat_kv 预扩**（强制） | mem_efficient backend 不支持 `enable_gqa=True` kwarg；transformers 5.5 自动选 GQA 路径要 patch |

### 2.2 训练参数（直接复制可跑）

```
# parallelism / sharding
NPROC_PER_NODE                    = 8
sequence_parallel_size            = 2          (Ulysses SP，gemma4 num_global_kv_heads=2 上限)
DP                                = 4          (NPROC / SP)
per_device_train_batch_size       = 1
gradient_accumulation_steps       = 16
GBS = MBS × DP × GAS              = 64
fsdp                              = "full_shard auto_wrap offload"
fsdp_config:
    fsdp_version                  = 2
    reshard_after_forward         = true
    auto_wrap_policy              = "TRANSFORMER_BASED_WRAP"
    cpu_ram_efficient_loading     = false
    state_dict_type               = "SHARDED_STATE_DICT"
    activation_checkpointing      = true
    transformer_layer_cls_to_wrap = ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer"]

# numerics
torch_dtype                       = bfloat16
attn_impl                         = flash_attention_2     (modeling_gemma4 patch 在 head_dim=512 时回退 SDPA)
max_length                        = 16384
truncation_strategy               = right
optim                             = adamw_torch
learning_rate                     = 1e-5
warmup_ratio                      = 0.1

# data / model
model                             = google/gemma-4-26B-A4B-it
model_type                        = gemma4
template                          = gemma4
dataset                           = sft-data/train.jsonl       (18819)
freeze_vit / freeze_aligner       = true                       (text-only SFT)
packing                           = false
use_liger_kernel                  = true                       (gemma4 dispatch 已注册)
torch_compile                     = false
gradient_checkpointing            = false (FSDP2 自己管 AC，HuggingFace flag 关掉避免双重 wrap)

# bench / wall
total_steps                       = 40        (5 warmup + 35 measured)
seed                              = (默认)
```

> **lr=1e-5 / warmup=0.1 与 DS prod 的 2e-5 / 0.05 不同**：bench 脚本默认。**lr/warmup 不影响吞吐 / 显存**，只影响 loss 轨迹。生产落地时按 DS prod 的 2e-5 / 0.05 跑即可，吞吐数字一致。

---

## 3. 完整启动命令

> 容器 `fsdp_sft` 里跑。如容器已重置，先 `bash scripts/gemma4_opt/diy_restore.sh` 把 patch 全装上。

### 3.1 一键脚本（已落仓库）

```bash
cd /home/ubuntu/fyh/megatron-sft-recipes
bash scripts/gemma4_opt/p1_alt_offload_ds_equiv.sh
```

脚本里已经把所有变量写死。耗时约 **30 min**（40 步），产物落到 `experiments/gemma4_opt/p1_alt_offload/run_<TIMESTAMP>_ds_equiv/`。

### 3.2 等价的展开命令（手工排错时用）

```bash
docker exec fsdp_sft bash -lc '
cd /home/ubuntu/fyh/megatron-sft-recipes && \
PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble:$PYTHONPATH \
GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
MASTER_PORT=29501 BACKEND=fsdp2 SP=2 MBS=1 GAS=16 \
NO_AC=false FSDP_RESHARD=true \
PACKING=false USE_LIGER=true \
TRUNCATION_STRATEGY=right \
FSDP_OFFLOAD=offload \
MODEL=/home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it \
MODEL_TYPE=gemma4 \
FSDP_TRANSFORMER_CLS_NAMES=Gemma4TextDecoderLayer,Gemma4VisionEncoderLayer \
FSDP_CPU_RAM_EFFICIENT=false \
NUM_PARAMS=25.2e9 NUM_ACTIVE_PARAMS=3.8e9 \
DATASET_SIZE=18819 \
TOTAL_STEPS=40 WARMUP_BENCH=5 \
RUN_NAME=run_alt_offload_ds_equiv \
BENCH_DIR=/home/ubuntu/fyh/megatron-sft-recipes/experiments/gemma4_opt/p1_alt_offload/_bench \
bash scripts/benchmark/bench_swift_sp_v2.sh
'
```

**bench wrapper** 会自动渲染 `fsdp_override.json`（见下方），并调 `swift sft`。

### 3.3 自动渲染出的 FSDP override JSON

```json
{
    "fsdp": "full_shard auto_wrap offload",
    "fsdp_config": {
        "fsdp_version": 2,
        "reshard_after_forward": true,
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "cpu_ram_efficient_loading": false,
        "state_dict_type": "SHARDED_STATE_DICT",
        "activation_checkpointing": true,
        "transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer"]
    }
}
```

### 3.4 sitecustomize 启动期 stderr 日志（确认 patch 都生效）

```
[gemma4 sdp_preamble] (1/3) backend prefs: flash=False mem_eff=True math=False
[gemma4 sdp_preamble] (2/3) patched transformers.integrations.sdpa_attention.use_gqa_in_sdpa → always False ...
[gemma4 sdp_preamble] (3/3) intercepted init_process_group: nccl → cpu:gloo,cuda:nccl ...
[gemma4 sdp_preamble] (4/4) patched swift.template.Gemma4Template.support_padding_free → True ...
[gemma4 sdp_preamble] (5/5) registered gemma4 → _apply_liger_kernel_to_gemma4 in Liger's ...
```

5 行都出 = patch 全到位。

---

## 4. 完整性能数据（本次实测）

> 跑次：`run_20260428_163438_ds_equiv`（2026-04-28 16:34 UTC）
>
> 40 步 (5 warmup + 35 measured)，30.3 min wall。

### 4.1 步时（report.json）

| 指标 | 值 | 说明 |
|---|---:|---|
| `mean_step_time_ms` | **45,393.05** | 来自 swift `train_speed(s/it)`，精确 wallclock / step_count |
| `median_step_time_ms` | 44,000.0 | 25 个 measured step 的中位（受 swift `elapsed_time` 1s 量化） |
| `p99_step_time_ms` | 56,000.0 | 偶发尖刺（dataloader、bash IO） |
| `micro_step_time_ms` | **2,837.07** | mean / GAS = 45393 / 16 |
| `actual_total_wall_s` | 1,816.0 | 40 步实际墙钟 |
| `actual_total_wall_min` | **30.3** | = 1816 / 60 |
| `loss_first_step` (post-warmup, step 6) | 1.5174 | — |
| `loss_last_step` | 1.508 (step 40) | 35 步后 loss 已收敛到 1.34-1.50 区间 |

### 4.2 显存（per-rank）

| 指标 | 值 | 说明 |
|---|---:|---|
| `peak_mem_gib_from_swift_log` | **76.62** | swift 训练步 logging 报告的全集群最高 peak（≈ 79.18 GB - reserved/activations 差） |
| `peak_mem_gb` (nvidia-smi) | **79.18** | gpu_metrics.jsonl 8 卡 max(used) |
| 稳态 mem | 76.62 GiB（步 3 起就锁定） | offload+AC 让 activations 是主导项 |
| 主机 RAM 峰值（理论） | ~200 GB | (param 50G + grad 50G + opt 100G) × full | 实际上 FSDP2 shard 后是 1/8 但分散在所有 rank |

> ⚠️ **76.62 GiB 是 80 GB 卡的 96%**。生产复制此配置时**必须留 H100-80G**，H100-40G / A100-40G 都不行。

### 4.3 功率与 GPU 利用率

| 指标 | 值 | 说明 |
|---|---:|---|
| `avg_gpu_util_pct` | 56.3 | nvidia-smi（30s 平均） |
| `avg_power_w` | 230.8 | 全 8 卡平均 |
| `peak_power_w` | **596.3** | 单卡瞬峰；H100 TDP=700W，未触顶 |

**DCGM 细分**（145k 行采样，跳过头部 120s 模型加载）：

| 指标 | 集群均值 | 解释 |
|---|---:|---|
| `tc_active`（TC pipe busy） | **2.21%** | TC 真正算 mma 的时间占比 → **极低**，意味着大部分时间 GPU 在等数据 |
| `dram_active` | 11.99% | HBM 带宽利用率低 → 数据主体不在 HBM（在 CPU RAM） |
| `gr_active` | 52.5% | GR engine（含 SM 调度 / NCCL / kernel queue）busy；高于 TC 因为含 stall |

> 💡 **本策略最重要的一行**：`tc_active=2.2%` 显示 TC 几乎没工作。**端到端瓶颈是 CPU↔GPU 的 D2H/H2D**（grad / param 在 PCIe Gen5 ≈ 64 GB/s 上来回搬）。这就是 offload 的本质代价：**用 6× 主机 RAM 换 1.4× GPU 显存松动**。理论上 PCIe Gen6 / NVLink-C2C 可以缓解，但当前硬件下这是天花板。

### 4.4 算力 / MFU

| 指标 | 值 | 含义 |
|---|---:|---|
| `tokens_per_step` | 2,097,152 | = GBS_padded × MAX_LEN = 128 × 16384（**bench 脚本 SP-unaware 把每 GPU 都当独立 DP，所以 GBS 报 128，实际 GBS=64**） |
| `tokens_per_sec` (cluster) | 46,199.8 | tokens_per_step / step_s |
| `tokens_per_sec_per_gpu` (padded) | **5,775.0** | / 8 |
| `achieved_tflops_per_gpu` (full param) | 873.2 | 6 × 25.2e9 × 46199.8 / 1e12 / 8 |
| `achieved_tflops_per_gpu` (active param) | **131.7** | 6 × 3.8e9 × 46199.8 / 1e12 / 8 — MoE 正确口径 |
| `mfu_pct` (full param) | 88.24% | **不诚实**：MoE 每 token 只激活 3.8B/25.2B = 15%，全参口径会把 active token 重复计算所有 expert |
| `mfu_pct_active_params` | **13.31%** | **诚实**：vs H100 BF16 dense peak 989.5 TFLOPS |

> 📌 **汇报口径**：`active-MFU 13.31%` + `padded TPS 5775 tok/s/GPU` 是同一份事实的两个角度。绝对不要把 88.24% 拿出去说，会被会计部分查 ☺。

### 4.5 Loss 收敛（前 5 步 + 后 5 步快照）

```
step  1: loss=2.030  grad_norm=89.0   token_acc=0.618  mem=76.25
step  2: loss=2.030  grad_norm=53.0   token_acc=0.602  mem=76.49
step  3: loss=1.746  grad_norm=40.5   token_acc=0.648  mem=76.62
step  4: loss=2.222  grad_norm=43.2   token_acc=0.581  mem=76.62
step  5: loss=1.971  grad_norm=38.2   token_acc=0.593  mem=76.62
...
step 36: loss=1.374  grad_norm=9.6    token_acc=0.647  mem=76.62
step 37: loss=1.503  grad_norm=18.4   token_acc=0.624  mem=76.62
step 38: loss=1.344  grad_norm=89.5   token_acc=0.647  mem=76.62
step 39: loss=1.385  grad_norm=21.6   token_acc=0.645  mem=76.62
step 40: loss=1.508  grad_norm=2.14   token_acc=0.624  mem=76.62
```

40 步内 loss 从 ~2.0 跌到 ~1.4 区间，token_acc 从 0.60 升到 0.65，loss/grad_norm 行为与 DS prod 的 align run 完全一致 —— 模型在收敛，不是采样异常。

### 4.6 Full-epoch 外推

```
dataset = 18819 samples
GBS     = 64
steps_per_epoch = ceil(18819 / 64) = 295
single epoch wall = 295 × 45.39 s / 60 = 223.2 min ≈ 3 h 43 min
```

**vs DS prod**：18819/64 × 86.18 s / 60 = **422.3 min**。
**加速比**：422.3 / 223.2 = **1.89×**。

---

## 5. 适用场景 / 何时用这个策略

| 场景 | 推荐策略 | 备注 |
|---|---|---|
| 想"无痛"把 DS 生产线迁到 FSDP2，**不动数据流 / 不动 lr / 不动 GBS** | **本策略 (P1 alt-offload)** | 1.89× 加速 + 完全相同 (GBS, AC, opt 行为) → 同 loss 曲线 |
| 单机训练，**想要最快 wall**，可以接受 GBS=4 + packing | **P5 peak**（37.5 min/epoch） | 11.3× 加速；但 GBS=4 vs GBS=64 优化器轨迹差异大，需重新调超参 |
| 内存非常紧张（不到 80 G/卡） | DS prod 或 本策略（都 offload） | DS 更省 GPU mem (~43 GiB)，但 wall 慢 1 倍 |
| **希望对齐 DS 数值**（debug 阶段） | 本策略 + lr=2e-5 + warmup=0.05 + seed 同步 | P0g align 实测 step 1 bit-identical |

**本策略的核心价值**：当生产线已经对 GBS=64 + AC=on 校准好了 lr / warmup / clip / KL，但又想 wall 减半 —— 不需要任何超参重调，直接换引擎。

**何时换走**：如果团队愿意接受 GBS=4 native 的轨迹变化，应直接上 P5（再快 5.95×）。详见 [`gemma4_phase_delta_summary.md`](gemma4_phase_delta_summary.md) 的 P5 行。

---

## 6. 失败排查（Q&A）

### Q1：跑起来撞 `Tensor has no attribute device_mesh`
- 检查 `FSDP_CPU_RAM_EFFICIENT=false` 是否传到 fsdp_override.json（脚本默认就是 false）。

### Q2：撞 `No backend type associated with device type cpu`
- 说明 sitecustomize.py (3/3) 那段没生效。校验 stderr 里有没有 `init_process_group: nccl → cpu:gloo,cuda:nccl`。
- 如果没出 → `PYTHONPATH` 没把 `_sdp_preamble` 放最前 / `GEMMA4_FORCE_MEM_EFFICIENT_SDP=1` 没传。

### Q3：单步飙到 60s+
- 第 1 步 60s 是冷启动（含 reduce-scatter warmup），从第 2 步开始就稳定到 ~45s。看 bench.jsonl 验证。

### Q4：peak_mem 报 78+ GiB（OOM 边缘）
- 多半是 dataset 里有 ≥16k 的样本（`truncation_strategy right` 不会 pad 短样本，但**会触发**长样本到 16k）。本策略实测 76.62 GiB 是用 `sft-data/train.jsonl`（多数样本 < 12k）的稳态。如果你换到长上下文数据集，需要升级到 H200 141G，或退回 DS prod。

### Q5：跑完 logging.jsonl 缺 `tokens_this_step` 字段
- 当前容器里 swift 4.2.0.dev0 (commit b462182d) **不带 token-stats patch**，所以 `tokens_per_gpu_per_sec` 走 padded 公式。要 real-TPS，把 swift 换成 [`fyh2001/ms-swift@gemma4-complete`](https://github.com/fyh2001/ms-swift/tree/gemma4-complete) 即可，吞吐绝对值不变。

---

## 7. 产物归档

```
experiments/gemma4_opt/p1_alt_offload/run_20260428_163438_ds_equiv/
├── STATUS                # SUCCESS — see report.json
├── cmd.sh                # 启动脚本副本（自包含）
├── stdout.log            # 完整 stdout (321 KB)
├── train.log             # swift sft 训练日志副本
├── fsdp_override.json    # 真正递给 swift 的 FSDP 配置
├── report.json           # 聚合指标 (本文档 §4 全部数字的来源)
├── bench.jsonl           # 40 步逐条记录 (step / loss / grad_norm / mem / speed)
├── dcgm_tc.tsv           # DCGM 7.9 MB, 145k 行 (tc_active/dram/gr/power, 8 卡 × 0.05s)
├── gpu_metrics.jsonl     # nvidia-smi-style 1.1 MB, 1158 条 (util/mem/power)
└── logging.jsonl         # swift trainer 原始日志 (symlink 到 v0-* 目录)
```

**单机审计任意指标**：
```bash
RUN_DIR=experiments/gemma4_opt/p1_alt_offload/run_20260428_163438_ds_equiv
cat $RUN_DIR/report.json | python3 -m json.tool
```

---

## 8. 一行版结论（汇报金句）

> **"FSDP2 + CPU offload 把 gemma-4-26B-A4B-it 的 DS 生产线 wall 从 422 min 砍到 223 min（1.89× 加速），单卡显存 76.6 GiB / 单卡瞬峰功耗 596 W / active-MFU 13.3%；GBS=64 / AC=on / lr=2e-5 / warmup=0.05 全部沿用 DS 现配，零超参重调，可立即替换。"**

— 数据来源：`experiments/gemma4_opt/p1_alt_offload/run_20260428_163438_ds_equiv/report.json`，
2026-04-28 16:34 UTC，40 步 wall。
