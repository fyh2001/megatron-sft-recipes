# Qwen3.5-9B × 8×H100 训练加速实战：从 baseline 到 3.8× 的完整路径

> 这是一份**面向复现 + 学习**的全过程文档。覆盖：诊断方法、每个优化轴为什么能（不能）有效、参数含义、实测前后对比，以及一字不差的复现命令。
>
> 实验日期：2026-04-23/24 · 硬件 8 × NVIDIA H100 80GB SXM · 模型 Qwen3.5-9B（VLM，freeze ViT，7.94 B trainable）· 序列长度 16384 · ms-swift 4.2.0.dev0 (git main) · torch 2.10.0+cu129 · NCCL 2.27.5

---

## 目录

- [0. 环境准备](#0-环境准备)
- [1. Baseline：为什么"看着挺快"其实没在干活](#1-baseline)
- [2. MFU 三种口径辨析（先把尺子校准）](#2-mfu-三种口径辨析)
- [3. 诊断工具栈（DCGM / torch.profiler / nsys）](#3-诊断工具栈)
- [4. 第一波优化：NO_AC / no_reshard / MBS=2 / combo_easy（wall time 没省，但学到了瓶颈）](#4-第一波优化)
- [5. 真瓶颈：padding waste 与 packing](#5-真瓶颈padding-waste-与-packing)
- [6. Liger kernels：fused RMSNorm + SwiGLU + RoPE](#6-liger-kernels)
- [7. 失败的尝试（学习意义）](#7-失败的尝试)
- [8. 三后端对比：DS / Megatron 能不能套同样优化](#8-三后端对比)
- [9. 最终配置 & 一键生产训练命令](#9-最终配置--一键生产训练命令)
- [附录 A：完整数据表](#附录-a完整数据表)
- [附录 B：关键文件清单](#附录-b关键文件清单)

---

## 0. 环境准备

### 0.1 硬件 & 软件栈

| 项 | 值 |
|---|---|
| GPU | 8 × H100 80GB SXM（NVSwitch 全互联，NV18 / 477 GB/s 双向 per pair） |
| Docker 容器 | `fsdp_sft`（基于 `modelscope-registry...:ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.0-modelscope1.35.4-swift4.1.2`） |
| Python / torch / NCCL | 3.12 / **2.10.0+cu129** / **2.27.5** |
| transformers / accelerate | 5.5.4 / 1.13.0 |
| ms-swift | **4.2.0.dev0**（git main，**必须**，pypi 4.1.2 有 SP bug） |
| flash-attn / liger_kernel | 2.8.3 / installed |
| 数据 | `sft-data/train.jsonl`，18819 条多轮对话样本 |

### 0.2 必装/必起的两件事

```bash
# (a) 升级 ms-swift 到 git main，吃到 PR #9162/#9167/#9189（FSDP2 Ulysses SP 修复）
docker exec fsdp_sft pip install --force-reinstall --no-deps --no-cache-dir \
    git+https://github.com/modelscope/ms-swift.git@main
docker exec fsdp_sft python -c "import swift; print(swift.__version__)"
# 期望输出 4.2.0.dev0

# (b) 起 dcgm-exporter 10 Hz（profile 时用，不开也能训练）
docker run -d --rm --gpus all --name dcgm-exporter-fast \
    -p 9500:9400 --cap-add SYS_ADMIN \
    nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3-4.1.3-ubuntu22.04 \
    dcgm-exporter -c 100
# 验证 PROF metrics 开了
curl -s localhost:9500/metrics | grep -c "^DCGM_FI_PROF"
# 期望 ≥ 5
```

### 0.3 nsys 2025（H100 必需，apt 装的 2021 不认 Hopper）

CUDA toolkit 12.9 自带 nsys 2025.2.1 但路径冷门。重新链一下：

```bash
docker exec fsdp_sft ln -sf \
    /opt/nvidia/nsight-compute/2025.2.1/host/target-linux-x64/nsys \
    /usr/local/bin/nsys
docker exec fsdp_sft nsys --version
# 期望 NVIDIA Nsight Systems version 2025.2.1.0
```

如果 `apt list --installed | grep nsight` 显示 2021.3，**那个识别不了 H100 kernel**，只能采到 NVTX，不要用。

---

## 1. Baseline

### 1.1 起点：默认 FSDP2 + Ulysses SP=2

```bash
docker exec fsdp_sft bash -lc "cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  MASTER_PORT=29500 \
  BACKEND=fsdp2 SP=2 MBS=1 \
  TOTAL_STEPS=200 WARMUP_BENCH=5 \
  RUN_NAME=baseline \
  BENCH_DIR=/home/ubuntu/perf_opt/megatron_output/walkthrough \
  bash scripts/benchmark/bench_swift_sp_v2.sh"
```

实际它最后会执行（`bench_swift_sp_v2.sh` 转成的命令）：

```bash
swift sft \
    --model /root/.cache/huggingface/models/Qwen/Qwen3.5-9B \
    --model_type qwen3_5 \
    --dataset sft-data/train.jsonl \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --max_length 16384 \
    --truncation_strategy delete \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_steps 200 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --freeze_vit true --freeze_aligner true \
    --fsdp fsdp2 \
    --sequence_parallel_size 2 \
    --dataloader_num_workers 2 \
    --dataset_num_proc 4 \
    --save_strategy no \
    --logging_steps 1 \
    --output_dir <BENCH_DIR>/baseline
```

### 1.2 Baseline 数据

| 指标 | 值 | 来源 |
|---|---:|---|
| steady step time | **0.95 s** | 200 步 EMA（跳过前 5 warmup） |
| peak mem | 28.6 GiB | swift `memory(GiB)` |
| swift 公式 MFU | 62.6% | `mfu_pct` in report.json |
| 真 MFU（per-rank GEMM/wall） | **2.3%** | torch.profiler step 3 实测 |
| peak power | 247 W | DCGM 10 Hz 峰值 |
| GPU avg util | **8%** | `avg_gpu_util_pct` |

### 1.3 关键观察：GPU 几乎在睡觉

H100 SXM TDP **700 W**，baseline peak 只 **247 W = 35% TDP**。GPU avg util 8% 说明大部分时间在等。这就是诊断的入口。

---

## 2. MFU 三种口径辨析

**这是整个工作的方法论核心**，弄不清楚会被假数字骗。

### 2.1 ① swift 公式 MFU（不可信用作底线）

```
swift_mfu = 6 × N_params × tokens_per_step / step_time / num_gpus / 989_TFLOPs
```

- 假设：所有 FLOPs 都经 tensor core；GPU 100% busy；tokens_per_step 全是真 token（含 padding）
- **baseline 报 62.6%**——这数字是**虚的**。证据：peak power 只 247 W，物理上不可能跑 60% MFU
- 适用场景：作为 **同一 backend 内** 的相对比较；**不能跨 backend 比**（Megatron TP=4 切小 GEMM tile 会让公式 MFU 偏低）

### 2.2 ② TC 占 aggregate GPU busy 时间（口径不一致）

```
tc_of_agg = (GEMM_ms + FlashAttn_ms) / total_kernel_ms_aggregated_over_all_streams
```

陷阱：
- FSDP2 的 NCCL stream 与 compute stream 异步并行 → aggregate / wall 可达 113%（多流相加）
- Megatron TP=4 stream 串行较多 → aggregate / wall ≈ 100%
- **分母不一样**，直接对比会让 Megatron 看着 TC 比例高（因为分母小）

我开始时用了这个口径，得出 "Megatron 51% > FSDP2 44%" 的结论，是**错的**。

### 2.3 ③ Per-rank GEMM time / wall（推荐口径）

```
true_mfu_upper_bound = (GEMM_aggregate_ms / num_ranks) / wall_capture_ms
```

- 这是"每张卡在 wall 时间内，tensor core pipeline 在跑的比例"
- 是**真 MFU 的上界**（假设 GEMM 跑到 989 TF 峰值；实际更低）
- 跨 backend、跨 stream 配置都能直接比
- **本文后续所有"真 MFU"都用这个口径**

实测（nsys 2025 + 8 ranks aggregate + 10s steady-state 窗口）：

| 配置 | 真 MFU 上界 |
|---|---:|
| baseline | ~3% |
| **pack_liger（最终配置）** | **50.4%** |
| Megatron TP=4 SP + packing | 51.1% |

### 2.4 ④ DCGM peak power（物理旁证）

H100 TDP 700 W。peak power 越接近 TDP，GPU 越在干活。这个谁也作不了假。

---

## 3. 诊断工具栈

不会诊断就只能盲调。三件套：

### 3.1 DCGM 10 Hz 采样

DCGM `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` 是"过去 100 ms 窗口内 tensor core pipe 活跃 cycles 比例"。dcgm-exporter `-c 100` = 100 ms 刷新；scraper 也 100 ms 拉取。

```bash
# 已在 0.2 起好的 dcgm-exporter-fast
# scraper：会自动被 bench_swift_sp_v2.sh 拉起
# 输出：<BENCH_DIR>/<run>/dcgm_tc.tsv
# 字段：ts gpu tc_active dram_active gr_active power_w
```

实现：[scripts/benchmark/dcgm_scrape.py](../scripts/benchmark/dcgm_scrape.py)，环境变量 `DCGM_SCRAPE_INTERVAL_S` 控间隔（默认 0.1）。

### 3.2 torch.profiler 一步精捕（rank 0 only）

适合：单步 kernel breakdown，看 forward/backward/optimizer 的细 kernel 名字。

实现：
- [scripts/benchmark/torch_profile_callback.py](../scripts/benchmark/torch_profile_callback.py) — `TrainerCallback`，在指定 step 启停 profiler
- [scripts/benchmark/swift_sft_profiled.py](../scripts/benchmark/swift_sft_profiled.py) — 启动 wrapper，monkey-patch `Trainer.__init__` 注入 callback（用 `functools.wraps + __signature__` 保留签名，避免 swift 用 `inspect.signature` 选 `tokenizer`/`processing_class` 时选错）
- [scripts/benchmark/analyze_torch_profile.py](../scripts/benchmark/analyze_torch_profile.py) — 解析 chrome trace，按 kernel 名字分类（GEMM/FlashAttn/NCCL/Memory/Elementwise/...）

```bash
docker exec fsdp_sft bash -lc "cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  TOTAL_STEPS=10 PROFILE_STEP=5 \
  BACKEND=fsdp2 SP=2 MBS=2 NO_AC=true FSDP_RESHARD=false \
  PACKING=true USE_LIGER=true \
  RUN_NAME=run BENCH_DIR=/tmp/myprofile \
  bash scripts/benchmark/bench_swift_sp_profile.sh"
# 输出：/tmp/myprofile/run/{trace_rank0_step5.json, analysis.txt}
```

**注意**：torch.profiler 只能采 rank 0，加 ~30% step 时间开销。**不适合**对比 wall time，**适合**看 kernel 分布。

### 3.3 nsys 2025（apples-to-apples 跨 backend 对比的金标准）

8 ranks 同时采、所有 stream 都看到、10 s 窗口 steady-state。

```bash
NSYS=/opt/nvidia/nsight-compute/2025.2.1/host/target-linux-x64/nsys

docker exec fsdp_sft ${NSYS} profile \
    --trace=cuda,nvtx,cublas,cudnn \
    --trace-fork-before-exec=true \
    --cuda-memory-usage=false --sample=none \
    --delay=150 --duration=10 \
    --output=/tmp/myrun.nsys-rep --force-overwrite=true \
    torchrun --nproc_per_node 8 --master_port 29600 \
        /usr/local/lib/python3.12/site-packages/swift/cli/sft.py \
        <swift sft 完整参数...>

# 解析：
docker exec fsdp_sft nsys stats --report cuda_gpu_kern_sum \
    --format csv /tmp/myrun.nsys-rep > /tmp/kernels.csv
# 然后按 kernel 名字分类（参考 analyze_torch_profile.py 的 classify()）
```

**delay=150 duration=10** 意味着等 150 s（cover 模型加载 + compile cold start + 几个 warmup 步）然后采 10 s。具体延迟数得根据 step time 和 compile 时长调；可以先 `delay=0` 跑短的看时间线，再调。

### 3.4 关键陷阱

- **DCGM 1 Hz 太粗**：早期我们用 1 Hz 采，对 1.4 s/step 的 baseline 出现严重 aliasing。**必须 10 Hz**（`-c 100`）
- **同时跑 DCGM + nsys** 在某些情况下会冲突 profile counter。Profile 时建议关 dcgm-exporter
- **GPU 被其他进程占用时**，DCGM 可能不暴露 PROF metrics（要 exclusive profile slot）

---

## 4. 第一波优化

主线：**逐个轴试，看哪个改动让 wall time 降**。结果是：**单个轴都没省 wall**，但发现真 MFU 路径，为下一波 packing 铺路。

### 4.1 优化轴 1：NO_AC（关闭 activation checkpointing）

**问题**：swift 的 fsdp2 preset 默认开 `activation_checkpointing: true`：

```json
{
    "fsdp_version": 2,
    "reshard_after_forward": true,
    "activation_checkpointing": true   // ← 默认开
}
```

AC 的代价：每个 transformer block 在反向时**重做一次 forward**，多 33% FLOPs。设计目的是**省 activation memory**。但 baseline peak mem 才 26 GB / 80 GB，**根本不缺内存**——AC 是没必要的开销。

**怎么改**：`bench_swift_sp_v2.sh` 暴露 `NO_AC` env，true 时写一个 fsdp_override.json 关掉 AC。

```bash
NO_AC=true bash scripts/benchmark/bench_swift_sp_v2.sh
```

**实测效果**：

| 指标 | baseline | no_ac | Δ |
|---|---:|---:|---:|
| step time | 1.42 s | 1.32 s | −7% |
| peak mem | 26.3 GiB | 26.3 GiB | 0 |
| tokens/s/GPU | 11 529 | 12 416 | +8% |

省了 7% step。但 wall 没显著变（per-step 数字小、EMA 噪声大）。

### 4.2 优化轴 2：reshard_after_forward=false（FSDP2 ZeRO-2 模式）

**问题**：FSDP2 默认 `reshard_after_forward=true`（FULL_SHARD / ZeRO-3），意思是 forward 后立即丢掉 unsharded params；反向再 AllGather 一次。

**这浪费了一次完整的 AllGather**（NCCL 时间 191 ms 里有 132 ms 是 forward AllGather + 反向 AllGather）。

**改成 false**：forward 后保留 unsharded params，反向直接用——**省一次 AllGather**，但 peak mem 涨（每个 rank 永久持有 unsharded params 的副本）。

```bash
FSDP_RESHARD=false bash scripts/benchmark/bench_swift_sp_v2.sh
```

实现：在 fsdp_override.json 里设 `"reshard_after_forward": false`。

**实测效果**：

| 指标 | baseline | no_reshard | Δ |
|---|---:|---:|---:|
| step time | 1.42 s | 1.55 s | +9% (??) |
| peak mem | 26 GiB | **45 GiB** | +73% |
| NCCL 时间 | 191 ms | 146 ms | **−24%** |
| DCGM TC busy 20% | 0.25% | **8.73%** | **×35** |

**奇怪点**：NCCL 显著降了、DCGM TC 显著升了，但 step time 反而**变慢一点**。原因不明（推测：peak mem 涨到 45 GB 后某种 allocator pressure；或 stream overlap 模式变了）。

### 4.3 优化轴 3：MBS=2（per device batch size）

**问题**：MBS=1 + GBS=8 + 8 GPUs 意味着每步 8 个样本。每个样本 16384 token 容量，但平均真实长度只 ~2823 token，**padding 浪费 83%**。增大 MBS 让单步处理更多样本→压更多算力到 GPU。

```bash
MBS=2 bash scripts/benchmark/bench_swift_sp_v2.sh
```

**实测效果（坑）**：

| 指标 | baseline | mbs2 | 影响 |
|---|---:|---:|---|
| step time | 1.42 s | **17.6 s** | **+1140%！** |
| peak mem | 26 → 50 GiB | | |
| step time 分布 | 稳定 | 2/3/15/37/38 s 大幅波动 | |

**惨败**。原因：peak mem 50 GB 接近 H100 80 GB 的 60%，一旦 forward 内存峰值再 spike，FSDP2 触发**慢路径**（NCCL 反复申请释放 buffer），单步从 2 s 飙到 38 s。

**教训**：单独开 MBS=2 不行，必须配 NO_AC + no_reshard 来稳住内存模式。

### 4.4 组合 combo_easy（NO_AC + MBS=2 + no_reshard）

**关键尝试**：把上面三个轴一起开。

```bash
BACKEND=fsdp2 SP=2 MBS=2 \
NO_AC=true FSDP_RESHARD=false \
bash scripts/benchmark/bench_swift_sp_v2.sh
```

**实测效果**：

| 指标 | baseline | combo_easy | 解读 |
|---|---:|---:|---|
| step time | 1.42 s | **2.40 s** | +69% |
| MBS（per step samples） | 8 | **16** | ×2 |
| **per-sample wall** | 0.18 s | **0.15 s** | **−17%！** |
| peak mem | 26 GiB | **54 GiB** | +108% |
| peak power | 247 W | **692 W** | +180% |
| avg power | 120 W | 228 W | +90% |
| GPU util | 8% | **51%** | +43pp |
| 真 MFU（profile） | ~3% | **~17%** | ×6 |

**关键发现**：
- step time 翻了 1.7×，但每步处理 2× 样本 → **per-sample 时间降了 17%**
- 功耗冲到 692 W，几乎贴 TDP——**GPU 真在干活了**
- 30 步长跑 0 NaN，loss 正常收敛 1.75 → 0.74

**但是！** 用同样 1600 samples 做 wall time 比对：

| | wall_baseline (200 步 MBS=1) | wall_combo (100 步 MBS=2) |
|---|---:|---:|
| 1600 samples 总 wall | **244 s** | **246 s** |

**完全一样**。combo_easy 让 GPU 真在干活、真 MFU 升到 17%，但**没省总训练时间**。

**为什么？** 因为 **GPU 不是瓶颈，padding 是**。每步多塞 token 进 GPU，GPU 算得快了；但 padding 占 83% 的 token，多算的也大部分是 padding。所以总训练时间不变。

**这是整个工作最关键的诊断**。下一步必须解决 padding。

---

## 5. 真瓶颈：padding waste 与 packing

### 5.1 问题量化

数据集 `sft-data/train.jsonl`：18 819 条多轮对话样本。**packing 后**会发现：
- 总 packs 数：3252
- 每个 pack 平均长度：**16 341 token**（接近 max_len 16384，几乎 100% 填充）
- 每个 pack 平均原样本数：18819 / 3252 = **5.8 条**
- 推算：单条样本平均真实 token = 16341 × 3252 / 18819 = **2823 token**

baseline 没开 packing 时：
- 每条样本被 pad 到 16384 token，**真实 token / nominal token = 2823/16384 = 17%**
- **GPU 80% 时间在算 padding**

### 5.2 解决方案：`--packing true`

swift 的 packing 实现：把多条 short samples 用 `cu_seqlens`（cumulative sequence lengths）拼成一个 16384 长度的"超样本"，flash_attention_2 用 `varlen` 路径正确处理边界。**不需要改模型结构**。

```bash
# bench_swift_sp_v2.sh 接受 PACKING=true env，加一行 --packing true 到 swift sft
PACKING=true bash scripts/benchmark/bench_swift_sp_v2.sh
```

### 5.3 优化轴 4：pack（combo_easy + packing）

```bash
BACKEND=fsdp2 SP=2 MBS=2 \
NO_AC=true FSDP_RESHARD=false \
PACKING=true \
bash scripts/benchmark/bench_swift_sp_v2.sh
```

**实测效果**（100 步 wall time）：

| 指标 | combo_easy | pack | Δ |
|---|---:|---:|---:|
| step time | 2.40 s | **3.88 s** | +62% |
| 每步真实 tokens | ~45 k | **261 k** | **×5.8** |
| peak mem | 54 GiB | 59 GiB | +9% |
| peak power | 692 W | **707 W** | 99% TDP |
| avg power | 228 W | **443 W** | ×1.9 |
| 真 MFU（profile） | ~17% | **~37%** | ×2 |
| **full-epoch wall** | 37 min | **13 min** | **−65%（2.8×）** |

**这次是真胜利**。step time 长了 62%，但真实 token 翻了 6 倍，效率翻倍。

### 5.4 packing 的代价 / 注意

- **dataset 预处理时间增加**：第一次跑 packing，`dataset_num_proc=4` 下需要 ~1-2 min 把 18819 条样本打包成 3252 packs
- **loss 数值变了**：因为 cross-entropy 现在按 pack 内多条样本拼接计算，单步 loss 数值和不 packing 时不一样（绝对值会变小，因为 normalization 不同）。**收敛趋势仍正常**
- **要求 attn_impl=flash_attention_2 或 sdpa**：传统 attn 不会用 cu_seqlens

---

## 6. Liger kernels

### 6.1 问题：elementwise kernel 太多

profile pack 一步看 GPU kernel 分布：

| 类别 | 时间 | % |
|---|---:|---:|
| GEMM | 393 ms | 38.8% |
| NCCL | 246 ms | 24.3% |
| **Elementwise**（RMSNorm, SwiGLU, RoPE, mask, dropout, copy） | **130 ms** | **12.9%** |
| Memory shuffle | 125 ms | 12.3% |
| Others | 118 ms | 11.7% |

13% 时间花在小 kernel。每个 kernel 启动有 ~5-10 μs CPU 调度 + GPU launch overhead。这种"小算力 + 大开销"是融合的最佳目标。

### 6.2 解决方案：Liger kernels

[Liger-Kernel](https://github.com/linkedin/Liger-Kernel) 是 Triton 写的融合算子库，把：
- `RMSNorm`（Qwen 系列用）
- `SwiGLU`（Qwen FFN 激活）
- `RoPE`（旋转位置编码）

各自融成单个 Triton kernel。每个原本要 3-5 个 elementwise + reduce kernel，融成 1 个。

swift 集成方式：`--use_liger_kernel true`。swift 会自动识别模型架构，monkey-patch 对应模块（Qwen3.5 的 RMSNorm/SwiGLU/RoPE 都有 Liger 实现）。

```bash
USE_LIGER=true bash scripts/benchmark/bench_swift_sp_v2.sh
```

### 6.3 优化轴 5：pack_liger（pack + Liger）

```bash
BACKEND=fsdp2 SP=2 MBS=2 \
NO_AC=true FSDP_RESHARD=false \
PACKING=true USE_LIGER=true \
bash scripts/benchmark/bench_swift_sp_v2.sh
```

**实测效果**：

| 指标 | pack | pack_liger | Δ |
|---|---:|---:|---:|
| step time | 3.88 s | **2.88 s** | **−26%** |
| peak mem | 59 GiB | 59 GiB | 0 |
| avg power | 443 W | **519 W** | +17% |
| GPU util | 68% | **82%** | +14pp |
| 真 MFU（nsys） | ~37% | **~50%** | +13pp |
| **full-epoch wall** | 13 min | **9.8 min** | **−25%** |

Liger 单独贡献了 25% wall time 缩短。

### 6.4 Qwen3.5 GDN 层兼容性

Qwen3.5 是 24 层 GDN（Gated DeltaNet，线性注意力）+ 8 层 full-attention 的混合架构。

- **8 层 full-attention 的 RMSNorm/SwiGLU/RoPE**：Liger 直接 patch 上去，正常工作
- **24 层 GDN 的 RMSNorm**：Liger 没有 `LigerRMSNormForQwen3Next` 专用版本，swift 会跳过（不报错，只是这部分没融合）
- 启动时会有 warning：`The cross_entropy loss function defined in Liger Kernel will not take effect`——这是因为 Qwen3.5 task_type=causal_lm 用 swift 自己的 CE，不用 Liger CE。**忽略即可，不影响其他 kernel 融合**

如果将来 Liger 出了 GDN 版本，预计还能再省 5-10% step time。

---

## 7. 失败的尝试

学习意义大于成功案例。每条都告诉你**这条路不要走**或**为什么走不通**。

### 7.1 ❌ torch.compile（torch 2.10 + Inductor + TF32 API 冲突）

**预期收益**：融合更多 kernel，砍 launch overhead，wall −15-20%

**实测**：

```bash
TORCH_COMPILE=true bash scripts/benchmark/bench_swift_sp_v2.sh
# 结果：rank 4 OOM-style 崩
# Error: torch._inductor.exc.InductorError: RuntimeError: PyTorch is checking
# whether allow_tf32_new is enabled for cuBlas matmul, Current status indicate
# that you have used mix of the legacy and new APIs to set the TF32 status
# for cublas matmul
```

**根因**：torch 2.10 引入了新的 `allow_tf32_new` API，Inductor 用新 API 查询。但 swift（或某个依赖）用旧 API `torch.backends.cuda.matmul.allow_tf32 = True` 设置过。混用导致 Inductor 报错。

**workaround 我没试出来的**：
- `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` 在 torch 2.10 没用
- 改 `torch._inductor.config.fallback_random` 也无效
- 真正的 fix 估计要等 torch 2.11 或手动 patch swift 不调旧 API

**结论**：本机环境下 compile 不可用。等 torch 2.11 / swift 升级。

### 7.2 ❌ NCCL NVLS 算法（FSDP2 小消息不适合）

**预期**：H100 SXM 有 NVSwitch，NVLS（NV Link Sharp）算法能 in-network 做 reduce，砍一半流量

**实测**：

```bash
NCCL_ALGO=NVLS,Ring \
NCCL_PROTO=Simple,LL,LL128 \
NCCL_NVLS_ENABLE=1 \
NCCL_BUFFSIZE=8388608 \
NCCL_MIN_NCHANNELS=16 \
... bash scripts/benchmark/bench_swift_sp_v2.sh
# step time: 3.34 s (vs pack_liger 2.88 s, +16%)
```

**反而慢了 16%**。

**根因**：NVLS 算法适合 ≥几 MB 的大消息（典型场景：Megatron TP 的 grad AllReduce）。FSDP2 是 **per-parameter shard**：每个 transformer 子模块独立 AllGather，每次消息几十 KB-几 MB。NVLS 在小消息上 setup overhead 大于带宽收益，整体反而慢。

**结论**：FSDP2 + 小消息场景，**NCCL 默认（自动选 Ring/Tree）已经最优**。不要手工设 NCCL_ALGO。

### 7.3 ❌ MBS=4（peak mem 撞墙）

```bash
MBS=4 bash scripts/benchmark/bench_swift_sp_v2.sh
# step time: 30.6 s (vs MBS=2 17.6 s, vs MBS=2+packing 3.88 s)
# peak mem: 75.1 GiB (80 GiB 极限)
# NCCL 时间爆到 5.25 s
```

peak mem 75 / 80 GB，FSDP2 在内存压力下 NCCL 持续慢路径。再大就直接 OOM。

**结论**：H100 80 GB + Qwen3.5-9B + seq=16k 下，MBS 上限是 2。

### 7.4 ❌ wrap_large（SIZE_BASED_WRAP）

```bash
FSDP_WRAP_POLICY=SIZE_BASED_WRAP FSDP_MIN_NUM_PARAMS=100000000 \
bash scripts/benchmark/bench_swift_sp_v2.sh
# 启动崩溃，rank 2 exit 1
```

预期：用 size-based wrap 把多个 transformer block 合成一个 FSDP unit，AllGather 次数从 98 降到 ~20。但 swift / accelerate 的 SIZE_BASED_WRAP 实现有 bug（具体 trace 没深挖）。

**结论**：保持 `TRANSFORMER_BASED_WRAP`（默认）。

### 7.5 ❌ SP=1 + MBS=2（OOM）

不开 Ulysses SP，每 rank 处理完整 seq=16k × MBS=2 = 32k token 的 activation。OOM。

**结论**：seq=16k × 9B 模型场景下，**SP=2 是必需的**（不为速度，为放下）。

### 7.6 ❌ dataloader_num_workers=8 + 预分词（CPU 已不是瓶颈）

```bash
DATALOADER_WORKERS=8 DATASET_NUM_PROC=16 LAZY_TOKENIZE=false \
PACKING=true USE_LIGER=true \
bash scripts/benchmark/bench_swift_sp_v2.sh
# step time: 2.87 s (pack_liger 2.88 s) - 几乎一样
```

**结论**：packing 让 dataloader 工作量大幅降低（每 step 只读 16 个 packs vs 16 个 samples），CPU 早已不是瓶颈。**保持默认 num_workers=2 即可**。

---

## 8. 三后端对比

### 8.1 同样优化能套到 DS 和 Megatron 吗？

| 优化轴 | FSDP2 | DS ZeRO-3 | Megatron TP=4 SP |
|---|:---:|:---:|:---:|
| `--packing true` | ✅ | ✅ 但 loss=0 bug 挡死 | ✅ **已默认开** |
| `--use_liger_kernel true` | ✅ | ✅ | ❌ Megatron 用 TransformerEngine |
| MBS=2 | ✅ | ⚠️ 需 `train_micro_batch_size_per_gpu=auto`（见 [zero3_nopin_auto.json](../scripts/benchmark/sp_offload_configs/zero3_nopin_auto.json)） | ⚠️ 没试，可能 OOM |
| NO_AC | ✅ | ✅（用 `--gradient_checkpointing false`） | ✅（用 `RECOMPUTE=none`） |
| reshard_after_forward=false | ✅ | ❌ 等价物是改 ZeRO-2，但 mem ×3 | N/A |

### 8.2 DS：loss=0 bug 实测

```bash
BACKEND=ds SP=2 MBS=2 \
PACKING=true USE_LIGER=true \
DS_CONFIG=scripts/benchmark/sp_offload_configs/zero3_nopin_auto.json \
bash scripts/benchmark/bench_swift_sp_v2.sh
```

结果：step 2.77 s（看着挺好），但 loss 从 step 2 开始恒 0、grad_norm 恒 √2≈1.4142（HF Trainer 对 NaN grad 的安全替代值）。**模型不更新**。

```
{"step": 1, "loss": 12.91, "grad_norm": 13.07}
{"step": 2, "loss": 0.00,  "grad_norm": 1.41421356}
{"step": 3, "loss": 0.00,  "grad_norm": 1.41421356}
... 99/100 steps loss=0 ...
```

这是 modelscope 还没修干净的 DS ↔ Ulysses SP 集成 bug。**当前 DS 不可用**，不管套什么优化。等 modelscope 后续 PR。

### 8.3 Megatron：本来就不慢

```bash
USE_MEGATRON_BACKEND=true \
MODEL=/root/.cache/huggingface/models/Qwen/Qwen3.5-9B \
TP=4 PP=1 CP=1 MBS=1 GBS=8 MAX_LEN=16384 \
TOTAL_STEPS=100 WARMUP_BENCH=10 \
RECOMPUTE=none FREEZE_VIT=true \
BENCH_DIR=/home/ubuntu/perf_opt/megatron_output/walkthrough/mega \
bash scripts/benchmark/bench_megatron.sh
```

实测（100 步）：

| 指标 | Megatron TP=4 SP + packing | FSDP2 pack_liger | 比较 |
|---|---:|---:|---|
| step time | 2.20 s | 2.88 s | Megatron 更短 |
| 每步真实 tokens | 131 k | **262 k** | FSDP2 ×2 |
| tokens/s/GPU | 7 447 | **11 360** | **FSDP2 +53%** |
| peak power | —（gpu_monitor 1Hz 未捕获） | 706 W | — |
| 真 MFU（nsys） | **51.1%** | 50.4% | 打平 |
| **full-epoch wall** | 14.9 min | **9.8 min** | **FSDP2 快 1.5×** |

**两后端真 MFU 打平在 ~50%**。FSDP2 快 1.5× 不是单卡更强，是：
1. 有效 DP 大：FSDP2 SP=2 → DP=4；Megatron TP=4 → DP=2（TP 把 4 卡绑成一组）
2. GEMM tile 大：FSDP2 每卡算 d_model × d_model（4096 × 4096）；Megatron TP=4 切到 d_model × d_model/4（4096 × 1024），cuBLASLt WGMMA 在 K<2048 时效率掉 30-40%

### 8.4 什么时候用 Megatron

- **必须用 pypi swift 4.1.2**（不能装 git main）：FSDP2+SP 的 loss=0 bug 没修，只有 Megatron 路稳定
- **更大模型**：30B / 70B 在 8×H100 上用 FSDP2 ZeRO-3 + SP 内存够呛，Megatron TP=8 是更好的选择
- **更长序列**：Megatron CP（context parallel）能切到 128k+；FSDP2 + Ulysses SP 一般到 32k 就吃力。但 mcore_bridge 当前对 Qwen3.5 hard-asserted `context_parallel_size == 1`，CP 暂时走不通

---

## 9. 最终配置 & 一键生产训练命令

### 9.1 最终推荐：`pack_liger`

```
BACKEND=fsdp2     # FSDP2，PyTorch 原生 DTensor 实现
SP=2              # Ulysses sequence parallel，每对 rank 共享 seq 维
MBS=2             # 每张卡每步 2 个 packs
NO_AC=true        # 关 activation checkpointing（内存够，不需要 recompute）
FSDP_RESHARD=false  # FSDP2 ZeRO-2 模式：forward 后保留 unsharded params
PACKING=true      # 把多 short samples 拼到 max_len，消除 padding waste
USE_LIGER=true    # Triton 融合 RMSNorm + SwiGLU + RoPE
```

预期效果（实测）：
- 1 epoch wall: **9.8 min**（baseline 37 min 的 26%）
- peak mem: 59 GiB
- peak power: 706 W（99% TDP）
- 真 MFU: ~50%
- 30 步长跑 0 NaN，loss 正常下降

### 9.2 快速复现（接 bench 脚本，30 秒拉起）

```bash
docker exec fsdp_sft bash -lc "cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  MASTER_PORT=29555 BACKEND=fsdp2 SP=2 MBS=2 \
  NO_AC=true FSDP_RESHARD=false \
  PACKING=true USE_LIGER=true \
  TOTAL_STEPS=400 WARMUP_BENCH=5 \
  RUN_NAME=prod \
  BENCH_DIR=/home/ubuntu/perf_opt/megatron_output/run \
  bash scripts/benchmark/bench_swift_sp_v2.sh"
```

`TOTAL_STEPS=400` 大约对应 dataset 一遍（3252 packs / GBS=16 ≈ 203 steps，加上 warmup ~250 ≈ 一 epoch；要多个 epoch 改这里或加 `--num_train_epochs`）。

### 9.3 直接调 swift sft（不走 bench wrapper，纯生产）

`bench_swift_sp_v2.sh` 实质是把 env 转成 swift CLI flag。下面是**展开后的完整命令**（粘贴可直接跑）：

```bash
# 1. 先生成 fsdp_override.json（NO_AC + no_reshard）
mkdir -p /home/ubuntu/perf_opt/megatron_output/prod
cat > /home/ubuntu/perf_opt/megatron_output/prod/fsdp_override.json <<'EOF'
{
    "fsdp": "full_shard auto_wrap",
    "fsdp_config": {
        "fsdp_version": 2,
        "reshard_after_forward": false,
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "cpu_ram_efficient_loading": true,
        "state_dict_type": "SHARDED_STATE_DICT",
        "activation_checkpointing": false
    }
}
EOF

# 2. 生产训练命令
docker exec fsdp_sft bash -lc '
NPROC_PER_NODE=8 MASTER_PORT=29555 \
swift sft \
    --model /root/.cache/huggingface/models/Qwen/Qwen3.5-9B \
    --model_type qwen3_5 \
    --dataset /home/ubuntu/perf_opt/megatron-sft-recipes/sft-data/train.jsonl \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --max_length 16384 \
    --truncation_strategy delete \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --packing true \
    --use_liger_kernel true \
    --freeze_vit true --freeze_aligner true \
    --fsdp /home/ubuntu/perf_opt/megatron_output/prod/fsdp_override.json \
    --sequence_parallel_size 2 \
    --dataloader_num_workers 2 \
    --dataset_num_proc 4 \
    --logging_steps 5 \
    --save_steps 200 \
    --save_total_limit 3 \
    --save_strategy steps \
    --eval_strategy no \
    --output_dir /home/ubuntu/perf_opt/megatron_output/prod
'
```

每个 flag 的含义（高亮关键的）：

| flag | 值 | 含义 |
|---|---|---|
| `--torch_dtype bfloat16` | bf16 | H100 上 bf16 是甜点；不要 fp16 |
| `--attn_impl flash_attention_2` | FA2 | packing 必需用 varlen 路径，FA2 / sdpa 都行 |
| `--max_length 16384` | 16k | seq 上限；packing 会填到这个长度 |
| `--truncation_strategy delete` | 删除 | 单条样本若超 16k 直接丢，不截断 |
| `--per_device_train_batch_size 2` | 2 | MBS=2，每卡每步处理 2 个 packs |
| `--gradient_accumulation_steps 1` | 1 | 不需要 accum；GBS = 2 × 8 = 16 packs 已够 |
| `--packing true` | true | **核心**：拼 short samples 到 max_len |
| `--use_liger_kernel true` | true | **核心**：融合 RMSNorm/SwiGLU/RoPE |
| `--freeze_vit true --freeze_aligner true` | freeze | VLM 模型只训文本骨干 |
| `--fsdp <path>` | json | FSDP override（NO_AC + reshard=false） |
| `--sequence_parallel_size 2` | 2 | Ulysses SP=2，把 seq 维切两半（all-to-all） |
| `--dataloader_num_workers 2` | 2 | packing 后 CPU 不忙，2 够用 |

### 9.4 监控

跑训练的同时另起 dcgm-exporter（前面 0.2 已起）+ 一个简单的 watch：

```bash
# 显存 + 功耗实时
watch -n 1 nvidia-smi

# 看 DCGM tc_active（要 dcgm-exporter 已起）
watch -n 1 'curl -s localhost:9500/metrics | grep -E "PIPE_TENSOR_ACTIVE\{gpu=\"0\"" | head -1'

# swift log（在另一终端）
tail -f /home/ubuntu/perf_opt/megatron_output/prod/v0-*/logging.jsonl
```

### 9.5 Profile 验证（确认还在 50% MFU）

每次重大配置变更后，建议跑一次 profile 复核真 MFU 没退化：

```bash
# 短跑 + nsys 采 10s
docker exec fsdp_sft bash -lc "cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  TOTAL_STEPS=80 PROFILE_STEP=5 \
  BACKEND=fsdp2 SP=2 MBS=2 NO_AC=true FSDP_RESHARD=false \
  PACKING=true USE_LIGER=true \
  RUN_NAME=verify \
  BENCH_DIR=/tmp/verify \
  bash scripts/benchmark/bench_swift_sp_profile.sh"

# 看 analysis.txt 里的 GEMM%、FlashAttn%
cat /tmp/verify/verify/analysis.txt
# 期望：GEMM ~41%, FlashAttn ~3%, NCCL ~26%
```

---

## 附录 A：完整数据表

所有 14 个测点的 raw data 已自动汇总到：

```bash
python scripts/benchmark/build_all_runs_table.py
# → megatron_output/bench_fsdp_opt/_all_runs.md
```

按 7 个实验 block 分组（baseline / 单轴 / 失败 / 组合 / wall 验证 / packing+liger / 三后端 / profile）。每行包括：steady step、peak mem、tokens/s nominal+real、full-epoch wall、swift MFU、DCGM tc busy、avg/peak power、loss、status。

---

## 附录 B：关键文件清单

### 脚本

| 文件 | 作用 |
|---|---|
| [scripts/benchmark/bench_swift_sp_v2.sh](../scripts/benchmark/bench_swift_sp_v2.sh) | 主 bench wrapper，所有 env 旋钮入口 |
| [scripts/benchmark/bench_swift_sp_profile.sh](../scripts/benchmark/bench_swift_sp_profile.sh) | 短跑 + torch.profiler 一步精捕 |
| [scripts/benchmark/swift_sft_profiled.py](../scripts/benchmark/swift_sft_profiled.py) | swift CLI wrapper，注入 torch.profiler callback |
| [scripts/benchmark/torch_profile_callback.py](../scripts/benchmark/torch_profile_callback.py) | TrainerCallback 实现，签名保留是关键 |
| [scripts/benchmark/analyze_torch_profile.py](../scripts/benchmark/analyze_torch_profile.py) | torch.profiler chrome trace → kernel 分类 |
| [scripts/benchmark/dcgm_scrape.py](../scripts/benchmark/dcgm_scrape.py) | DCGM 10 Hz 抓取，写 dcgm_tc.tsv |
| [scripts/benchmark/run_fsdp_opt_sweep.sh](../scripts/benchmark/run_fsdp_opt_sweep.sh) | 10 配置 × 2 阶段（train + profile）扫盘 |
| [scripts/benchmark/build_fsdp_opt_summary.py](../scripts/benchmark/build_fsdp_opt_summary.py) | 扫盘结果聚合（早期版本，只覆盖 10 配置） |
| [scripts/benchmark/build_all_runs_table.py](../scripts/benchmark/build_all_runs_table.py) | **全部 14 测点聚合**（推荐用这个） |
| [scripts/benchmark/bench_megatron.sh](../scripts/benchmark/bench_megatron.sh) | Megatron 后端 bench（默认带 packing） |

### 配置 / 文档

| 文件 | 作用 |
|---|---|
| [scripts/benchmark/sp_offload_configs/zero3_nopin.json](../scripts/benchmark/sp_offload_configs/zero3_nopin.json) | DS ZeRO-3，MBS 写死 1 |
| [scripts/benchmark/sp_offload_configs/zero3_nopin_auto.json](../scripts/benchmark/sp_offload_configs/zero3_nopin_auto.json) | DS ZeRO-3，MBS=auto（用于 MBS>1 的对比测试） |
| [docs/sp_offload_benchmark_summary.md](sp_offload_benchmark_summary.md) | 精简结论文档（用于汇报） |
| [docs/sp_offload_benchmark_report.md](sp_offload_benchmark_report.md) | 完整报告（早期 7 配置矩阵 + 踩坑细节） |
| **本文档** | 学习 + 复现 walkthrough |

### 原始数据目录

| 路径 | 内容 |
|---|---|
| `megatron_output/bench_fsdp_opt/baseline/` | tier1 baseline 训练 + DCGM |
| `megatron_output/bench_fsdp_opt/{no_ac,mbs2,mbs4,no_reshard,combo_easy}/` | 各单轴 + combo 训练 |
| `megatron_output/bench_fsdp_opt/{pack,pack_liger,pack_liger_dl}/` | packing + Liger 各阶段 |
| `megatron_output/bench_fsdp_opt/pack_liger_nsys/profile.nsys-rep` | **pack_liger nsys 2025 trace（apples-to-apples 对比基准）** |
| `megatron_output/bench_fsdp_opt/mega/megatron/` | Megatron TP=4 SP 100 步训练 |
| `megatron_output/bench_fsdp_opt/mega_prof/megatron/profile.nsys-rep` | **Megatron nsys 2025 trace** |
| `megatron_output/bench_fsdp_opt/_all_runs.md` | 自动汇总表 |

---

## 写在最后

这套优化路径在 **Qwen3.5-9B × seq=16384 × 8×H100 × dataset=18819 多轮对话**这个具体配置下，把 wall time 从 37 min 压到 9.8 min（3.8×），真 MFU 从 2.3% 提到 50.4%。

**不是所有结论都通用**：
- 换更大模型（30B+），FSDP2 通信开销会增大，Megatron TP 优势会显现
- 换更长序列（>32k），Ulysses SP 不够用，Megatron CP 优势出来
- 换更短/更长 + 更密的 dataset（每条都接近 max_len），packing 收益会缩小（因为 baseline 也没那么多 padding）
- 换更新的 torch（>=2.11），torch.compile 这条路可能打通
- FP8 没试，理论上能再砍 30-40%（H100 FP8 是 bf16 的 2× 算力）

**通用的方法论**：
1. **永远先看物理证据**（DCGM peak power、avg power）。功耗低 = GPU 在睡觉，单看公式 MFU 会被骗
2. **profile 工具至少两条**（torch.profiler + nsys），交叉验证；DCGM 当快速旁证
3. **MFU 口径必须固定**，跨配置比较只用 per-rank GEMM/wall 一种口径
4. **每个优化轴单独跑一遍**，再叠组合。直接堆叠会让你不知道是哪条在起作用、哪条在拖后腿
5. **wall time 不变但 MFU 涨了** = GPU 不是瓶颈，要找上游（CPU dataloader / padding / IO）
