# Megatron TP=4 SP 优化实战 walkthrough（Qwen3.5-9B）

> 与 [`fsdp2_optimization_walkthrough.md`](fsdp2_optimization_walkthrough.md) 配套。本文专注 **Megatron / mcore_bridge** 后端在同样硬件 / 模型 / dataset 上的优化路径。
>
> 实验日期：2026-04-23/24 · 硬件 8×H100 80GB · Qwen3.5-9B (freeze ViT) · seq=16384 · ms-swift 4.2.0.dev0 · mcore_bridge 1.1.2

> **环境准备 / MFU 口径辨析 / DCGM & nsys 工具栈**：直接看 [fsdp2 doc §0-§3](fsdp2_optimization_walkthrough.md#0-环境准备)。本文只讲 Megatron 特有的部分。

---

## 目录

- [1. Baseline：TP=4 SP `recompute=none` + packing（已是优化版）](#1-baseline)
- [2. 为什么 baseline 默认就带 packing](#2-为什么-baseline-默认就带-packing)
- [3. 已实测有效的轴](#3-已实测有效的轴)
- [4. 失败 / 走不通的轴](#4-失败--走不通的轴)
- [5. 没试但有希望的轴（写给下一波）](#5-没试但有希望的轴写给下一波)
- [6. Megatron vs FSDP2 横向对比](#6-megatron-vs-fsdp2-横向对比)
- [7. 最终配置 & 一键生产训练命令](#7-最终配置--一键生产训练命令)

---

## 1. Baseline

### 1.1 起点：什么算 Megatron 的 baseline

Megatron 不像 FSDP2 那样有"全默认"配置——TP/PP/CP/recompute/packing 任一项都强烈影响性能。我们的 baseline 选择是**几次试错后能稳定跑长跑的配置**：

| 维度 | 值 | 选择理由 |
|---|---|---|
| TP（tensor parallel） | **4** | TP=2 baseline 装不下（实测 OOM） |
| PP（pipeline parallel） | 1 | 单节点 8 卡，PP overhead > 收益 |
| CP（context parallel） | 1 | mcore_bridge 对 Qwen3.5 hard-asserted CP=1（详见 §4.1） |
| `recompute_granularity` | **none** | 实测：none 比 selective / full 都快，且 peak mem 还够 |
| `packing` | true | 默认开（[bench_megatron.sh](../scripts/benchmark/bench_megatron.sh) line 145） |
| `distributed_optimizer` | true | 默认开 |
| `overlap_grad_reduce` / `overlap_param_gather` | true | 默认开，让 grad reduce 和 param gather 与 compute 并行 |
| MBS / GBS | 1 / 8 | TP=4 → 有效 DP=2，MBS=1 比 MBS=2 安全 |

### 1.2 启动命令

```bash
docker exec fsdp_sft bash -lc "cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  USE_MEGATRON_BACKEND=true \
  MODEL=/root/.cache/huggingface/models/Qwen/Qwen3.5-9B \
  TP=4 PP=1 CP=1 MBS=1 GBS=8 MAX_LEN=16384 \
  TOTAL_STEPS=100 WARMUP_BENCH=10 \
  RECOMPUTE=none FREEZE_VIT=true \
  BENCH_DIR=/home/ubuntu/perf_opt/megatron_output/walkthrough/mega \
  bash scripts/benchmark/bench_megatron.sh"
```

转换后实际跑的 `megatron sft` 完整命令（line 134-164 of bench_megatron.sh）：

```bash
megatron sft \
    --model /root/.cache/huggingface/models/Qwen/Qwen3.5-9B \
    --dataset sft-data/train.jsonl \
    --save_safetensors true \
    --tensor_model_parallel_size 4 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --packing true \
    --max_length 16384 \
    --lr 1e-5 --min_lr 1e-6 \
    --lr_warmup_fraction 0.1 \
    --lr_decay_style cosine \
    --train_iters 100 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --recompute_granularity none \
    --use_distributed_optimizer true \
    --overlap_grad_reduce true \
    --overlap_param_gather true \
    --freeze_vit true --freeze_aligner true \
    --output_dir <BENCH_DIR>/ckpt \
    --save_steps 99999 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --no_save_optim true --no_save_rng true
```

### 1.3 Baseline 数据

| 指标 | 值 | 来源 |
|---|---:|---|
| steady step time | **2.20 s** | 100 步实测，跳过 compile cold start |
| peak mem | **47.7 GiB** | swift `memory(GiB)` |
| swift / megatron 公式 MFU | 37.9% | report.json `mfu_pct` |
| **真 MFU**（per-rank GEMM/wall, nsys 2025 实测） | **51.1%** | 见 §6 |
| avg power | 433 W | nvidia-smi 1Hz 平均 |
| GPU avg util | 69% | report.json |
| **full-epoch wall** | **14.9 min** | 53.1M real tokens / (7447 tok/s/GPU × 8) |
| loss 验证 | 1.57 → 1.34（100 步），grad_norm 4.96 → 1.55，0 NaN | bench log + `train.log` |

50 步长跑（早期实测）：loss 1.57 → 1.36，grad_norm 5.0 → 1.4，**全程 0 NaN**。Megatron 这条路是**当前唯一无需依赖 swift main 的稳定生产配置**。

### 1.4 关键观察

Megatron baseline 已经是**重度优化过的状态**：
- packing 默认开，所以没有 padding waste 问题（这一点和 FSDP2 baseline 完全相反）
- `recompute=none` 默认开（见 §3.1 选择过程）
- 通信和计算 overlap 默认开

所以"FSDP2 那 3.8× 加速"在 Megatron 上**没有等量空间**——大头都已经在 baseline 里了。

---

## 2. 为什么 baseline 默认就带 packing

`bench_megatron.sh` line 145：

```bash
megatron sft \
    ...
    --packing true \
    ...
```

这行**不是后加的**。从最初 commit 起就是默认值，原因：

1. **mcore_bridge 的 packing 实现成熟**：用 `cu_seqlens` + flash_attention varlen，和 FSDP2 用的是同一套底层；mcore 还提供 `--reset_position_ids` 和 `--reset_attention_mask` flag 处理 sample 边界（默认开）
2. **Megatron 训练范式天生就是大批量 + 长序列**：不带 packing 是对算力的浪费
3. **`--cross_entropy_loss_fusion true`**：CE loss 内核融合一并算 packed sample 的 token-level loss 归一化，正确处理 multi-sample boundaries

这意味着：**Megatron 没法通过"加 packing"来加速**——它一直在用 packing。所以本文不会出现 "packing 让 wall time 砍 2.8×" 这种章节。

如果你想验证 packing 带来的差异，可以临时加 `--packing false` 重跑（**没在本轮验证**，因为没意义），预期 step time 会变慢 ~2-3× 同时显存占用会大幅下降（因为很多步处理 padding）。

---

## 3. 已实测有效的轴

### 3.1 `RECOMPUTE=none`（关 activation 重算）

mcore 支持三种 recompute 粒度：

| 值 | 含义 | 内存代价 | 计算代价 |
|---|---|---|---|
| `none` | 不重算，全保留 activation | 高 | 0 |
| `selective` | 只对部分层重算（默认对 attention 层） | 中 | +15% FLOPs |
| `full` | 每个 transformer block 都重算 | 低 | +33% FLOPs |

**baseline 选 `none`**。证据来自早期完整报告：

| RECOMPUTE | step time | peak mem | 状态 |
|---|---:|---:|---|
| none（推荐） | 2.20 s | 47.7 GiB | ✅ |
| selective | 2.45 s（+11%） | 36 GiB | ✅ 但更慢 |
| full | 2.90 s（+32%） | 28 GiB | ✅ 但最慢 |

H100 80GB 下 `recompute=none` 显存还够（只占 60%），没必要为了省内存付出 11-32% 计算代价。

**例外场景**：如果你换更大模型（30B+），none 可能 OOM，那才考虑 selective。

### 3.2 `--use_distributed_optimizer true` + `--overlap_grad_reduce true` + `--overlap_param_gather true`

三个都默认开。它们的作用：

- **distributed_optimizer**：把 optimizer states（Adam 的 m / v 张量，每参 8 bytes fp32）切到多卡。等价 ZeRO-1。**省内存**，没速度代价
- **overlap_grad_reduce**：反向算到一半就启动 grad ReduceScatter，让通信和反向 compute 并行
- **overlap_param_gather**：分布式优化器更新后立即开始下一步的 param AllGather，让通信和下一步 forward 并行

这三个开关合起来贡献了 Megatron baseline 的 51.1% 真 MFU 里的"通信掩盖"部分。**不要关**。

### 3.3 `--cross_entropy_loss_fusion true`

CE loss 计算融合成一个 kernel，省掉 vocab=248320 维的 softmax + log + sum 中间张量。对 Qwen3.5 这种大 vocab 模型尤其重要：节省 ~2 GB activation memory + 几 ms 时间。**默认开，不要关**。

---

## 4. 失败 / 走不通的轴

### 4.1 ❌ Megatron CP（Context Parallel）on Qwen3.5

**预期收益**：CP 把 seq 维切到多卡，让长 seq 训练放得下 + 通信成本低于 Ulysses SP

**实测**：直接抛 hard assert error。

```python
# /usr/local/lib/python3.12/site-packages/mcore_bridge/model/mm_gpts/qwen3_5.py:26-29
class Qwen3_5MoeGatedDeltaNet(_HuggingFaceModule, _Qwen3_5MoeGatedDeltaNet):
    def __init__(self, config: TransformerConfig, submodules, layer_number, **kwargs):
        assert config.context_parallel_size == 1, \
            'Qwen3_5 currently does not support context parallel.'
```

**根因**：mcore_bridge 的 `Qwen3_5MoeGatedDeltaNet` 继承自 HuggingFace 的 `Qwen3_5MoeGatedDeltaNet`（用 `super().forward()` 借壳 HF 实现），**没接 mcore 自己的 CP-aware GDN module**。后者在 `mcore/modules/gated_delta_net.py:123` 有完整 `tensor_a2a_cp2hp` 实现，但只在 `qwen3_next` native path 用到。

**修复工作量**：把 `mm_gpts/qwen3_5.py` 里的 `Qwen3_5MoeGatedDeltaNet` 继承链从 HF 借壳换成 mcore 原生 GDN module，同时把 RoPE / RMSNorm 也接上。预估**半天到一天**，本轮没做。

**结论**：Megatron CP 对 Qwen3.5 当前**完全不可用**。`bench_megatron.sh` 保留了 `CP=N` env var，等 mcore_bridge 修好后可直接用。

### 4.2 ❌ TP=2 + `recompute=selective`（OOM 过）

**早期实测（不带 packing 时代）**：TP=2 PP=1 + selective recompute + distributed_optimizer，**Triton 报 CUDA OOM**。证据保留在早期报告 §5.4 + log 里。

**现在带 packing 后**：未实测。理论上 packing 不会让显存爆得更厉害（packing 减少 padding，但同时 activation 是真实 token 的），所以 TP=2 + packing 仍然可能 OOM。

如果想试，方案见 §5.1。

### 4.3 ❌ 装更老的 nsys 想 profile

`apt install nsight-systems` 装的是 **2021.3**（Ubuntu 22.04 仓库版本），**不识别 H100 的 WGMMA 指令**，profile 出来 GPU 部分基本是空的。必须用 CUDA toolkit 12.9 自带的 **nsys 2025.2.1**：

```bash
docker exec fsdp_sft ln -sf \
    /opt/nvidia/nsight-compute/2025.2.1/host/target-linux-x64/nsys \
    /usr/local/bin/nsys
```

`bench_megatron.sh` 第 99 行已经引用 2025 路径，加 `PROFILE=true` 就能起 nsys 采 profile（详见 §6）。

---

## 5. 没试但有希望的轴（写给下一波）

### 5.1 TP=2 + SP（最有希望追平 FSDP2 pack_liger）

**为什么有希望**：
- FSDP2 pack_liger 比 Megatron baseline 快 1.5×，根因之一是 GEMM tile 大（FSDP2 算 d_model × d_model，Megatron TP=4 切到 d_model × d_model/4 = 4096 × 1024，cuBLASLt WGMMA 在 K<2048 时 TC 利用率掉 30-40%）
- TP=2 把 K 还原到 2048，**接近 cuBLASLt 最佳点**
- 同时 TP=2 → 有效 DP=4（vs TP=4 的 DP=2），每步处理 2× 样本

**预估收益**：
- step time: 2.20 → ~1.5 s
- 真 MFU: 51% → 60%+
- full-epoch wall: 14.9 → ~10 min（追平甚至超过 FSDP2 pack_liger）

**风险**：显存。baseline TP=2 OOM 过，但当时没带 packing。带 packing 后**需要实测**：

```bash
docker exec fsdp_sft bash -lc "cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  USE_MEGATRON_BACKEND=true \
  MODEL=/root/.cache/huggingface/models/Qwen/Qwen3.5-9B \
  TP=2 PP=1 CP=1 MBS=1 GBS=8 MAX_LEN=16384 \
  TOTAL_STEPS=20 WARMUP_BENCH=5 \
  RECOMPUTE=none FREEZE_VIT=true \
  BENCH_DIR=/tmp/mega_tp2 \
  bash scripts/benchmark/bench_megatron.sh"
# 看 train.log 里有没有 OOM。如果 OOM，试 RECOMPUTE=selective
```

### 5.2 MBS=2

TP=4 时 MBS=2 让每个 DP rank 处理 2 个 16k packs。Activation 翻倍。当前 baseline peak mem 47.7 GiB，翻倍到 ~70 GiB 接近 OOM 边界。

```bash
TP=4 MBS=2 GBS=16 ... bash bench_megatron.sh
```

如果不 OOM，预期 step time +60%、tokens/step +100%，net wall 缩短 ~25%。**没实测**。

### 5.3 FP8 训练（TransformerEngine）

H100 杀手锏。bf16 peak 989 TF/s → FP8 peak **1979 TF/s（×2）**。

Megatron 原生支持 FP8 via TransformerEngine：

```bash
megatron sft \
    ...
    --fp8_format e4m3 \           # 或 hybrid (e4m3 forward + e5m2 backward)
    --fp8_amax_history_len 1024 \
    --fp8_amax_compute_algo max \
    ...
```

需要：
- `transformer_engine >= 1.5`（容器里有，需要确认版本）
- mcore_bridge 把 nn.Linear 都换成 te.Linear（`Qwen3_5MoeGatedDeltaNet` 继承链里有没有 nn.Linear 是关键）
- amax 校准：first ~50 steps 收敛 amax，loss 数值会有波动

预估收益（**纯理论**）：GEMM 时间 ÷2，但 GDN scan 不能 FP8 化，所以 wall 缩短 25-30%（FSDP2 同样估算）。**没实测**，工作量约 1-2 天。

### 5.4 PP=2 / PP=4（多节点才有意义）

单节点 8 卡时，PP overhead > 收益。PP 适合：
- **多节点训练**（PP 跨节点的通信窗口大，能掩盖 inter-node bandwidth）
- **超大模型**（70B+ 不切 PP 单节点放不下）

8 卡 + 9B 场景下不要试。

---

## 6. Megatron vs FSDP2 横向对比

### 6.1 nsys 2025 实测（10s steady-state，8 ranks aggregate）

| 指标 | FSDP2 pack_liger | Megatron TP=4 SP+packing |
|---|---:|---:|
| **GEMM**（nvjet cuBLASLt） | 41.1% of agg | **46.4%** |
| **FlashAttn** | 3.4% | **4.9%** |
| **NCCL** | 26.1% | 29.1% |
| **GDN/fla（线性 attn，非 TC）** | 5.3% | 7.9% |
| Elementwise（RMSNorm/SwiGLU/RoPE 等） | **11.7%** | 1.9% |
| Memory shuffle（split_with_sizes / chunk_cat / cat） | **8.1%** | 2.8% |
| **Per-rank GEMM time / wall**（真 MFU 上界） | **50.4%** | **51.1%** |
| **Per-rank aggregate busy % wall** | 113%（async stream overlap） | 99.6%（较串行） |

### 6.2 真实吞吐（用户角度）

| 指标 | FSDP2 pack_liger | Megatron TP=4 SP | 谁赢 |
|---|---:|---:|---|
| step time | 2.88 s | 2.20 s | Megatron 单步快 |
| 每步真实 tokens | 262 k | 131 k | **FSDP2 ×2** |
| **tokens/s/GPU** | **11 360** | 7 447 | **FSDP2 +53%** |
| achieved TF/s per GPU（6ND 公式） | 614 | 402 | **FSDP2 +53%** |
| avg power | **519 W** | 433 W | **FSDP2 +20%** |
| **full-epoch wall** | **9.8 min** | 14.9 min | **FSDP2 快 1.5×** |

### 6.3 为什么真 MFU 打平但 throughput 差 1.5×

- **真 MFU 50% vs 51% 是"tensor core 忙时间占比"**，但**TC 在那段时间里实际产的 FLOPs 不一样**
- FSDP2 GEMM tile：seq × d_model × d_model = 16384 × 4096 × 4096，cuBLASLt 跑大 tile 接近 989 TF
- Megatron TP=4 GEMM tile：seq × d_model × d_model/4 = 16384 × 4096 × 1024，K=1024 让 WGMMA 利用率掉 30-40% → 实际只跑 ~600 TF
- 加上 FSDP2 MBS=2 + 有效 DP=4 → tokens/step 是 Megatron 的 2×

**所以 Megatron 在 9B × seq=16k × 8 卡这个点上比 FSDP2 慢，不是因为 Megatron 差**，而是 TP=4 切得太细。**Megatron 真正的优势在更大模型 / 更长序列**（见 §7.4）。

### 6.4 nsys profile 一行命令（验证你自己的环境）

```bash
NSYS=/opt/nvidia/nsight-compute/2025.2.1/host/target-linux-x64/nsys

# 直接走 bench_megatron.sh 的 PROFILE=true 路径
docker exec fsdp_sft bash -lc "cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  USE_MEGATRON_BACKEND=true \
  MODEL=/root/.cache/huggingface/models/Qwen/Qwen3.5-9B \
  TP=4 PP=1 CP=1 MBS=1 GBS=8 MAX_LEN=16384 \
  TOTAL_STEPS=50 WARMUP_BENCH=10 \
  RECOMPUTE=none FREEZE_VIT=true \
  PROFILE=true PROFILE_DELAY=170 PROFILE_DURATION=10 \
  BENCH_DIR=/tmp/mega_prof \
  bash scripts/benchmark/bench_megatron.sh"

# 解析
docker exec fsdp_sft nsys stats --report cuda_gpu_kern_sum \
    --format csv /tmp/mega_prof/megatron/profile.nsys-rep > /tmp/mega_kern.csv

# 分类（参考 analyze_torch_profile.py 的 classify() 函数）
python3 scripts/benchmark/analyze_torch_profile.py /tmp/mega_kern.csv
```

`PROFILE_DELAY=170` 是为 Megatron 模型加载 + compile cold start（约 17s）+ 几个 warmup steps 流出空间。具体延迟数依模型 / 数据规模而异，可以先 `PROFILE_DURATION=180` 跑一次画出时间线，再决定 delay。

---

## 7. 最终配置 & 一键生产训练命令

### 7.1 当前推荐：Megatron TP=4 SP `recompute=none` + packing

这就是 baseline，**没新东西要叠**。

### 7.2 复现命令（bench wrapper）

```bash
docker exec fsdp_sft bash -lc "cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  USE_MEGATRON_BACKEND=true \
  MODEL=/root/.cache/huggingface/models/Qwen/Qwen3.5-9B \
  TP=4 PP=1 CP=1 MBS=1 GBS=8 MAX_LEN=16384 \
  TOTAL_STEPS=400 WARMUP_BENCH=10 \
  RECOMPUTE=none FREEZE_VIT=true \
  BENCH_DIR=/home/ubuntu/perf_opt/megatron_output/prod_mega \
  bash scripts/benchmark/bench_megatron.sh"
```

### 7.3 直接调 `megatron sft`（生产用）

```bash
docker exec fsdp_sft bash -lc '
NPROC_PER_NODE=8 \
megatron sft \
    --model /root/.cache/huggingface/models/Qwen/Qwen3.5-9B \
    --dataset /home/ubuntu/perf_opt/megatron-sft-recipes/sft-data/train.jsonl \
    --save_safetensors true \
    --tensor_model_parallel_size 4 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --packing true \
    --max_length 16384 \
    --lr 1e-5 --min_lr 1e-6 \
    --lr_warmup_fraction 0.05 \
    --lr_decay_style cosine \
    --train_iters 400 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --recompute_granularity none \
    --use_distributed_optimizer true \
    --overlap_grad_reduce true \
    --overlap_param_gather true \
    --freeze_vit true --freeze_aligner true \
    --output_dir /home/ubuntu/perf_opt/megatron_output/prod_mega/ckpt \
    --save_steps 200 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --no_save_optim true --no_save_rng true
'
```

每个关键 flag 含义：

| flag | 值 | 作用 |
|---|---|---|
| `--tensor_model_parallel_size 4` | 4 | TP=4，4 卡为一个模型分片组 |
| `--sequence_parallel true` | true | TP 内开 sequence parallelism（**和 Ulysses SP 不是一回事**，是 mcore 的 LayerNorm/Dropout 层 seq 维分布优化） |
| `--micro_batch_size 1` | 1 | 每张卡每个 micro-step 处理 1 个 pack |
| `--global_batch_size 8` | 8 | 每个 train iter 处理 8 个 packs（= 8 × 16384 ≈ 131k tokens） |
| `--packing true` | true | **核心**：拼 short samples 到 max_len |
| `--cross_entropy_loss_fusion true` | true | CE 内核融合，省 ~2 GB activation |
| `--recompute_granularity none` | none | 不重算 activation（H100 80GB 够装） |
| `--use_distributed_optimizer true` | true | optimizer states 切片，省内存 |
| `--overlap_grad_reduce true` | true | grad reduce 与反向 compute 并行 |
| `--overlap_param_gather true` | true | param gather 与下一步 forward 并行 |
| `--freeze_vit / --freeze_aligner` | true | VLM 只训文本骨干 |
| `--no_save_optim / --no_save_rng` | true | checkpoint 不存 optim states 和 RNG state（省时省盘） |

### 7.4 什么时候选 Megatron 而不是 FSDP2 pack_liger

| 场景 | 推荐 | 理由 |
|---|---|---|
| **必须用 pypi swift 4.1.2**（不能装 git main） | Megatron | FSDP2+SP 在 swift 4.1.2 有 loss=0 bug |
| **9B × seq=16k × 8 卡** | **FSDP2 pack_liger** | 实测 wall 9.8 vs 14.9 min |
| **30B+ 模型** | **Megatron** | TP=8 把模型切下去，FSDP2 ZeRO-3 大模型通信会爆 |
| **seq > 32k** | **Megatron**（前提：mcore_bridge CP 修好） | Ulysses SP 在 32k+ 通信开销大于收益 |
| **必须长跑稳定（数千步）** | Megatron | 已实证 50 步无 NaN；FSDP2+SP 30 步实证（更长没测） |
| **必须用 FP8 提速** | Megatron | mcore + TE 集成完善，swift 这边没原生 FP8 SFT 支持 |

### 7.5 监控

```bash
# 主 log
tail -f /home/ubuntu/perf_opt/megatron_output/prod_mega/ckpt/v0-*/logging.jsonl

# 显存 + 功耗
watch -n 1 nvidia-smi

# DCGM tc_active (10 Hz)
watch -n 1 'curl -s localhost:9500/metrics | grep -E "PIPE_TENSOR_ACTIVE\{gpu=\"0\"" | head -1'
```

---

## 写在最后

Megatron 的 baseline 已经包含了 packing / no-recompute / distributed-optimizer / overlap 这些主要优化，所以本文比 [fsdp2 walkthrough](fsdp2_optimization_walkthrough.md) 短得多——不是 Megatron 简单，而是 **Megatron 的"调优"主要发生在选 TP/PP/CP 维度上**，单一维度内的 knob 不多。

剩下没榨的空间：**TP=2 + packing**（最可能追上 FSDP2）和 **FP8**（理论上能再砍 30%）。下次有 GPU 直接试 §5.1。

不要再去试 Megatron CP 直到 mcore_bridge 修了 Qwen3.5 的 hard assert（§4.1）。
