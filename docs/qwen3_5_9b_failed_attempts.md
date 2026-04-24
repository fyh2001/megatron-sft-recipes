# Qwen3.5-9B：那些"没跑起来"的尝试与教训

> 这是 [`fsdp2_optimization_walkthrough.md`](fsdp2_optimization_walkthrough.md) 的**史前史**。
>
> 在我们最终落地"`GBS=8 + SP=2 + pack_liger`，9.8 min/epoch"这个配置之前，先有一段两周左右**用大 batch + 长序列硬撞墙**的尝试。本文把那段失败记录整理出来，让你理解：
> - **为什么 GBS 从 384 降到了 8**
> - **为什么必须用 Ulysses SP=2**
> - **为什么 Megatron CP 被堵死**
> - **为什么中途想换模型到 Qwen2.5-14B 又转回来**

> 原始计划文档：[`qwen3_5_9b_benchmark_plan.md`](qwen3_5_9b_benchmark_plan.md)（保留未删，作为 ambition 的参考）

---

## 目录

- [1. 原始野心：GBS=384 × seq=16384 × 三后端 × 2h](#1-原始野心gbs384--seq16384--三后端--2h)
- [2. 撞墙 ① Qwen3.5 GDN 训练时 workspace ∝ seq](#2-撞墙--qwen35-gdn-训练时-workspace--seq)
- [3. 撞墙 ② transformers 后端的 SP 在 Qwen3.5 上根本不工作](#3-撞墙--transformers-后端的-sp-在-qwen35-上根本不工作)
- [4. 撞墙 ③ Megatron CP 被 mcore_bridge hard-asserted 关掉](#4-撞墙--megatron-cp-被-mcore_bridge-hard-asserted-关掉)
- [5. 撞墙 ④ fla 库本身也不靠谱](#5-撞墙--fla-库本身也不靠谱)
- [6. 中途换模型尝试：Qwen2.5-14B / Ministral-3-14B](#6-中途换模型尝试qwen25-14b--ministral-3-14b)
- [7. 最终决定：回 Qwen3.5-9B，把 GBS 砍到 8 + 加 SP=2](#7-最终决定回-qwen35-9b把-gbs-砍到-8--加-sp2)
- [8. 教训汇总](#8-教训汇总)

---

## 1. 原始野心：GBS=384 × seq=16384 × 三后端 × 2h

**计划**（见 [qwen3_5_9b_benchmark_plan.md §设计决定](qwen3_5_9b_benchmark_plan.md)）：

| 维度 | 值 |
|---|---|
| 模型 | Qwen3.5-9B（VLM，32 层 = 24 GDN + 8 full attn，hidden=4096，vocab=248320） |
| seq | **16 384**（原生支持 262k，不需 YaRN） |
| MBS / GAS / GBS | 1 / 48 / **384** |
| tokens/step | **6.29 M** |
| 三后端 | FSDP2+compile / DeepSpeed ZeRO-3 / Megatron TP=2 PP=1 |
| 步数 | 15 步（5 warmup + 10 measure） |
| 时间预算 | 2 小时（含 nsys profile） |

每步 6.29 M token，相比上一份 Qwen2.5-7B baseline 的 32k token/step 是 **~200×**。一次"实战级"的真实训练 SFT bench。

**预期结果**（计划里写的）：
> "FSDP/DS 边缘（~55-70 GB/卡），Megatron TP=2+SP 宽裕"

**实测后所有预期都翻了**。

---

## 2. 撞墙 ① Qwen3.5 GDN 训练时 workspace ∝ seq

### 2.1 现象

跑 smoke（GBS=8 × seq=16384 × MBS=1 × bf16 × FSDP2）—— **直接 OOM**。peak mem 显示 78.8 + 0.4 GiB（H100 80GB 容量上限）。

### 2.2 根因：GDN backward 不是 O(1) state

Qwen3.5-9B 的 text backbone 是 **hybrid 架构**：
- 24 层 **Gated DeltaNet**（线性注意力，复杂度 O(seq)）
- 8 层 full attention（O(seq²)）
- 每 4 层一次 full attn（`full_attention_interval=4`）

宣传上 GDN 是"O(seq) 复杂度，state 固定大小，能跑 262k 上下文"。**这只对推理成立**。

训练时 GDN 走 [fla 库](https://github.com/fla-org/flash-linear-attention) 的 `chunk_gated_delta_rule` kernel。它把 seq 切成 chunk（典型 256 一块），forward 时按 chunk 顺序累加 state。**backward 时需要把所有 chunk 的中间 state 同时保留在显存里**：

```python
# fla/ops/common/chunk_delta_h.py:690 (lines may differ in newer versions)
h = k.new_empty(B, NT, H, K, V)         # 比如 [1, 256, 16, 128, 128] = 270 MB / 层
v_new = torch.empty_like(u)              # backward 专用 scratch
```

24 层 GDN 同时 alloc 这些 state，**workspace 整体仍然 ∝ seq**，远不是 "fixed size"。

### 2.3 实测 seq 扫描（FSDP 单卡上限）

`MBS=1 × NPROC=8 × bf16 × flash_attention_2`：

| seq | GRAD_CKPT=false | GRAD_CKPT=true |
|---:|---|---|
| 2 048 | ✅ OK | — |
| 4 096 | ✅ OK | — |
| 5 120 | ❌ OOM (peak 64 + 4.7 GiB) | — |
| 8 192 | ❌ OOM | ⚠️ 走 transformers 原生 ckpt 才能不踩 FSDP plugin metadata bug |
| 16 384 | ❌ OOM (78.8 + 0.4 GiB) | ✅ peak ~67 GB（compile=false） |

**结论 1**：FSDP × MBS=1 × seq=16384 单 rank 装不下 9B + GDN workspace。**差 ~400 MB 而已**——是结构性的，再优化 allocator 也救不回来。

**结论 2**：必须开 `gradient_checkpointing=true`（用计算换显存），但这意味着每个 transformer block 反向时**多做一次 forward**，FLOPs +33%。

### 2.4 推理 vs 训练的常见误解

社区分析（Sebastian Raschka / AI Business Dispatch / HF 模型卡讨论）反复说 GDN "state 固定、能跑 long context"。这是对的——**对推理**。
- **推理**：generation 是 token-by-token autoregressive，state 永远只持有 last chunk 的累加值，O(1) 不变
- **训练**：chunk-parallel backward 需要**所有 chunk 同时存在**才能算 grad，O(seq) workspace 不可避免

如果你看到任何"GDN 训练能跑 100k"的声明，先确认它是不是 LoRA / 极小 batch / 多机才做到的。

---

## 3. 撞墙 ② transformers 后端的 SP 在 Qwen3.5 上根本不工作

### 3.1 现象

为了绕开 §2 的 OOM，自然想到加 **Ulysses SP=2** 把 seq 维度切两半，每 rank 只看 seq=8192。结果：

```bash
swift sft --sequence_parallel_size 2 ...  # ms-swift 4.1.2
# → 第一个 forward 直接 TypeError 崩
```

具体错误：

```
TypeError: SequenceParallel._prepare_flash_attn.<locals>.flash_attention_mask()
    missing 1 required positional argument: 'cache_position'
```

### 3.2 根因：transformers 5.5 的 mask API 改了，swift SP 没跟

ms-swift 4.1.2 的 [`swift/sequence_parallel/ulysses.py`](https://github.com/modelscope/ms-swift) line 190 用旧签名 monkey-patch `flash_attention_mask`：

```python
# swift 4.1.2 的旧签名
def flash_attention_mask(batch_size, cache_position, kv_length, ...):
    ...
```

但 transformers 5.5.4 已经改成 kwargs-only 新签名：

```python
def flash_attention_mask(
    batch_size, q_length, kv_length,
    q_offset, kv_offset, mask_function, attention_mask, ...
):
    ...
```

`cache_position` 被 `q_offset/kv_offset` 拆开。swift 的 monkey-patch 没用，转发参数对不上→ `TypeError`。

### 3.3 上游 evidence

- **[ms-swift #8181](https://github.com/modelscope/ms-swift/issues/8181)**：用户报告 Qwen3.5 + sequence_parallel_size 在 transformers 后端崩。modelscope 官方回复："**please use megatron-swift**"——意思是这条路他们不修，让你改用 Megatron 后端。
- **[ms-swift 官方 examples/models/qwen3_5/mcore.sh](https://github.com/modelscope/ms-swift/blob/main/examples/models/qwen3_5/mcore.sh)**：连 35B 的官方示例都只敢用 `max_length=2048 + LoRA rank=8 + recompute_granularity=full + sequence_parallel=true`，没有"full-parameter + seq=16384"的成功案例
- **HF model card**："If you encounter out-of-memory (OOM) errors, consider reducing the context window."（官方自己承认 long ctx 训练会 OOM）

### 3.4 后来怎么解决

写了一份补丁：[`scripts/benchmark/swift_sp_patch.py`](../scripts/benchmark/swift_sp_patch.py)，在 swift 加载前 monkey-patch `SequenceParallel._prepare_flash_attn`，把三个 mask hook（`flash_attention_mask` / `sdpa_mask` / `create_causal_mask`）的签名对齐 transformers 5.5。

入口：[`scripts/benchmark/swift_sft_patched.py`](../scripts/benchmark/swift_sft_patched.py) 是 `swift sft` 的 wrapper，先 import patch 再调 `sft_main()`。

后来（2026-04-21/23）modelscope 自己也修了：
- [PR #9162](https://github.com/modelscope/ms-swift/pull/9162) Qwen3.5 GDN SP 集成
- [PR #9167](https://github.com/modelscope/ms-swift/pull/9167) transformers 5.4+ mask API 兼容
- [PR #9189](https://github.com/modelscope/ms-swift/pull/9189) Qwen3.5 SP bugfix

合到 `ms-swift main`，但 pypi 4.1.2 依然没修。**当前如果你用 pypi 版，必须用我们这个 swift_sp_patch.py 走 `swift_sft_patched.py` 入口**。升 git main 后这个 shim 不再需要。

详细情况见 [`sp_offload_benchmark_report.md` §5.1 / §10](sp_offload_benchmark_report.md)。

---

## 4. 撞墙 ③ Megatron CP 被 mcore_bridge hard-asserted 关掉

### 4.1 现象

为了支持 long seq 训练，自然想到 **Megatron 的 Context Parallel (CP)**。它把 seq 维度切到多卡，每 rank 只看部分 seq，结合 ring-attention 跨 rank 拼起来。理论上能切到 128k+。

跑一下：

```bash
USE_MEGATRON_BACKEND=true CP=2 ... bash bench_megatron.sh
```

直接抛 assert：

```python
# /usr/local/lib/python3.12/site-packages/mcore_bridge/model/mm_gpts/qwen3_5.py
class Qwen3_5MoeGatedDeltaNet(_HuggingFaceModule, _Qwen3_5MoeGatedDeltaNet):
    def __init__(self, config: TransformerConfig, submodules, layer_number, **kwargs):
        assert config.context_parallel_size == 1, \
            'Qwen3_5 currently does not support context parallel.'
```

### 4.2 根因

mcore_bridge 1.1.2 的 `Qwen3_5MoeGatedDeltaNet` 继承自 HuggingFace 的 `Qwen3_5MoeGatedDeltaNet`（直接借壳 HF 实现，用 `super().forward()` 转发），**没接 mcore 自己的 CP-aware GDN module**。

mcore 自己的 `modules/gated_delta_net.py:123` **有**完整 `tensor_a2a_cp2hp` 实现（CP 模式下 token 跨 rank 的 all-to-all 切换），但只在 `qwen3_next` native path 里用——`qwen3_5` bridge 没接进去。

### 4.3 修复工作量

把 `mm_gpts/qwen3_5.py` 的 `Qwen3_5MoeGatedDeltaNet` 继承链从 HF 借壳换成 mcore 原生 GDN module，同时把 RoPE / RMSNorm 也接上。**估计半天到一天**。本轮没做。

`bench_megatron.sh` 保留了 `CP=N` env var，等 mcore_bridge 修了直接用。当前**Megatron 这条路在 Qwen3.5 上 CP 完全不可用**，只剩 TP/PP/SP（mcore 原生 sequence parallel，不是 Ulysses 那种 SP）。

---

## 5. 撞墙 ④ fla 库本身也不靠谱

为兜底也想过自己接 fla 高阶 API：

| issue | 内容 |
|---|---|
| [fla #241](https://github.com/fla-org/flash-linear-attention/issues/241) | Gated DeltaNet 在某些 head_dim 下 shared memory 溢出，需要手动降 `BV` block_size |
| [fla #790](https://github.com/fla-org/flash-linear-attention/issues/790) | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 在 Blackwell 上 autotune 选错 config，**silent 精度发散** |

fla 库本身在 GDN 长序列这块**还在快速迭代、时不时出 silent bug**。如果碰上 silent 精度问题，loss 看着正常但模型实际收敛到错误的解。**风险大于收益**，不深入追求。

---

## 6. 中途换模型尝试：Qwen2.5-14B / Ministral-3-14B

### 6.1 想法：跳过 GDN 这堆问题

既然 Qwen3.5 的 GDN 这么麻烦，换一个 dense transformer 模型不就好了？候选：

| 候选 | 参数量 | 架构 | 问题 |
|---|---|---|---|
| Mistral / Mistral-7B-Instruct | 7B | dense | 太小，没意思 |
| **Ministral-3-14B-Instruct-2512** | 14B | dense | 1) 只发 FP8 格式，需要 dequantize 到 bf16；2) **mcore_bridge 1.1.2 没有 mistral3 bridge**，Megatron 这一栏直接 DOA |
| **Qwen2.5-14B-Instruct** | 14.77B | dense（48 层 hidden=5120） | 满足条件 |

选 Qwen2.5-14B 写了完整修订计划：[`qwen2_5_14b_benchmark_plan.md`](qwen2_5_14b_benchmark_plan.md)。

### 6.2 14B smoke 实测：单卡同样塞不下

跑 FSDP smoke：

| seq | GRAD_CKPT=false | GRAD_CKPT=true |
|---:|---|---|
| 2 048 | ✅ 561 ms/step | — |
| 4 096 | ✅ 1324 ms/step | — |
| 5 120 | ❌ OOM (64 + 4.7 GiB) | — |
| 16 384 | ❌ OOM (78.8 + 0.4 GiB) | ✅ peak 67.9 GB（compile=false）/ 53.8 GB（compile=true） |

14B 的 OOM 模式和 9B GDN 几乎一样：seq=16384 × MBS=1 × no-ckpt **差 ~400 MB**，必须开 ckpt。

**讽刺的是**：换到 14B 也没解决问题。dense model 没 GDN workspace 问题，但模型本身大 1.5 倍，单卡 activation 仍然爆。**最终都得开 grad checkpointing 或 SP**。

### 6.3 Megatron 14B 实测：发现 TP=4+SP 是甜点

Megatron 14B 调参扫描（GBS=8 × seq=16384 × 4 步，**实测**）：

| Config | step time | peak mem | 状态 |
|---|---:|---:|---|
| TP=2 + selective recompute | — | — | ❌ OOM |
| TP=2 + full recompute | 14.85 s/it | 56.5 GiB | ✅ |
| TP=4 + full recompute | 17.42 s/it | 42.5 GiB | ✅（但 +17% 慢） |
| TP=4 + selective recompute | 13.81 s/it | 53.1 GiB | ✅ |
| **TP=4 + no recompute** | **13.03 s/it** | **50.8 GiB** | **✅ 最优** |
| TP=4 + selective recompute + MBS=2 | — | — | ❌ OOM |

**重要发现**（后来直接搬到 Qwen3.5-9B 配置上）：
- 原 plan 假设 TP=2 是 sweet spot，**实测在 seq=16384 下 TP=4 + SP 才是**
- TP=4 + sequence parallel 把 seq 切 4 份（每 TP rank 只看 seq=4096 的 activation），**省 25 GB**，足以关掉 recompute
- TP=4 + full recompute 反而比 TP=2 + full recompute 慢——TP=4 通信开销只在能换来**关掉 recompute**时才划算

### 6.4 14B 完整 bench 也没跑完

**为什么 14B plan 也没全跑通**：那时 ms-swift 还没修 Qwen3.5 SP，但我们想验证的核心问题（FSDP vs Megatron vs DS 在 seq=16384 实战下谁更快）在 14B 上**也得开 grad ckpt**，得到的对比就不"干净"了。

加上时间预算只有 4 小时，跑一遍三后端 bench + nsys profile + 写报告够呛。

最后选择：**回到 Qwen3.5-9B，但放弃 GBS=384 这个野心，缩到 GBS=8 + SP=2**——这样能：
- 单卡显存压力降一截（每 rank 只看 seq=8192 因为 SP=2）
- 不需要开 grad ckpt（节省 33% FLOPs）
- 9B 比 14B 小，每步真的能跑 1-3 s 而不是 60-180 s
- 能在 1 小时内做完三后端对比 + 多轮迭代优化

---

## 7. 最终决定：回 Qwen3.5-9B，把 GBS 砍到 8 + 加 SP=2

### 7.1 配置降级

| 维度 | 原 plan | 落地版 | 理由 |
|---|---|---|---|
| 模型 | Qwen3.5-9B | **同（保留）** | dense 14B 也撞墙，没必要换 |
| seq | 16384 | 同 | 业务需求 |
| GBS | **384** | **8** | 缩 48 倍，单步 ~2s 而不是 60-180s，能多轮迭代 |
| MBS / GAS | 1 / 48 | 1 / 1 | GAS=1 简化 |
| 三后端 | FSDP2 / DS / Megatron | 同 | |
| FSDP/DS 通信 | 默认 | **+ Ulysses SP=2** | 必需（否则 9B + GDN + seq=16k 装不下） |
| Megatron 通信 | TP=2 PP=1 | **TP=4 PP=1 + SP（mcore native）** | 14B 实测出来的甜点配置照搬 |
| Megatron CP | 计划开 | **关（hard-asserted off）** | mcore_bridge bug |
| 训练步数 | 15 | 5（smoke）/ 30-50（长跑验证）/ 100-200（wall time 测量） | 多档对应不同验证目标 |

### 7.2 这个降级带来的好处

降到 GBS=8 后，每个实验只要 1-5 分钟。可以**快速迭代**：
- 单轴扫盘（NO_AC / no_reshard / MBS=2 / wrap_large / sp1_mbs2 / compile）—— 10 个配置 60 分钟
- 组合验证（combo_easy）—— 10 分钟
- 长跑稳定性验证（30 步 / 50 步）—— 5-10 分钟
- packing + Liger 优化（pack / pack_liger / pack_liger_dl）—— 30 分钟
- 三后端横向 + nsys profile —— 1 小时

整套优化路径前后做了 **14 组实验**，全部记录在 [`bench_fsdp_opt/_all_runs.md`](../megatron_output/bench_fsdp_opt/_all_runs.md)。

### 7.3 故事的转折点

降到 GBS=8 + SP=2 后：
- 第一波单轴尝试（baseline / no_ac / no_reshard / mbs2 / combo_easy）—— **wall time 没显著改善**（baseline 37 min/epoch → combo_easy 37 min/epoch），但发现 GPU 真 MFU 上去了（功耗 247W → 692W）
- **关键诊断**：dataset 单条样本平均只有 2823 token，padding 到 16384 浪费 83%。**真瓶颈不是 GPU 算力，是 padding waste**
- 加 `--packing true` —— wall **37 min → 13 min（2.8×）**
- 加 `--use_liger_kernel true` —— wall **13 min → 9.8 min（再 25%）**

最终配置 [`pack_liger`](fsdp2_optimization_walkthrough.md#9-最终配置--一键生产训练命令)：
- 1 epoch wall: **9.8 min**（baseline 37 min 的 26%）
- 真 MFU: **~50%**（nsys 实测）
- peak power: **706 W**（H100 TDP 700W）
- 30 步长跑 0 NaN

详细优化路径见 [`fsdp2_optimization_walkthrough.md`](fsdp2_optimization_walkthrough.md)。

---

## 8. 教训汇总

写给后人少走弯路：

### 8.1 不要相信"GDN/linear attn 训练能跑长 context"

宣传是推理结论。**训练 chunk-parallel backward 需要 O(seq) workspace**，不能省。

如果你必须用 Qwen3.5-9B / Qwen3-Next / 任何带 GDN 的模型 + 长 seq，**默认得开 SP=2**（甚至 SP=4）。这不是性能优化，是**为了能装下**。

### 8.2 跨代模型升级前先做 smoke seq 扫描

不要照搬上一代的"GBS=384, no-ckpt"配置直接跑新模型。先做 5 分钟 smoke：

```bash
for SEQ in 2048 4096 8192 16384; do
    MBS=1 MAX_LEN=$SEQ TOTAL_STEPS=3 GRAD_CKPT=false bash bench_fsdp.sh
done
```

看哪个 seq 开始 OOM。**如果 OOM seq 远低于业务需求，就提前规划 SP / TP / 缩 GBS / 开 ckpt**，不要等到正式 bench 才发现。

### 8.3 大 GBS 不一定快

原 plan 想 GBS=384 是因为"大 batch tokens/step 多 = throughput 高"。但实际：
- 每 step 60-180 s，**调参 + 等 OOM 反馈极慢**
- 一次 ckpt cold start 损失就大
- 多个 microbatch 累计的 grad 内存 + activation peak 超 H100 80GB 容量

**先用小 GBS 把 throughput / MFU / 稳定性都调好，再放大 GBS**。GBS=8 → GBS=64 → GBS=384 这种渐进式比直接 GBS=384 稳得多。

### 8.4 别低估 ms-swift / mcore_bridge 的 model-specific bug

ms-swift 和 mcore_bridge 这种"上游 model adapter 库"有大量 model-specific 代码路径，**新模型出来通常前 1-2 个月各种边界 bug**：
- ms-swift Qwen3.5 SP 集成 bug 修了 3 个 PR（#9162/#9167/#9189）才稳定
- mcore_bridge Qwen3.5 CP 至今没修
- fla 库 GDN kernel 在 H100/Blackwell 上 silent 精度问题

**用新发布 6 个月内的模型 + 复杂并行配置，要预留 30% 时间给踩 bug**。

### 8.5 实测 > 计划

原 plan 假设 "FSDP/DS 边缘 ~55-70 GB，Megatron TP=2 宽裕"。实测全反过来：
- FSDP/DS 都得开 ckpt
- Megatron TP=2 OOM，必须 TP=4

**先 smoke 拿 5 步实测数据再下结论**，不要靠"看模型大小估算"。

---

## 写在最后

这份 doc 的目的是把"**为什么我们最后做的事是 GBS=8 + SP=2 而不是 GBS=384**"的完整推理链留下。如果你接手这个项目想做新方向（比如换模型、换 backend、换序列长度），**先看 §8 的 5 条教训**，能避免相同的弯路。

后续优化路径请看 [`fsdp2_optimization_walkthrough.md`](fsdp2_optimization_walkthrough.md)（FSDP2 主线）/ [`megatron_optimization_walkthrough.md`](megatron_optimization_walkthrough.md)（Megatron 路线）/ [`deepspeed_optimization_walkthrough.md`](deepspeed_optimization_walkthrough.md)（DS loss=0 bug 踩坑）。
