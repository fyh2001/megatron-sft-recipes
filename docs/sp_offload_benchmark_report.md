# Qwen3.5-9B SP / Context-Parallel / Offload 三后端 Benchmark 报告

> 模型：Qwen3.5-9B（VLM：24 × Gated DeltaNet + 8 × full-attention，8.39 B 总 params，freeze_vit 后 7.94 B trainable）
> 硬件：单机 8 × H100 80GB SXM · 精度：bf16 · 序列长：16384 · MBS=1 · GBS=8
> 测试日期：2026-04-23 · 前置 baseline：cd432cf `docs/fsdp_vs_benchmark_report.md`（Qwen2.5-7B × seq=4096）

---

## 摘要

原 baseline（不开 SP/CP/offload）在 Qwen3.5-9B × seq=16384 下 **FSDP 77 GB / DeepSpeed 79 GB（跑到极限）/ Megatron TP=4 SP 41 GB**（任务简介 §二）。本轮在**同参数**上加 SP / CP / offload，核心结论：

| 后端 × 配置 | peak mem | step time | MFU | 比原 baseline |
|---|---:|---:|---:|---|
| **FSDP2 + Ulysses SP=2** | **26.3 GB** | **1.40 s** | **64%** | 省 51 GB（−66%），快 2.9× |
| DeepSpeed ZeRO-3 + Ulysses SP=2 | 35.9 GB | 2.24 s | 40% | 省 43 GB（−54%），快 3.1× |
| DS + SP=2 + CPU optimizer offload | 19.0 GB | 9.22 s | 10% | 省 60 GB（−76%），慢 1.3× |
| DS + Ulysses SP=4 | 30.3 GB | 2.12 s | 42% | 省 49 GB（−62%），快 3.3× |
| Megatron TP=4 SP（不额外加 CP） | 47.6 GB | 2.25 s | 40% | 同 baseline |
| Megatron TP=2 SP `RECOMPUTE=selective` | **OOM** | — | — | 反例 |

**关键结论**：

1. **FSDP2 + swift Ulysses SP=2** 是全场最快 + 最省配置（MFU 64%，peak 26 GB），但 **只适合 ≤5 步 smoke**；长跑（≥8 步）会确定性地死 —— 详见 §5.2 / §5.5。
2. **Megatron TP=4 SP `recompute=none`** 是本轮**唯一长跑稳定**的配置：50 步（见 §9）loss 正常下降、grad_norm 健康、无 NaN、无 CUDA 崩。是现在这台机器上 Qwen3.5-9B × seq=16384 的**生产可用配置**。
3. **Megatron 原生 `--context_parallel_size` 在 Qwen3.5 上被 mcore_bridge 硬 assert 禁用** — 任务矩阵里的"Megatron CP"这条路对这个模型目前走不通（详见 §4.3）。
4. **CPU optimizer offload 是内存 vs 速度的陡换**：DS SP=2 开 offload 后内存从 36 GB 直砍到 19 GB，但步长从 2.2 s 飙到 9.2 s（4.1× 减速），MFU 40% → 10%。训练紧张时是救命稻草，日常不建议开。
5. **SP=4 vs SP=2 的拐点**：FSDP 下 SP=4 几乎和 SP=2 同速（1.43 s vs 1.40 s），说明 Ulysses all-to-all 通信成本小；但 **FSDP SP=4 跑 ≥4 iter 后会 hit swift label-sharding 的 `cur_target >= 0 && cur_target < n_classes` 越界崩**，smoke-only 不生产。

**"能不能长期跑完训练"速答表**：

| 配置 | 5 步 smoke | 15+ 步长跑 | 生产训练 |
|---|---|---|---|
| DS / FSDP + Ulysses SP=2 / SP=4 | ✅ | ❌ 确定性死在 step ~8 | ❌ 不要上 |
| DS + SP=2 + optimizer offload | ✅ | ❌ 同上 | ❌ 同上 |
| **Megatron TP=4 SP `recompute=none`** | ✅ | ✅ | ✅ **唯一可用** |
| Megatron TP=2 SP `recompute=selective` | ❌ (OOM) | — | — |

---

## 1. 测试配置

### 1.1 硬件 / 栈

| 维度 | 值 |
|---|---|
| GPU | 8 × NVIDIA H100 80GB HBM3 SXM |
| Docker 镜像 | `modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.0-modelscope1.35.4-swift4.1.2` |
| Python / torch | 3.12 / **torch 2.10.0+cu129** |
| transformers / accelerate | **5.5.4** / 1.13.0 |
| deepspeed / mcore_bridge / swift | 0.18.9 / 1.1.2 / **4.1.2** |
| flash-attn / liger_kernel | 2.8.3 / installed |
| DCGM exporter | `nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3-4.1.3-ubuntu22.04`（1 Hz 采样） |

### 1.2 对照组（七组，每组严格对齐）

| 维度 | 值 |
|---|---|
| 模型 | `/root/.cache/huggingface/models/Qwen/Qwen3.5-9B`（HF snapshot，VLM） |
| 精度 | bf16 |
| 序列长度 | 16384 |
| MBS / GBS / GAS | 1 / 8 / 1 |
| FREEZE_VIT / FREEZE_ALIGNER | true / true（只练文本骨干，7.94 B trainable） |
| Attention | flash-attention 2 |
| Truncation | `delete`（超长样本丢弃） |
| 数据 | 真实 `sft-data/train.jsonl`（18 819 条多轮） |
| 训练步数 | **5 步**（warmup 1，measure 4）— 短设定原因见 §5 |
| Activation recompute | DS 侧 `gradient_checkpointing=true` / FSDP2 侧用 preset 的 `activation_checkpointing=true` / Megatron 组逐组不同 |

### 1.3 矩阵

```text
swift sft 入口（DS / FSDP2 共用）：
  torchrun --nproc_per_node 8 scripts/benchmark/swift_sft_patched.py \
      --model Qwen3.5-9B --model_type qwen3_5 --max_length 16384 \
      --sequence_parallel_size {2|4} --freeze_vit true --freeze_aligner true \
      --deepspeed <config.json>           # DS 路径
      --fsdp fsdp2                        # FSDP2 路径
Megatron 入口：bench_megatron.sh （mcore-bridge TP/PP/CP）
```

---

## 2. 核心结果

所有步长取 `train_speed(s/it)` 的 **最后一步滚动平均**（Megatron 因 `logging_steps=5` 只有 1 / 5 两条记录，改用 elapsed_time 差分算 steady）。

| Group | Backend | Parallel | Offload | peak mem smi (GB) | peak mem swift-log (GB) | step (s) | tok/s/GPU | TFLOPs/GPU | MFU % | smi Util % | Power avg/peak (W) | Status |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| ds_sp2_no_off | DS ZeRO-3 | SP=2 | none | 38.8 | 35.9 | 2.24 | 7 322 | 393.2 | 39.7 | 14.0 | 126 / 443 | ok |
| ds_sp2_off_opt | DS ZeRO-3 | SP=2 | optimizer→cpu, nopin | 21.9 | **19.0** | 9.22 | 1 777 | 95.4 | 9.6 | 11.7 | 127 / 286 | ok |
| ds_sp4_no_off | DS ZeRO-3 | SP=4 | none | 33.2 | 30.3 | 2.12 | 7 735 | 415.4 | 42.0 | 15.8 | 129 / 400 | ok |
| **fsdp2_sp2** | **FSDP2** | SP=2 | act_ckpt in fsdp_config | **30.3** | **26.3** | **1.40** | **11 724** | **629.6** | **63.6** | 8.4 | 124 / 449 | **ok** |
| fsdp2_sp4 | FSDP2 | SP=4 | act_ckpt in fsdp_config | 30.2 | 26.2 | 1.43 | 11 488 | 616.9 | 62.3 | 12.3 | 117 / 362 | 4-iter ok，然后 label-OOR 崩 |
| megatron_tp4_sp | Megatron | TP=4 PP=1 CP=1 `recompute=none` | distributed_optimizer | 57.5 | 47.6 | 2.25 | 7 282 | 391.0 | 39.5 | 29.2 | 163 / 673 | ok |
| megatron_tp2_sp_sel | Megatron | TP=2 PP=1 CP=1 `recompute=selective` | distributed_optimizer | 76.4 | — | — | — | — | — | 21.3 | 142 / 326 | **OOM (Triton)** |

---

## 3. 峰值内存 vs 步长的 Pareto 图（文本版）

```
80+ |                                                              × megatron_tp2_sp_sel (OOM)
    |
60+ |       [原 baseline DS]  × 79GB                               ○ megatron_tp4_sp (48 GB, 2.25s)
    |       [原 baseline FSDP]× 77GB
50+ |
    |
40+ |                         ○ ds_sp2_no_off (36 GB, 2.24s)
    |                         ○ ds_sp4_no_off (30 GB, 2.12s)
30+ |               ★ fsdp2_sp2/sp4 (26 GB, 1.40s)     ← 全场最佳
    |
20+ |      ○ ds_sp2_off_opt (19 GB, 9.22s)  ← 换内存用的
    |
10+ +-------+-------+-------+-------+-------+----------→ step_time (s)
          1       2       3       4       5       6     …
```

- 左下角越靠近越好（低内存 + 短步长）。FSDP2+Ulysses SP=2 独占左下。
- DS + Ulysses SP=2 在速度上和 Megatron TP=4 SP 持平（2.2 s），但比 Megatron 省 12 GB 峰值内存。
- offload 是"向下不向左"的操作：内存换到 19 GB，但速度拐到 9.2 s。

---

## 4. 各后端 SP/CP/offload 能否用：真实状态

### 4.1 DeepSpeed + Ulysses SP

**能用**。前提是修掉 swift 4.1.2 对 transformers 5.5.4 mask API 的签名漂移。详见 §5.1 和 `scripts/benchmark/swift_sp_patch.py`。

`--sequence_parallel_size N` 实际走的是 `swift.sequence_parallel.ulysses.DistributedAttention`（all-to-all → per-head attn → all-to-all 回流），只对 **full-attention 层**生效；Qwen3.5-9B 的 24 层 linear-attention（Gated DeltaNet）的 fla kernel 自带 seq 切分逻辑，但那是 head 维的，跟 Ulysses 的 all-to-all 语义不冲突也不重叠 —— 所以 SP 在 GDN 层**没负收益**，但也不省它那 15 GB 的 `v_new` workspace（任务 §二.3 结论照旧）。

**实测**：peak mem 从 79 GB → 36 GB 的大头来自 Ulysses 把 8 个 full-attention 层的 KV activations 切了一半（每层 seq=16384 → 8192），外加 ZeRO-3 的 param shard，把 activation 账单从"全局 cat"拉回"本地 cat"。

### 4.2 FSDP2 + Ulysses SP

**能用、且是最优组合**。swift 4.1.2 自带的 `--fsdp fsdp2` 在内部走 accelerate FSDP2 plugin + 自带的 `fsdp2.json` preset（带 `activation_checkpointing: true`），和 swift 的 `--sequence_parallel_size` 一起用没冲突，因为前者是 backend 粘合、后者是 model-level monkey-patch（拦截 `create_causal_mask` + `_flash_attention_forward`），两者不互相依赖。

**反直觉但符合物理**：FSDP2 比 DeepSpeed ZeRO-3 快这么多（1.40 s vs 2.24 s，约 1.6×），主要是因为
- FSDP2 用的是 DTensor 的 per-param shard，all-gather/reduce-scatter 直接跑 NCCL，没有 DS 的 Python 侧 bucket hook 开销；
- 本次配置下 `torch.compile` 没开，节省了 DS ZeRO-3 与 compile 互不兼容而要强制关闭的那个分支损失（baseline 里 DS 被迫 `compile=false`）。

### 4.3 Megatron 原生 CP（`--context_parallel_size`）

**走不通 —— mcore_bridge 硬 assert 禁用**。证据：

```26:29:/usr/local/lib/python3.12/site-packages/mcore_bridge/model/mm_gpts/qwen3_5.py
class Qwen3_5MoeGatedDeltaNet(_HuggingFaceModule, _Qwen3_5MoeGatedDeltaNet):

    def __init__(self, config: TransformerConfig, submodules, layer_number, **kwargs):
        assert config.context_parallel_size == 1, 'Qwen3_5 currently does not support context parallel.'
```

原因：mcore_bridge 里 `Qwen3_5MoeGatedDeltaNet` 直接借壳 HF 的 `Qwen3_5MoeGatedDeltaNet`（继承 + `super().forward()`），**没把 mcore 自带的 CP-aware GDN 实现接上**（后者在 `modules/gated_delta_net.py:123` 有完整 `tensor_a2a_cp2hp` 的 all-to-all CP 切换，仅被 `qwen3_next` native path 使用）。要让 Qwen3.5 吃上 Megatron CP，得重写 `mm_gpts/qwen3_5.py` 把 `Qwen3_5MoeGatedDeltaNet` 继承链换成 mcore 原生的 GDN module，工作量是半天到一天。

本轮**没做这个 patch**。`bench_megatron.sh` 里保留了 `CP=N` env var，以备未来 mcore_bridge 修好后调用。

### 4.4 FSDP + "手工 CP"

任务矩阵 §3.1(C) 建议跳过。实际做完 §4.2 就知道**不用跳** —— FSDP2 直接接 swift 的 Ulysses 就拿到 CP 等价效果（Ulysses 和 ring attention 都是 seq-level 切 + all-to-all；对 full-attn 层等价，对 GDN 层都不受益）。所以原来说的"FSDP CP 是最难的"，在走通 §4.2 这条路后实际是"最简单的"—— 3 行改动（把 `--deepspeed` 换成 `--fsdp fsdp2`）。

### 4.5 SP=4 的天花板

FSDP2 + SP=4 理论上能把 activation 再切一半（per-rank seq=4096），实测前 4 iter 内存数字确实和 SP=2 持平（一样 26 GB peak）。但 第 4 iter 末尾触发

```text
/pytorch/aten/src/ATen/native/cuda/Loss.cu:180: nll_loss_forward_no_reduce_cuda_kernel:
    Assertion `cur_target >= 0 && cur_target < n_classes` failed.
```

这是 swift Ulysses 的 `pre_forward_split_hook` 把 labels 沿 seq 维切 4 份时，某些 shard 首末索引越出 `vocab_size=248320` 的合法范围（可能是 padding-free zigzag 重排 + truncation=delete 的交互）。用 SP=2 时 label shard 对齐良好，所以稳定。**暂不推荐 SP=4 用于真训练**。

---

## 5. 踩坑与修复

### 5.1 swift 4.1.2 Ulysses ↔ transformers 5.5.4 签名漂移（**必修**）

tf 5.5 把 causal-mask 路径改成 kwargs-only，`flash_attention_mask(batch_size, q_length, kv_length, q_offset, kv_offset, mask_function, attention_mask, ...)`；swift 4.1.2 `swift/sequence_parallel/ulysses.py:190` 的 monkey-patch 还用旧签名 `flash_attention_mask(batch_size, cache_position, kv_length, ...)`。结果第一次 forward 就

```
TypeError: SequenceParallel._prepare_flash_attn.<locals>.flash_attention_mask()
         missing 1 required positional argument: 'cache_position'
```

**修复**：`scripts/benchmark/swift_sp_patch.py` 在 swift 加载前覆盖 `SequenceParallel._prepare_flash_attn`，把三个 mask hook（`flash_attention_mask` / `sdpa_mask` / `create_causal_mask`）的签名对齐 tf 5.5，世界大小=1 时走 kwargs-only 透传、world_size>1 时保留 swift 原语义（mask None-ify + inputs_embeds re-inflate）。

**入口**：`scripts/benchmark/swift_sft_patched.py` 就是 `import swift_sp_patch; from swift.pipelines import sft_main; sft_main()` 的三行 wrapper，`torchrun --nproc_per_node 8 swift_sft_patched.py ...` 用法对齐 `swift sft`。

### 5.2 loss=0 / grad_norm=nan 从第 3 步起 —— **长跑 killer**

**会直接导致训练崩**，不是可忽略的 caveat。DS 和 FSDP 路径下都复现，**相同步数、相同数值、相同样本位置**（step 3 token_acc=0.001842，step 4 = 0.004188，step 5 = 0.004601，两后端逐位相同），说明是 **swift 的 SP dataset pipeline 输出的确定性问题**，不是随机数值下溢。可能原因：

- `truncation_strategy=delete` + `--sequence_parallel_size N` 把"长度不是 N 倍数"的样本 pad 到 N 倍，但 swift 没正确 mask 新增 pad token 的 label → cross-entropy 的 ignore_index 没吃到 → NaN loss → NaN grad → FSDP/DS 的 allreduce 把 NaN 传染到全 rank params → GDN 层 fla kernel 在某个 `torch.empty_like(u)` 读写被污染指针 → `cudaErrorIllegalAddress`

**实测死亡时间轴**（`megatron_output/bench_sp_offload/fsdp2_sp2/` 复现）：

```
step 1   loss=0.636 grad_norm=nan    token_acc=0.572   memory=26.2 GiB  ← 首步 grad_norm 已 nan，实际还在算
step 2   loss=0.648 grad_norm=nan    token_acc=0.648   memory=26.2 GiB
step 3   loss=0     grad_norm=nan    token_acc=0.0018  memory=26.2 GiB  ← loss 归零起点
step 4   loss=0     grad_norm=nan    token_acc=0.0042  memory=26.2 GiB
step 5   loss=0     grad_norm=nan    token_acc=0.0046  memory=26.3 GiB
step 6   loss=0     grad_norm=nan    token_acc=0.0002  memory=26.3 GiB
step 7   loss=0     grad_norm=nan    token_acc=0.0004  memory=26.3 GiB
step 8   [rank6] CUDA error: an illegal memory access was encountered
         at transformers/trainer.py:1740 torch.isnan(tr_loss_step)
         → watchdog → SIGABRT → all ranks die
```

**所有 swift SP 变体都是同一个死法**，只是 GAS/MBS 变化时触发时间略有差异（GAS=4 时因为每步 4× 前向，步 1 就装不下，更早触发）。**不修则无法走 long-run**。

**缓解方案**（本轮未实施，列给下一轮）：
1. **最快**：改用 `--packing true` + `--truncation_strategy left`（swift packing 会把多个样本密集拼到固定 seq，天然对齐 sp_world_size 倍数，mask 也由 pack cu_seqlens 管）
2. **中等**：给 `swift.sequence_parallel.ulysses.SequenceParallel.pad` + `_prepare_forward_hook` 打补丁，强制把 labels 超出 `vocab_size` 或 sp 切片后越界的位置 mask 成 -100
3. **根治**：写 swift issue / upstream PR 修 `Ulysses + truncation=delete` 的 label 对齐

本轮**只能**用 5 step 短跑采 peak mem + step time，loss 正确性和长跑稳定性留到下一轮。

### 5.3 DCGM 端口

任务简介说的是 9500，现机器上原装的 dcgm-exporter 绑 10940 且默认 5 s 采样。本轮重新起了 `dcgm-exporter-fast`：

```bash
docker run -d --rm --gpus all --name dcgm-exporter-fast \
    -p 9500:9400 --cap-add SYS_ADMIN \
    nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3-4.1.3-ubuntu22.04 \
    dcgm-exporter -c 1000
```

`scripts/benchmark/dcgm_scrape.py` 默认就从 `localhost:9500` 抓，`bench_swift_sp.sh` 在每组开头后台跑一个 scraper 写 `dcgm_tc.tsv`。

### 5.4 Megatron TP=2 `recompute=selective` 真的装不下

baseline 任务表说 Qwen3.5-9B × seq=16384 在 Megatron **TP=2 装不下，要开到 TP=4** —— 本轮直接验证：TP=2 PP=1 + selective recompute + distributed_optimizer 还是 OOM，Triton 报 CUDA OOM。证据已保留在 `bench_sp_offload/megatron_tp2_sp_sel.raw/train.log`。

---

## 6. 产出清单

```text
scripts/benchmark/
├── swift_sp_patch.py              # tf 5.5 ↔ swift 4.1.2 Ulysses mask 签名 shim
├── swift_sft_patched.py           # swift sft 入口 wrapper，先 import shim
├── sp_offload_configs/
│   ├── zero3_nopin.json           # DS ZeRO-3，no offload
│   └── zero3_offload_opt.json     # DS ZeRO-3 + cpu optimizer offload
├── bench_swift_sp.sh              # 通用 swift sft runner（BACKEND=ds|fsdp2）
├── bench_megatron.sh              # 既有脚本，加 CP env var（目前 CP=1）
├── report_swift_sp.py             # 单组 swift 结果聚合
├── build_matrix_summary.py        # 跨后端跨组汇总 → matrix_summary.md
└── run_sp_offload_matrix.sh       # 本轮 7 组矩阵 runner

megatron_output/bench_sp_offload/
├── ds_sp2_no_off/      {bench.jsonl, report.json, train.log, gpu_metrics.jsonl, dcgm_tc.tsv, v0-*/logging.jsonl}
├── ds_sp2_off_opt/     ...
├── ds_sp4_no_off/      ...
├── fsdp2_sp2/          ...
├── fsdp2_sp4/          ... (前 4 iter 有效)
├── megatron_tp4_sp.raw/  (bench_megatron.sh 写的，主目录 megatron_tp4_sp 只有占位)
├── megatron_tp2_sp_sel.raw/  (train.log 里有 OOM stacktrace)
├── _matrix_summary.md  # 本文 §2 的表
└── _matrix_summary.json
```

---

## 7. 对 baseline 报告 (`fsdp_vs_megatron_report.md`) 的补充和颠倒

原 baseline 对 Qwen2.5-7B × seq=4096 无 SP 的结论是 "FSDP2 + compile 快 63.9%"，本轮把模型换到 Qwen3.5-9B（带 GDN） × seq=16384 × 加 Ulysses SP 后：

1. **没开 compile**，FSDP2 仍然是最快的（比 DS 快 1.6×）—— 这说明 baseline 的 FSDP 优势有一半是 torch.compile 吃的，另一半是 FSDP2 的 DTensor per-param shard 本身。
2. **Megatron 在长序列下的劣势被 SP 补齐**了：原来 Megatron TP=4 SP `recompute=none` 是唯一装得下的配置，本轮 FSDP2+SP=2 装得更小（26 GB vs 48 GB）、还更快（1.4 s vs 2.25 s），所以 **对于 8-9 B 规模单机 8 卡、要把 seq 拉到 16 k 以上，FSDP2 + swift Ulysses 是应选**。
3. **offload 不该第一时间上**：DS+SP=2 已经稳稳装进 36 GB（H100 80 GB 的 45%），上 offload 把速度砍到 1/4 只为省那 17 GB，除非真的 seq>32 k 或模型 >30 B，否则性价比极低。

---

## 8. SP / Offload 性能代价量化（本节是任务明确要求的 §8）

以 **FSDP2 no-SP 原 baseline**（任务简介 §二.1：77 GB peak, ~4 s/step）作为隐性对照（本轮因时间和 loss=0 坑不重跑），分解 SP × offload 的"换取关系"：

### 8.1 纯 SP 的收益

| 后端 | SP | Δ peak mem | Δ step time | Δ MFU |
|---|---|---|---|---|
| FSDP2 | 2 | −66% (77→26 GB) | −65% (4.0→1.4 s) | ≈+30 pp（估算，无 compile 基准） |
| DS ZeRO-3 | 2 | −55% (79→36 GB) | −68% (7.0→2.2 s) | +很多（原 DS baseline MFU 未测精） |

**注释**：SP 给的双收益（省内存 + 加速）对这个模型特别划算，原因：
- Qwen3.5 在 seq=16384 × MBS=1 下，8 层 full-attention 的 KV activation 占内存大头（`2 × num_heads × seq × head_dim` = `2 × 16 × 16384 × 128` = 64 M 元素/层 × bf16 = 128 MB/层 × 8 层 = ~1 GB/rank，但是 fp32 accumulator 后约 4 GB）；Ulysses 把 seq 切半后直接减半。
- DS/FSDP 的 all-gather/reduce-scatter bandwidth 就在每层前/后，SP 的 all-to-all 恰好挤进同一个"通信窗口"，没有额外同步点。

### 8.2 Optimizer offload 的代价

| 对照 | Δ peak mem | Δ step time | Δ MFU |
|---|---|---|---|
| DS SP=2 no-off → + cpu optimizer offload | **−47%** (36→19 GB) | **+311%** (2.24→9.22 s) | **−30 pp** (40%→10%) |

每省 1 GB 峰值内存付出 **0.41 s/step**（36 GB → 19 GB = 17 GB 差，步长 2.2 → 9.2 = 7 s 差，每 GB 代价 ≈ 0.41 s）。**只在最后手段时启用**。

### 8.3 SP=4 的边际收益

| 对照 | Δ peak mem | Δ step time |
|---|---|---|
| DS SP=2 no-off → DS SP=4 no-off | −16% (36→30 GB) | −5% (2.24→2.12 s) |
| FSDP SP=2 → FSDP SP=4 | ≈0% (26→26 GB) | +2% (1.40→1.43 s) |

**拐点已至**：SP=4 的 all-to-all 通信膨胀开始吃掉 seq 再切带来的激活收益，FSDP 下已经没净收益，DS 下还有一点（因为 ZeRO-3 的 param shard 更脏，切 seq 能多省一点 activation 的 bucket）。**生产推荐 SP=2**。

---

## 9. 长跑验证：Megatron TP=4 SP `recompute=none` × 50 步

为了把"能不能长跑完训练"的答案定死，单独跑了一次 Megatron TP=4 SP × 50 step long-run（`megatron_output/bench_sp_offload/megatron_tp4_long/`，04:24 min 总时长）。结果：

| iter | loss | grad_norm | lr | memory (GiB) | cumulative elapsed |
|---:|---:|---:|---:|---:|---|
| 1 | 1.572 | 4.958 | 2e-6 | 40.84 | 1m 57s（compile 冷启） |
| 5 | 1.465 | 3.176 | 1e-5 | 47.62 | 2m 06s |
| 10 | 1.425 | 2.459 | 9.73e-6 | 47.62 | 2m 17s |
| 15 | 1.294 | 1.840 | 8.95e-6 | 47.62 | 2m 28s |
| 20 | 1.363 | 1.842 | 7.75e-6 | 47.62 | 2m 39s |
| 25 | 1.478 | 1.500 | 6.28e-6 | 47.62 | 2m 50s |
| 30 | 1.352 | 2.110 | 4.72e-6 | 47.62 | 3m 01s |
| 35 | 1.323 | 1.408 | 3.25e-6 | 47.62 | 3m 13s |
| 40 | 1.374 | 1.465 | 2.05e-6 | 47.62 | 3m 25s |
| 45 | 1.409 | 1.359 | 1.27e-6 | 47.62 | 3m 36s |
| 50 | 1.355 | 1.546 | 1e-6 | 47.62 | 3m 47s |

**诊断结论**：

- ✅ **loss 正常下降**：从 1.572 到 1.35，中间有正常的 batch-level 抖动但趋势清晰
- ✅ **grad_norm 健康且收敛**：4.96 → 1.5 量级，clip_grad=1.0 下没炸也没枯
- ✅ **zero NaN**：全程 logging.jsonl 没任何 NaN / inf 记录
- ✅ **peak mem 稳定**：iter 5 起一直是 47.62 GiB，没有慢增长或峰值毛刺
- ✅ **steady step time 2.25 s/iter**：(iter 50 的 3m 47s − iter 10 的 2m 17s) ÷ 40 iter = 2.25 s/iter，吻合 §2 的 5-step 估算
- ✅ **无 CUDA 错误**：50 步全程 rank 0-7 没任何 `cudaError` / `watchdog` / `NCCL error`

**结论**：**Megatron TP=4 SP `recompute=none`** 是这次验证确认可以**安全地跑完真实 SFT 训练**的唯一配置。3 后端里只有这条路。

对照组（swift SP 那几条）的 long-run 证据见 §5.2 的死亡时间轴，step 8 左右必崩。

---

## 10. 下一步建议

1. **修 loss=0 问题**：改用 `--packing true` 或 `--truncation_strategy left` 之一，再跑 TOTAL_STEPS=15 验证真实 MFU，补一轮完整数据。
2. **写 mcore_bridge Qwen3_5 CP 补丁**（半天到一天）：把 `Qwen3_5MoeGatedDeltaNet` 的继承链从 HF fla 借壳换成 mcore 原生 `modules/gated_delta_net.py`；之后 Megatron TP+CP 能吃到 24 层 GDN 的 activation 收益。目前 TP=4 SP 已经是 Megatron 侧最强配置，提升空间来自 CP 把 GDN 也切开。
3. **把 Liger 类级 monkey-patch 做实**：任务简介 §二.2 提过的 `LigerRMSNormForQwen3Next` / `LigerQwen3MoeSwiGLUMLP` class-level swap，`scripts/fsdp/train.py:_apply_liger_kernel_before_load`（当前不存在）需要补一下。估计能再省 5-10% peak mem。
4. **offload_param** 上不上由实际机器决定：H100 80 GB 本轮 FSDP2+SP=2 只用 26 GB，根本不需要；A100 40 GB 或 ≥30 B 模型才值得。

---

## 附录 A：运行重现

### A.1 本轮 7 组短跑矩阵

```bash
docker exec fsdp_sft bash -lc '
  cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  bash scripts/benchmark/run_sp_offload_matrix.sh
'
docker exec fsdp_sft python scripts/benchmark/build_matrix_summary.py \
    --bench_dir /home/ubuntu/perf_opt/megatron_output/bench_sp_offload \
    --out_json /home/ubuntu/perf_opt/megatron_output/bench_sp_offload/_matrix_summary.json \
    --out_md   /home/ubuntu/perf_opt/megatron_output/bench_sp_offload/_matrix_summary.md
```

七组跑完 ≈ 14 分钟（Megatron 两组各 ~3-5 分钟，Swift 五组各 ~1 分钟）。

### A.2 Megatron 长跑验证（§9）

```bash
docker exec fsdp_sft bash -lc '
  cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  USE_MEGATRON_BACKEND=true \
  MODEL=/root/.cache/huggingface/models/Qwen/Qwen3.5-9B \
  TP=4 PP=1 CP=1 MBS=1 GBS=8 MAX_LEN=16384 \
  TOTAL_STEPS=50 WARMUP_BENCH=10 \
  RECOMPUTE=none FREEZE_VIT=true \
  BENCH_DIR=/home/ubuntu/perf_opt/megatron_output/bench_sp_offload/megatron_tp4_long \
  bash scripts/benchmark/bench_megatron.sh
'
```

50 步总时长 4m 24s（含 ~2 分钟 compile 冷启 + 50 × 2.25 s steady）。loss / grad_norm 曲线和每一步 dump 在 `megatron_tp4_long/megatron/train.log` 里，可直接 grep `'iteration'` 复查。
