# Qwen3.5-9B SP / CP / Offload 三后端 Benchmark 精简报告

> 模型：Qwen3.5-9B VLM（7.94 B trainable，freeze_vit）· 单机 8×H100 80GB · bf16 · seq=16384
> 测试日期：2026-04-23 · 完整报告：[`sp_offload_benchmark_report.md`](./sp_offload_benchmark_report.md)

---

## 一、一句话结论

**`FSDP2 + SP=2 + MBS=2 + NO_AC + no_reshard + packing + liger`（pack_liger）** 是本机最优配置：
- 跑完 1 epoch wall time 从 **37 min → 9.8 min**（**3.8×**）
- 真 hardware MFU 从 **~3% 提到 ~50%**（nsys 2025 实测 per-rank GEMM+Flash time / wall time）
- 峰值功耗从 **247 W → 706 W**（H100 TDP 700 W，已顶死）
- 平均功耗从 **120 W → 519 W**（+332%）
- GPU 平均利用率 **8% → 82%**

FSDP2 pack_liger 比 Megatron TP=4 SP + packing **快 1.5×**（9.8 min vs 14.9 min）。不能只看 "TC-busy time / wall" 这一个口径（50.4% vs 51.1% 打平）—— 还要看**单位 TC 时间里实际做了多少 FLOPs**：

| 口径 | FSDP2 pack_liger | Megatron TP=4 SP | 解读 |
|---|---:|---:|---|
| TC-busy time / wall（nsys 实测，上界） | 50.4% | **51.1%** | tensor core pipeline 开了多少占比 |
| achieved TF/s per GPU（6ND/step_time 推算） | **614** | 402 | 在 TC 忙的时间里，每秒真的产了多少 FLOPs |
| avg power（nvidia-smi 1Hz） | **519 W** | 433 W | 物理旁证：FSDP2 多烧 20% 电 |
| tokens/s/GPU（用户能直接看到） | **11 360** | 7 447 | **+53%** |

**Megatron 看起来 TC-busy 略高，但 achieved TFLOPs 比 FSDP2 低 35%**。原因：
- **TP=4 切小 GEMM tile**（K/N 从 4096 变 1024），cuBLASLt WGMMA 在 K<2048 时 TC cycle 利用率掉 30-40%
- **更小的有效 DP**：FSDP2 SP=2 → DP=4；Megatron TP=4 → DP=2

所以 FSDP2 pack_liger 快不是因为 tensor core pipeline 更忙，而是**每次 tensor core 忙碌的吞吐更高**（tile 大 + tokens/step 多 2×）。

DS ZeRO-3 + SP 在 Qwen3.5 + swift 4.2.0.dev0 git main 下 **loss=0 bug 依然存在**，pack+liger 也救不回来。

前提：`ms-swift ≥ 4.2.0.dev0`（git main），Qwen3.5 的 Liger kernels（RMSNorm / SwiGLU / RoPE）可用。只能用 pypi 4.1.2 的话 Megatron TP=4 SP `recompute=none` 仍是唯一长跑稳定路径。

---

## 二、核心结果表（10-step 扫盘 + 100-step wall time 验证）

baseline 参考（无 SP/CP/offload，Qwen3.5-9B × seq=16384）：FSDP 77 GB / DS 79 GB / Megatron TP=4 SP 41 GB。

### 优化轴增量贡献（每行在上一行基础上加一个改动）

| 配置 | steady step | 真实 tokens/step | peak mem | avg power | peak power | **full-epoch wall** | **vs baseline** |
|---|---:|---:|---:|---:|---:|---:|---:|
| FSDP2 + SP=2 (baseline) | 0.95 s | 22 k | 29 GB | 120 W | 247 W | **37.2 min** | 1.0× |
| + NO_AC + MBS=2 + no_reshard = **combo_easy** | 1.91 s | 45 k | 54 GB | 228 W | 692 W | 37.4 min | 1.0× |
| + **packing** = pack | 3.88 s | 261 k | 59 GB | 443 W | 707 W | 13.1 min | **2.8×** |
| + **Liger kernels** = **pack_liger** ★ | **2.88 s** | **261 k** | **59 GB** | **519 W** | **706 W** | **9.8 min** | **3.8×** |
| + 8 dataloader workers + 预分词 | 2.87 s | 261 k | 59 GB | 543 W | 708 W | 9.7 min | 3.8× |

★ **`pack_liger` = `BACKEND=fsdp2 SP=2 MBS=2 NO_AC=true FSDP_RESHARD=false PACKING=true USE_LIGER=true`**。额外加 8 worker 的 dataloader 基本无增益——瓶颈已经不在数据管道。

### 三后端 apples-to-apples 实测（nsys 2025 + 100 步 wall time，全部实测）

| 后端 | step time | peak mem | tokens/s/GPU | **per-rank TC-eligible / wall** | **full-epoch wall** | loss |
|---|---:|---:|---:|---:|---:|---|
| FSDP2 + SP=2 baseline | 0.95 s | 29 GB | 11 529 | ~5%（推算，未 nsys） | **37 min** | ✅ |
| **FSDP2 + pack_liger** ★ | 2.88 s | 59 GB | 11 360 | **50.4%** | **9.8 min** | ✅ |
| DS ZeRO-3 + SP=2 + pack_liger | 2.77 s | 53 GB | 11 803 | — | — | ❌ **loss=0 bug** |
| **Megatron TP=4 SP + packing** | 2.20 s | 48 GB | 7 447 | **51.1%** | **14.9 min** | ✅ |

> **"per-rank TC-eligible / wall"** = `(GEMM + FlashAttn 绝对时间 / rank 数) / wall window`。这是 tensor core pipeline 活跃时间占比，是真 MFU 的**上界**（假设 TC 跑到 989 TF 峰值）。
> 小心：不能用 `GEMM time / aggregate GPU busy time` —— FSDP2 的异步 stream 让 aggregate > 100% of wall，Megatron 的 stream overlap 较弱 aggregate ≈ 100%，分母不一致会造成伪对比。
> 另：TC-busy time 高不等于 achieved TFLOPs 高 —— Megatron TP=4 tile 小（K=1024），同样 TC 时间里 TC cycle 利用率比 FSDP2（K=4096）低。

**关键洞察**：

1. **TC-busy time 打平 ≠ achieved TFLOPs 打平**：FSDP2 pack_liger TC-busy 50.4% 但 achieved 614 TF/s；Megatron TC-busy 51.1% 但 achieved 只 402 TF/s。差异来自 TP=4 小 tile 的 TC 效率惩罚 + tokens/step 多 2×（MBS + 有效 DP 都大）
2. **FSDP2 Elementwise 11.7% + Memory 8.1%**（per-parameter DTensor shard 的隐性开销），Megatron 只有 1.9% + 2.8%。但 FSDP2 靠 async NCCL stream 掩盖了这些开销，achieved throughput 仍然领先 Megatron
3. **DS + Ulysses SP 在 Qwen3.5 上不可用**：pack_liger 叠上去 loss 从 step 2 起依然恒 0、grad_norm=√2（HF Trainer 对 NaN grad 的 fallback）；2+ 步是空转。根因是 modelscope 还没修的 DS ↔ swift SP 集成 bug
4. **Megatron 没法叠 Liger**：Megatron 用自家 `apex.normalization` / `transformer_engine.rmsnorm`，和 Liger 功能重叠；`bench_megatron.sh --packing true` 已经默认开了 packing

**Megatron 的真正优势（不在 Qwen3.5-9B × seq=16k 这个点上）**：
- **长跑最稳**：pypi swift 4.1.2 即可生产（不用 git main）
- **能装大模型**：TP=4 能把 30B/70B 放进 8×H100；FSDP2 ZeRO-3 + SP 在大模型下通信开销会明显拖累
- **CP（context parallel）** 能切 seq 到 128k+（如果 mcore_bridge 修掉 Qwen3.5 的 CP assert，见完整报告 §4.3）

### 四种 MFU 口径（全部实测，注意口径差异）

| 口径 | FSDP2 baseline | FSDP2 pack_liger | Megatron TP=4 SP | 是否可信 |
|---|---:|---:|---:|---|
| ① swift 公式 MFU（`6ND/step_time`） | 62.6% | 61.7% | 37.9% | ⚠️ 对 TP 不公平（小 tile 惩罚 + 假设全 TC） |
| ② TC 占 aggregate GPU 时间 | 10.5% | 44.5% | 51.3% | ⚠️ **分母口径不同**（async stream 会膨胀 FSDP 分母，让 Megatron 看着高） |
| ③ **Per-rank TC / wall time**（真 MFU 上界） | ~5% | **50.4%** | **51.1%** | ✅ **apples-to-apples**，可以直接对比 |
| ④ DCGM peak power | 247 W | 706 W | — | ✅ 物理证据 |

**口径 ③ 的计算**：`(GEMM_ms_aggregate + FlashAttn_ms_aggregate) / rank_count / wall_capture_ms`。例如 FSDP2 pack_liger：`(37234 + 3093) / 8 / 10000 = 50.4%`。

**原始 profile 文件**：
- FSDP2 baseline (torch.profiler)：`megatron_output/bench_sp_offload_profile/profile_fsdp2_sp2/trace_rank0_step3.json`
- FSDP2 pack_liger (torch.profiler + nsys 2025)：`megatron_output/bench_fsdp_opt/pack_liger_prof/` + `pack_liger_nsys/profile.nsys-rep`
- Megatron TP=4 SP (nsys 2025)：`megatron_output/bench_fsdp_opt/mega_prof/megatron/profile.nsys-rep`

---

## 三、优化轴诊断（10 组扫盘实测）

Profile 发现 baseline 的 step 3 rank 0 GPU kernel 分布：

| 类别 | 时间 | 占 GPU kernel | 占 wall time |
|---|---:|---:|---:|
| **NCCL**（AllGather 132 ms + ReduceScatter 59 ms + SendRecv 9.5 ms） | 200 ms | **63%** | 17% |
| FSDP buffer 重排（`split_with_sizes_copy` + `chunk_cat`） | 50 ms | 16% | 4% |
| **GEMM**（`nvjet_*` cuBLASLt JIT） | 31 ms | 10% | 2.6% |
| 其他（Adam / elementwise / reduction / flash_attn） | 38 ms | 12% | 3% |
| **GPU idle（CPU dispatch / dataloader）** | **~880 ms** | — | **~73%** |

**症状：通信 bound + CPU dispatch bound，不是算力 bound。** 因此有效优化方向：

| 改动 | 真 TC-eligible 变化 | 机制 |
|---|---:|---|
| 关 activation_checkpointing | 2.3% → 2.8% | 少一次 forward recompute |
| MBS 1 → 2（单独） | 2.3% → 2.2%（但 step time 爆炸到 17 s，不稳定） | 内存压力下触发 FSDP 慢路径 |
| `reshard_after_forward=false` | 2.3% → 2.1%（但 DCGM TC busy20 0.25% → **8.7%**） | 省反向 AllGather，NCCL −24% |
| **三者叠加（combo_easy）** | 2.3% → **17.0%** | **每步工作量 ×2 / 通信粒度减半 → 掩盖通信窗口** |
| SP=4 / mbs4 / wrap_large / sp1_mbs2 | N/A | mbs4 接 OOM；wrap_large/sp1_mbs2 本机实现崩 |
| torch.compile | N/A | torch 2.10 + Inductor + swift 的 TF32 API 冲突（上游 bug） |

---

## 四、长跑稳定性

| 配置 | swift 4.2.0.dev0 | 长跑验证 | 生产可用 |
|---|---|---|---|
| Megatron TP=4 SP `recompute=none` | ✅ | **50 步** loss 正常、grad_norm 健康 | ✅ 不依赖 swift SP 修复 |
| FSDP2 + SP=2（baseline） | ✅ | 30 步 loss 正常 | ✅ 但 MFU 只有 3% |
| **FSDP2 + combo_easy** ★ | ✅ | **30 步 loss 正常**（1.75→1.05→1.66）、peak 692 W 真干活 | **✅ 本机推荐** |
| FSDP2 + SP=4 | ✅ | 30 步 loss 2.37→0.36 | ✅ 省内存版 |
| DS ZeRO-3 + SP=2 | ⚠️ | loss 恒 0、grad clip 成 √2 | ❌ |

---

## 五、推荐配置速查

| 场景 | 配置 | full-epoch wall | peak mem | 真 MFU |
|---|---|---:|---:|---:|
| **本机最优（swift main）** ★ | `pack_liger`（combo_easy + packing + Liger） | **9.8 min** | 59 GB | **~50%**（nsys 实测） |
| 省内存一点 | combo_easy（不开 packing） | 37 min | 54 GB | ~17%（估算，未 profile） |
| 完全默认 | FSDP2 + SP=2 + MBS=1（baseline） | 37 min | 29 GB | ~5% |
| 只能用 pypi swift 4.1.2 或要大 TP | Megatron TP=4 SP `recompute=none` + packing | **14.9 min** | 48 GB | **~51%**（nsys 实测） |
| seq > 32 k 或模型 ≥ 30 B | 再叠 optimizer offload 或 Megatron TP=8 | — | — | — |

---

## 六、关键命令

升级 ms-swift 到 git main：

```bash
docker exec fsdp_sft pip install --force-reinstall --no-deps --no-cache-dir \
    git+https://github.com/modelscope/ms-swift.git@main
```

推荐配置（`pack_liger`）生产跑：

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

对应 swift CLI flag:
```
--packing true --use_liger_kernel true
--per_device_train_batch_size 2
--fsdp <override.json>  # reshard_after_forward:false, activation_checkpointing:false
```

Profile 任意配置以核查真实 MFU：

```bash
docker exec fsdp_sft bash -lc "cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  TOTAL_STEPS=5 PROFILE_STEP=3 \
  BACKEND=fsdp2 SP=2 MBS=2 NO_AC=true FSDP_RESHARD=false \
  BENCH_DIR=/tmp/myprofile RUN_NAME=run \
  bash scripts/benchmark/bench_swift_sp_profile.sh"
# Output: /tmp/myprofile/run/analysis.txt
```

---

## 七、参考

- 完整报告（含踩坑 / 社区 PR 时间线）：[`sp_offload_benchmark_report.md`](./sp_offload_benchmark_report.md)
- 10 组扫盘原始表：`megatron_output/bench_fsdp_opt/_opt_summary.md`
- Profiler 工具：[`scripts/benchmark/torch_profile_callback.py`](../scripts/benchmark/torch_profile_callback.py) + [`analyze_torch_profile.py`](../scripts/benchmark/analyze_torch_profile.py)
- 相关 ms-swift upstream PR：[#9162](https://github.com/modelscope/ms-swift/pull/9162) · [#9167](https://github.com/modelscope/ms-swift/pull/9167) · [#9189](https://github.com/modelscope/ms-swift/pull/9189)
- 原 baseline（Qwen2.5-7B × seq=4096）：`docs/fsdp_vs_benchmark_report.md`
