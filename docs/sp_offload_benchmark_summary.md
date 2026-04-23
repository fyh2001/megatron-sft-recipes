# Qwen3.5-9B SP / CP / Offload 三后端 Benchmark 精简报告

> 模型：Qwen3.5-9B VLM（7.94 B trainable，freeze_vit）· 单机 8×H100 80GB · bf16 · seq=16384 · MBS=1 / GBS=8
> 测试日期：2026-04-23 · 完整报告：[`sp_offload_benchmark_report.md`](./sp_offload_benchmark_report.md)

---

## 一、一句话结论

在本机环境下，**FSDP2 + swift Ulysses SP=2** 是综合最优（省内存 + 最快），但**必须升级到 `ms-swift 4.2.0.dev0` (git main)** 才能长跑；若只能用 pypi 版，**Megatron TP=4 SP `recompute=none`** 是唯一可用的生产配置。

---

## 二、核心结果表

baseline 参考（无 SP/CP/offload，Qwen3.5-9B × seq=16384）：FSDP 77 GB / DS 79 GB / Megatron TP=4 SP 41 GB。

| 配置 | peak mem | step time | MFU | 相对 baseline |
|---|---:|---:|---:|---|
| **FSDP2 + Ulysses SP=2** ★ | **26 GB** | **1.40 s** | **64%** | 省 51 GB（−66%），快 2.9× |
| FSDP2 + Ulysses SP=4 | 26 GB | 1.43 s | 62% | 同上，SP=4 无额外增益 |
| DS ZeRO-3 + Ulysses SP=2 | 36 GB | 2.24 s | 40% | 省 43 GB，快 3.1× |
| DS ZeRO-3 + Ulysses SP=4 | 30 GB | 2.12 s | 42% | 省 49 GB，快 3.3× |
| DS SP=2 + CPU optimizer offload | 19 GB | 9.22 s | 10% | 省 60 GB，但**慢 4.1×** |
| Megatron TP=4 SP `recompute=none` ★ | 48 GB | 2.25 s | 40% | 持平 baseline，但**已验 50 步长跑** |
| Megatron TP=2 SP `recompute=selective` | OOM | — | — | 反例：确认装不下 |

---

## 三、长跑稳定性（关键决策依据）

| 配置 | swift 4.1.2 (pypi) | swift 4.2.0.dev0 (git main) | 可生产 |
|---|---|---|---|
| DS + Ulysses SP=2 / SP=4 | ❌ step ~8 必崩 | ⚠️ 不崩但 loss=0，grad 被 clip 成 √2 | ❌ |
| DS + SP=2 + optimizer offload | ❌ 同上 | ⚠️ 同上 | ❌ |
| **FSDP2 + Ulysses SP=2** | ❌ step ~8 崩 | **✅ 30 步 loss 正常，token_acc 0.64→0.84** | **✅ 升 main** |
| **FSDP2 + Ulysses SP=4** | ❌ step 4 label OOR | **✅ 30 步 loss 2.37→0.36** | **✅ 升 main** |
| **Megatron TP=4 SP `recompute=none`** | ✅ 50 步验证通过 | ✅ | **✅ 无需 SP 修复** |

---

## 四、五条要点

1. **FSDP2 >> DS**：同样 SP=2，FSDP2 比 DeepSpeed ZeRO-3 快 1.6×（1.40 vs 2.24 s）、省 10 GB。差距来自 DTensor per-param shard 直接走 NCCL，无 DS Python bucket hook 开销。

2. **Megatron 原生 CP 走不通**：`mcore_bridge/model/mm_gpts/qwen3_5.py` 里硬 assert `context_parallel_size == 1`，因为 GDN 层借壳 HF 实现没接 mcore 原生 CP-aware 模块。修复工作量约半天到一天，本轮未做。

3. **Offload 是内存 ↔ 速度的陡换**：DS SP=2 开 optimizer offload 后内存 36 → 19 GB（−47%），但步长 2.2 → 9.2 s（+311%），每省 1 GB 付出 0.41 s/step。**日常不建议开**，仅在 OOM 时用。

4. **SP=4 拐点已至**：FSDP 下 SP=2/4 速度内存几乎一致，all-to-all 通信开始吃掉 seq 再切的收益；DS 下还有微小收益（param shard 更脏）。**生产推荐 SP=2**。

5. **swift 4.1.2 的 loss=0 bug**：DS / FSDP 两路径都会在 step 3 起 loss=0、grad_norm=NaN，step 8 触发 `cudaErrorIllegalAddress` 全 rank 死。根因是 Ulysses + `truncation=delete` 下 label shard 未正确 mask。**pypi 版必踩**，升 git main 后 FSDP2 路径已修（DS 路径仍未修）。

---

## 五、推荐配置速查

| 场景 | 配置 | 备注 |
|---|---|---|
| **pypi 最新，要稳** | Megatron TP=4 SP `recompute=none` | 50 步长跑已验，MFU 40% |
| **升 ms-swift main，要最快** | FSDP2 + Ulysses SP=2 | 32 GB peak，MFU ~64% |
| **升 ms-swift main，要最省** | FSDP2 + Ulysses SP=4 | 26 GB peak（SP=4 通信开销换内存） |
| **seq > 32 k 或模型 ≥ 30 B** | 再叠 optimizer offload | 仅在前三条装不下时启用 |

---

## 六、关键命令

升级 ms-swift 到 git main（解锁 FSDP2+SP 长跑）：

```bash
docker exec fsdp_sft pip install --force-reinstall --no-deps --no-cache-dir \
    git+https://github.com/modelscope/ms-swift.git@main
```

跑 v2 推荐矩阵（FSDP2 SP=2 / SP=4 各 30 步）：

```bash
docker exec fsdp_sft bash -lc "cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  BACKEND=fsdp2 SP=2 RUN_NAME=fsdp2_sp2_v2 TOTAL_STEPS=30 WARMUP_BENCH=5 \
  BENCH_DIR=/home/ubuntu/perf_opt/megatron_output/bench_sp_offload_v2 \
  bash scripts/benchmark/bench_swift_sp_v2.sh"
```

Megatron TP=4 SP 长跑参考（§9 长跑验证）：

```bash
docker exec fsdp_sft bash -lc '
  cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  USE_MEGATRON_BACKEND=true TP=4 PP=1 CP=1 MBS=1 GBS=8 MAX_LEN=16384 \
  TOTAL_STEPS=50 RECOMPUTE=none FREEZE_VIT=true \
  BENCH_DIR=/home/ubuntu/perf_opt/megatron_output/bench_sp_offload/megatron_tp4_long \
  bash scripts/benchmark/bench_megatron.sh'
```

---

## 七、参考

- 完整报告（含踩坑/诊断/社区 PR 时间线）：[`sp_offload_benchmark_report.md`](./sp_offload_benchmark_report.md)
- 相关 ms-swift upstream PR：[#9162](https://github.com/modelscope/ms-swift/pull/9162) · [#9167](https://github.com/modelscope/ms-swift/pull/9167) · [#9189](https://github.com/modelscope/ms-swift/pull/9189)
- 原 baseline（Qwen2.5-7B × seq=4096）：`docs/fsdp_vs_benchmark_report.md`
