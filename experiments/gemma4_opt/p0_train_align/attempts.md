# P0g — FSDP2 ↔ DS 训练曲线对齐 attempts

> 见 [walkthrough §0g](../../../docs/gemma4_optimization_walkthrough.md#0g-fsdp2--ds-训练曲线对齐gbs-一致化)。

## misaligned baseline（参考，不是新 run）

`_baselines_misaligned/compare.{tsv,txt}` —— 用 [compare_loss_curves.py](../../../scripts/benchmark/compare_loss_curves.py) 对比 P0b（FSDP2 GBS=4）和 P0c（DS GBS=64）的 logging.jsonl 前 40 步。结果：max abs Δloss/B = **127.2%**，max abs Δgrad/B = **2259%**，**完全不可比**（如预期，4 个轴都不一致）。这次对比仅用来 surface 问题，不算 align 尝试。

## align 尝试时间线

| run | 时间 | 状态 | 一句话结论 |
|---|---|---|---|
| _baselines_misaligned (compare-only) | 2026-04-25 19:55 | REFERENCE | misaligned 现状 surface：max Δloss/B = 127% |
| run_20260425_121042_mirror_ds | 2026-04-25 20:10 | FAILED | swift FSDP2+SHARDED_STATE_DICT 拒 `--save_only_model true`；host `python` 不存在 → gpu_monitor/dcgm 静默 no-op |
| run_20260425_121255_mirror_ds | 2026-04-25 20:12 | FAILED | CUDA OOM 在 step 0/40。global 层 head_dim=512 fallback SDPA math backend，attn matrix 4.3 GB 加在 52 GB FSDP2 idle resident 上撞顶。`truncation=right` 让第 1 样本满 16384 tokens（baseline `delete` resample 到 919 tok 才避开） |
| run_20260425_123301_mirror_ds | 2026-04-25 20:33 | FAILED | offload **生效了** —— forward 跑通（SDPA 4.27 GB 第一次 alloc 成功），OOM 推到 backward。原因：accelerate `cpu_offload` 在 FSDP2 下**只 offload params + grads，不 offload AdamW state**（39 GB master+m+v fp32 仍在 GPU）。peak 78.46 GB |
| run_20260425_123717_mirror_ds | 2026-04-25 20:37 | FAILED | 加了 `paged_adamw_32bit` 没效果。OOM 数字（75.26 GB / 4.27 GB）和 run_03 完全一致 — 因为 optimizer state 是 **lazy 创建**（第 1 次 `optimizer.step()` 才 alloc），但我们 OOM 在 step 0 的 backward，opt state 根本还没到 GPU 上。真正的吃内存大户是 **SDPA O(N²) + fp32 master upcast + 16384×262144 logits**，offload/paged 这条线管不到 |
| run_20260425_125107_mirror_ds | 2026-04-25 20:51 | FAILED | "Invalid backend"。sitecustomize 注入 mem_efficient 生效，但 transformers/integrations/sdpa_attention.py:39 `use_gqa_in_sdpa(attention_mask=None, ...)` 在 torch≥2.5 返回 True → 传 `enable_gqa=True` kwarg → mem_efficient 后端不支持这 kwarg（sdp_gqa_smoke.py 验证）。math 兜底会 OOM 20.5 GB |
| run_20260425_125808_mirror_ds | 2026-04-25 20:58 | FAILED | 进展明显（mem_efficient 真在 GPU 上跑了，OOM 量从 4.27→3.20 GB，peak 76 vs run_03 的 78），但仍差 10-20 GB headroom。fp32 master upcast (~13 GB) 在 GPU 上没动 |
| run_20260425_130059_mirror_ds | 2026-04-25 21:00 | FAILED | swift `activation_cpu_offload` callback bug — `activation_cpu_offload.py:327` assert saved tensor 不是 tuple，但 PyTorch checkpoint 在 FSDP2 上存的就是 tuple → AssertionError on backward |
| run_20260425_130316_mirror_ds | 2026-04-25 21:03 | FAILED | **fwd + bwd + opt-state alloc 全过 ✅**，只在 step 1 末尾的 `clip_grad_norm` 撞 PyTorch FSDP2 + CPUOffloadPolicy 的 NCCL/CPU 不兼容（`No backend type associated with device type cpu`）。grads 在 CPU 上，DTensor.vector_norm 走 NCCL all_reduce 拒收 CPU tensor |
| run_20260426_161445_mirror_ds | 2026-04-27 00:14 | FAILED | **α 成功！fwd + bwd + opt-state lazy-alloc + clip_grad_norm 全过 ✅**。新坑在 `optimizer.step()`：bnb `PagedAdamW32bit.prefetch_state` 调 CUDA `cprefetch(deviceid)` 但 FSDP2 offloaded params 是 CPU DTensor，`device.index=None` → TypeError |
| run_20260426_161834_mirror_ds | 2026-04-27 00:18 | **SUCCESS** ✅ | **40/40 训练全过 + Step 1 loss 等价证实**（Δloss=+0.04%）。tokens 全 40 步精确匹配 DS。Step 2+ 漂移定位为 lr schedule 错位（DS 实跑 max_steps=294→warmup_steps=15，align run 用 40→warmup_steps=2）。Post-tee bash 错误是中途改脚本行号偏移导致，不影响训练数据 |
| run_20260426_165953_warmup15 | 2026-04-27 01:00 | **SUCCESS** ✅ | **40/40 完整跑通**。Step 1-3 Δloss<0.1% ✅（lr 完全对齐到 1.33e-6/2.67e-6/4e-6 与 DS 一致）。Step 4-40 漂移 mean 5% / max 22%，**根因不是 lr 而是 `--attn_impl` 不同**：DS 用 `flash_attention_3`，align 用 `flash_attention_2`（FA3 vs FA2 + SDPA mem_eff vs SDPA math 的反向 numeric 差异，每步 bf16 atomic noise 累积）|

| run_*_warmup15_fa3 (run_12) | 2026-04-27 02:03 | FAILED | OOM at model load。run_11 的 torchrun children PIDs 没被清，每张卡占 30 GB → run_12 启动撞顶。`docker exec fsdp_sft pkill -9 -f swift` 清完，重启为 run_13 |
| run_20260426_180515_warmup15_fa3 (run_13) | 2026-04-27 02:05 | **SUCCESS** ✅✅ | 40/40 跑通。FA2→FA3 切换让 **Step 1 loss bit-identical（2.02941513 == 2.02941513，Δ=0 到 8 位小数）**。Step 2-3 也 < 0.1%。Step 4-40 漂移仍 ~5-7%（mean Δloss=6.3% / max 24.5%），grad_norm 在 step 4/6/12 出现 1048/1184/199 大 spike（DS 同 step 是 63/24/8.7）—— 这是 **FSDP2 + CPU offload + DTensor.linalg.vector_norm** 在 gloo+nccl mixed backend 下的偶发数值不稳定，与 attn_impl 无关 |

## 总结：A 路径完整跑通 + 训练等价边界已 surface

**13 个 run** 把 FSDP2 + offload + 26B + seq=16384 + GBS=64 + Ulysses SP=2 在 80 GB H100 上从"全部 fail" 推到"40/40 完整跑通 + step 1 bit-identical"。

**最终配方（5+1 件套）**：
1. `fsdp` 字串含 `offload` → CPUOffloadPolicy
2. mem_efficient SDPA backend（sitecustomize patch 1/3）
3. `use_gqa_in_sdpa→False` 强制 repeat_kv 预展开（sitecustomize patch 2/3）
4. `cpu:gloo,cuda:nccl` mixed backend（sitecustomize patch 3/3）
5. `--optim adamw_torch`（device-agnostic，对齐 FSDP offload CPU params）
6. `--attn_impl flash_attention_3` + `--warmup_steps 15`（让 step 1 bit-identical）

**训练等价分层结论**：

| 维度 | 对齐程度 | 限制原因 |
|---|---|---|
| Step 1 forward + loss | ✅ **bit-identical**（FA3 后） | — |
| Step 1-3 forward + loss | ✅ Δ < 0.1% | bf16 atomic noise |
| 数据顺序 / shuffle / truncation | ✅ 全 40 步 tokens 精确匹配 | — |
| lr schedule (warmup) | ✅ 完全相同 | — |
| Step 4-40 训练曲线 | ⚠️ mean ~5-7% 漂移 | grad_norm 偶发 spike（FSDP2 DTensor norm + CPU offload + gloo 数值不稳）+ global SDPA mem_eff vs math 微差累积 |
| grad_norm 数值 | ❌ 130% 系统性偏高 | DTensor full L2 vs DS partitioned norm，**计算语义不同**，与训练等价无关 |

**剩余漂移的根因（不修复，超 P0g 范围）**：
- DTensor.linalg.vector_norm 在 CPU-offloaded grads 上偶发 spike，导致 clip_coef ≈ 0 → 该 step 几乎无更新 → loss 滞后
- gemma4 global 层（head_dim=512）DS 走 SDPA math，FSDP2 align 走 SDPA mem_efficient（math 在 FSDP2 native 上必 OOM，无解）

如果要做 bit-identical 训练曲线，需要 PyTorch DTensor + offload 上游修复或 swift trainer 替换 clip 路径。
