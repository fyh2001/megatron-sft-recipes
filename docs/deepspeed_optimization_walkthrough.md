# DeepSpeed ZeRO-3 + Ulysses SP 优化实战 walkthrough（Qwen3.5-9B）

> 与 [`fsdp2_optimization_walkthrough.md`](fsdp2_optimization_walkthrough.md) 配套。本文专注 **DeepSpeed ZeRO-3 + ms-swift Ulysses SP** 后端。
>
> 实验日期：2026-04-23/24 · 硬件 8×H100 80GB · Qwen3.5-9B (freeze ViT) · seq=16384 · ms-swift 4.2.0.dev0 git main · DeepSpeed 0.18.9

> **环境准备 / MFU 口径辨析 / DCGM & nsys 工具栈**：直接看 [fsdp2 doc §0-§3](fsdp2_optimization_walkthrough.md#0-环境准备)。本文只讲 DS 特有的部分。

---

## TL;DR（必读 ⚠️）

**DS ZeRO-3 + Ulysses SP 在 Qwen3.5 上当前完全不可用**（截至 2026-04-24）：
- 默认 swift 4.1.2：step ~8 必崩（`cudaErrorIllegalAddress`）
- 升级到 swift 4.2.0.dev0 git main：不崩了，但 **loss 从 step 2 起恒 0、grad_norm 恒 √2**——**模型不更新**
- pack + Liger 等所有 FSDP2 优化叠上去都救不回来——根因是 DS ↔ swift Ulysses SP 集成 bug

**如果你不需要 Ulysses SP**（只用 DS ZeRO-3 省 optimizer state 内存，不切 seq），DS + packing + Liger 是可用的。但**无法装 Qwen3.5 + seq=16384**——单卡 activation 放不下。所以这条路**对 9B + seq=16k 也不可用**。

**当前唯一推荐**：用 [FSDP2 pack_liger](fsdp2_optimization_walkthrough.md#9-最终配置--一键生产训练命令) 或 [Megatron TP=4 SP](megatron_optimization_walkthrough.md)。本文剩下的内容是给**等 modelscope 修 bug 后**的复用记录，以及**如何检测你机器上是不是同一个 bug**。

---

## 目录

- [1. Baseline & loss=0 bug 复现](#1-baseline--loss0-bug-复现)
- [2. 怎么检测你也撞到这个 bug](#2-怎么检测你也撞到这个-bug)
- [3. 已尝试无效的"修复"](#3-已尝试无效的修复)
- [4. 理论上能用的退化路径（不带 SP）](#4-理论上能用的退化路径不带-sp)
- [5. 等 upstream 修了之后该怎么优化（参照 FSDP2）](#5-等-upstream-修了之后该怎么优化参照-fsdp2)
- [6. DS 特有的几个 config 注意](#6-ds-特有的几个-config-注意)

---

## 1. Baseline & loss=0 bug 复现

### 1.1 启动命令

```bash
docker exec fsdp_sft bash -lc "cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  MASTER_PORT=29500 \
  BACKEND=ds SP=2 MBS=1 \
  TOTAL_STEPS=30 WARMUP_BENCH=3 \
  DS_CONFIG=/home/ubuntu/perf_opt/megatron-sft-recipes/scripts/benchmark/sp_offload_configs/zero3_nopin.json \
  RUN_NAME=ds_baseline \
  BENCH_DIR=/home/ubuntu/perf_opt/megatron_output/walkthrough \
  bash scripts/benchmark/bench_swift_sp_v2.sh"
```

转换后实际跑的 `swift sft` 命令（关键 flag）：

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
    --max_steps 30 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --gradient_checkpointing true \
    --freeze_vit true --freeze_aligner true \
    --deepspeed scripts/benchmark/sp_offload_configs/zero3_nopin.json \
    --sequence_parallel_size 2 \
    --dataloader_num_workers 2 \
    --dataset_num_proc 4 \
    --save_strategy no \
    --logging_steps 1 \
    --output_dir <BENCH_DIR>/ds_baseline
```

`zero3_nopin.json` 内容（最简 ZeRO-3，无 offload）：

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "none"},
    "offload_param": {"device": "none"}
  },
  "bf16": {"enabled": true},
  "gradient_clipping": 1.0,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 1
}
```

### 1.2 实测结果（30 步）

| 指标 | 值 |
|---|---:|
| step time | 2.24 s |
| peak mem | 35.9 GiB |
| swift 公式 MFU | 39.7% |
| **loss** | **首步 12.84，第 2 步起恒 0** |
| **grad_norm** | **首步 24.89，第 2 步起恒 √2 ≈ 1.4142** |
| 30 步全部完成 | ✅（不崩，但模型没训） |

bench.jsonl 摘录：

```json
{"step": 1, "loss": 12.835, "grad_norm": 24.894, ...}
{"step": 2, "loss": 0.000,  "grad_norm": 1.414,  ...}
{"step": 3, "loss": 0.000,  "grad_norm": 1.414,  ...}
{"step": 4, "loss": 0.000,  "grad_norm": 1.414,  ...}
... 一直到 step 30 都是 loss=0, grad_norm=1.414 ...
```

### 1.3 这是什么意思

**`grad_norm = 1.414... = √2`** 是 HuggingFace Trainer 在检测到 NaN/Inf gradient 时 fallback 的"安全替代值"——**意味着 DS 真实算出的 grad 已经全是 NaN**。Trainer 把它替换成 √2 让 `clip_grad_norm` 不爆炸，然后正常调用 optimizer.step()——但因为 grad 是"假"的，**模型实际上没有得到任何有效更新**。

loss 恒 0 是因为 forward 也烂了（或者 NaN 沿 forward 传播被 mask 成 0）。

GPU 在跑（功耗 ~617 W 峰值），但跑的是空气。

---

## 2. 怎么检测你也撞到这个 bug

不要相信"step time 看着正常"。直接看 bench.jsonl：

```bash
# 跑短训练（10-30 步够了）
docker exec fsdp_sft bash -lc "cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  BACKEND=ds SP=2 MBS=1 TOTAL_STEPS=10 WARMUP_BENCH=2 \
  DS_CONFIG=scripts/benchmark/sp_offload_configs/zero3_nopin.json \
  RUN_NAME=detect_bug \
  BENCH_DIR=/tmp/ds_check \
  bash scripts/benchmark/bench_swift_sp_v2.sh"

# 查 loss / grad_norm
python3 -c "
import json
with open('/tmp/ds_check/detect_bug/bench.jsonl') as f:
    rows = [json.loads(l) for l in f if l.strip()]
losses = [r['loss'] for r in rows if r.get('loss') is not None]
grads = [r['grad_norm'] for r in rows if r.get('grad_norm') is not None]
zeros = sum(1 for l in losses if l == 0)
sqrt2 = sum(1 for g in grads if isinstance(g,(int,float)) and abs(g - 1.41421356) < 0.001)
print(f'loss: first={losses[0]:.3f}, last={losses[-1]:.3f}, zeros={zeros}/{len(losses)}')
print(f'grad_norm: first={grads[0]}, last={grads[-1]}, sqrt(2)-count={sqrt2}/{len(grads)}')
print('VERDICT:', 'BUG TRIGGERED, model not training' if zeros > 5 else 'looks ok')
"
```

如果输出 `BUG TRIGGERED, model not training`，那就是同一个 bug，不要继续用 DS+SP。

---

## 3. 已尝试无效的"修复"

### 3.1 ❌ 升 ms-swift 到 git main

升级到 ms-swift 4.2.0.dev0 (git main)，吃到 [PR #9162](https://github.com/modelscope/ms-swift/pull/9162) / [#9167](https://github.com/modelscope/ms-swift/pull/9167) / [#9189](https://github.com/modelscope/ms-swift/pull/9189) 的 SP 修复。

| | swift 4.1.2 (pypi) | swift 4.2.0.dev0 (git main) |
|---|---|---|
| FSDP2 + Ulysses SP | ❌ step ~8 崩 | ✅ 30 步 loss 正常 |
| **DS + Ulysses SP** | ❌ step ~8 崩 | ⚠️ **不崩了，但 loss=0 grad=√2** |

升 main 修了 FSDP2 这条路，但 **DS 这条路没修干净**。modelscope 的 PR description 也只展示了 FSDP2 的训练曲线，没贴 DS 的。

### 3.2 ❌ 加 packing + Liger（pack_liger 套路）

把 FSDP2 的最优套路 `pack + liger + MBS=2` 全套到 DS 上：

```bash
docker exec fsdp_sft bash -lc "cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  BACKEND=ds SP=2 MBS=2 \
  PACKING=true USE_LIGER=true \
  DS_CONFIG=scripts/benchmark/sp_offload_configs/zero3_nopin_auto.json \
  TOTAL_STEPS=100 WARMUP_BENCH=5 \
  RUN_NAME=ds_pack_liger \
  BENCH_DIR=/home/ubuntu/perf_opt/megatron_output/walkthrough \
  bash scripts/benchmark/bench_swift_sp_v2.sh"
```

注意要用 [`zero3_nopin_auto.json`](../scripts/benchmark/sp_offload_configs/zero3_nopin_auto.json)（`train_micro_batch_size_per_gpu: "auto"`），否则 DS 会拒绝 MBS≠1 的设置。

实测 100 步：
- step time 2.77 s（看着正常）
- peak mem 53 GiB
- peak power 617 W
- **loss step 1 = 12.91, step 2-100 全部 0**
- **grad_norm step 1 = 13.07, step 2-100 全部 1.41421356**

**bug 完全没绕过**。pack/liger 在 DS 上和裸 baseline 一样没救。

### 3.3 ❌ 把 swift 自己的 `swift_sp_patch.py` shim 加上

早期我们写了 `scripts/benchmark/swift_sp_patch.py` 给 swift 4.1.2 打 transformers 5.5 mask API 签名补丁。在 swift 4.2.0.dev0 main 上理论上不需要，但试过加上没区别——因为 main 已经包含等价 fix（PR #9167）。bug 不在 mask 签名层，在更深的 SP label-shard 层。

### 3.4 ❌ 关 `--gradient_checkpointing`（NO_AC 等价物）

```bash
GRAD_CKPT=false ... bash bench_swift_sp_v2.sh
```

DS 的 `--gradient_checkpointing false` ≈ FSDP2 的 NO_AC=true。试了，**loss=0 bug 不变**。这是 SP 集成层的 bug，和 activation checkpointing 无关。

### 3.5 ❌ 各种 NCCL 调优

如 [fsdp2 doc §7.2](fsdp2_optimization_walkthrough.md#72--nccl-nvls-算法fsdp2-小消息不适合) 所述，NCCL NVLS 在 FSDP2 上反而慢。DS 上没专门测，但 **loss=0 是数值 bug，不是性能问题**——任何性能优化都没用。

---

## 4. 理论上能用的退化路径（不带 SP）

如果你**真的必须用 DeepSpeed**（比如有现成的 DS + Megatron-DeepSpeed 训练 pipeline），可以**关掉 Ulysses SP**：

```bash
# 关掉 --sequence_parallel_size，改成 1（=不开 SP）
docker exec fsdp_sft bash -lc "cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  BACKEND=ds SP=1 MBS=1 \
  PACKING=true USE_LIGER=true \
  DS_CONFIG=scripts/benchmark/sp_offload_configs/zero3_nopin.json \
  TOTAL_STEPS=10 WARMUP_BENCH=3 \
  RUN_NAME=ds_no_sp \
  BENCH_DIR=/tmp/ds_check \
  bash scripts/benchmark/bench_swift_sp_v2.sh"
```

**预期问题**：每 rank 处理完整 seq=16384 token 的 activation。Qwen3.5-9B + seq=16k + bf16，per-rank activation 约 35-40 GB。加上 ZeRO-3 sharded params + grads + optimizer states 约 12 GB（9B/8 × ~10 bytes/param）。**单卡 ≈ 50 GB**，理论上 H100 80GB 装得下。**但本轮没实测**。

如果你跑这个能拿到正常 loss（非 0），那就是 SP 路径专属 bug；模型本身在 DS+ZeRO3 下能训。

如果跑 OOM，那就只能：
- 加 `--gradient_checkpointing true`（吃 33% 计算，省 activation）
- 或开 `offload_optimizer: {"device": "cpu"}` 把 optimizer states 卸到 CPU（吃通信 overhead）

这两条都比 FSDP2 pack_liger 慢得多，没什么实用价值。

---

## 5. 等 upstream 修了之后该怎么优化（参照 FSDP2）

如果将来 modelscope 修了 DS + Ulysses SP 的 loss=0 bug，把 FSDP2 的优化套路平移过来。可移植性矩阵：

| FSDP2 优化轴 | DS 等价物 | 备注 |
|---|---|---|
| `--packing true` | 完全一样 `--packing true` | swift CLI flag |
| `--use_liger_kernel true` | 完全一样 | DS 走 swift Trainer，Liger monkey-patch 同样起作用 |
| MBS=2 | 改 DS config：`train_micro_batch_size_per_gpu: "auto"`（用 [zero3_nopin_auto.json](../scripts/benchmark/sp_offload_configs/zero3_nopin_auto.json)）+ `--per_device_train_batch_size 2` | DS config 不能写死 MBS，否则 swift 报 mismatch |
| NO_AC（关 act ckpt） | `--gradient_checkpointing false`（**注意**：DS 用这个 flag，不像 FSDP2 在 fsdp_config 里） | swift CLI flag |
| `reshard_after_forward=false`（FSDP2 ZeRO-2） | DS 不能直接做。等价物是改 `stage: 2`（DS ZeRO-2），但 mem ×3 | 不推荐 |
| Ulysses SP=2 | `--sequence_parallel_size 2` | 一样的 swift flag |

预估命令（**等 bug 修了再用**）：

```bash
docker exec fsdp_sft bash -lc '
NPROC_PER_NODE=8 swift sft \
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
    --learning_rate 1e-5 --warmup_ratio 0.05 \
    --packing true \
    --use_liger_kernel true \
    --gradient_checkpointing false \
    --freeze_vit true --freeze_aligner true \
    --deepspeed /home/ubuntu/perf_opt/megatron-sft-recipes/scripts/benchmark/sp_offload_configs/zero3_nopin_auto.json \
    --sequence_parallel_size 2 \
    --dataloader_num_workers 2 \
    --dataset_num_proc 4 \
    --save_strategy steps --save_steps 200 \
    --logging_steps 5 \
    --output_dir /home/ubuntu/perf_opt/megatron_output/prod_ds
'
```

**预估性能**（参照 FSDP2 pack_liger 类比，DS ZeRO-3 通信 overhead 略高）：
- step time: ~3.5 s（FSDP2 pack_liger 是 2.88 s）
- 真 MFU: ~40%（FSDP2 是 50%）
- full-epoch wall: ~12 min（FSDP2 是 9.8 min）

DS ZeRO-3 比 FSDP2 慢的根因：DS 的 param sharding 走 Python-side bucket hooks，比 FSDP2 的 DTensor + 直接 NCCL 多一层 Python 开销（[fsdp2 doc §1.3 / §6](fsdp2_optimization_walkthrough.md) 有数据）。但**只要能跑出正常 loss**，DS 也就够用。

---

## 6. DS 特有的几个 config 注意

### 6.1 `train_micro_batch_size_per_gpu` 必须和 swift 一致

DS 的 config JSON 里这个字段必须等于 swift CLI 的 `--per_device_train_batch_size`，否则启动时报：

```
ValueError: Please correct the following DeepSpeed config values that mismatch
TrainingArguments values:
- ds train_micro_batch_size_per_gpu=1 vs hf per_device_train_batch_size=2
```

**最稳的写法**：DS config 用 `"auto"` 让 swift 注入：

```json
{
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

我们的 [zero3_nopin_auto.json](../scripts/benchmark/sp_offload_configs/zero3_nopin_auto.json) 就是这种写法。

### 6.2 `bf16: {"enabled": true}` 不能省

DS 的 bf16 enable 是在 config JSON 里，不是从 swift 的 `--torch_dtype bfloat16` 自动传过去的。漏写会以 fp32 跑（×2 显存 + 慢一倍）。

### 6.3 `gradient_clipping` 双写

swift 默认 `--max_grad_norm 1.0`。DS config 也有 `gradient_clipping`。两者会双写，**保持一致即可**（都设 1.0）。

### 6.4 不要混用 swift `--fsdp` 和 `--deepspeed`

它们是互斥的——选一个 backend。混用 swift 会拒绝。

### 6.5 ZeRO-3 + offload 是双刃剑

| Config | peak mem | step time | 适用 |
|---|---|---|---|
| ZeRO-3, no offload（[zero3_nopin.json](../scripts/benchmark/sp_offload_configs/zero3_nopin.json)） | ~36 GB | 2.24 s | 默认推荐 |
| ZeRO-3 + optimizer offload to CPU | ~19 GB | **9.22 s（×4 慢）** | 仅在 OOM 时用 |
| ZeRO-3 + param offload to CPU | 更省 | 更慢 | 几乎不用 |

每省 1 GB 显存付出 ~0.4 s/step。**只在装不下时才考虑 offload**。

---

## 写在最后

这份 walkthrough 主要是**踩坑记录**——告诉你为什么 DS 在 Qwen3.5 + SP 这条路当前不能用、怎么检测 / 绕开。

核心结论：
1. **现状（2026-04-24）**：DS + Ulysses SP 在 Qwen3.5 上 loss=0 bug，**不能生产**
2. **检测方法**：跑 10 步看 grad_norm 是不是恒 √2、loss 是不是恒 0（§2 一行命令）
3. **替代**：用 [FSDP2 pack_liger](fsdp2_optimization_walkthrough.md) 或 [Megatron TP=4 SP](megatron_optimization_walkthrough.md)
4. **等 bug 修了**：套 §5 的命令直接用，预估比 FSDP2 慢 20%

Track 上游 fix：搜 [ms-swift issues / PRs](https://github.com/modelscope/ms-swift/issues?q=deepspeed+ulysses) 关键字 "deepspeed ulysses" 或 "qwen3_5 sp"。
