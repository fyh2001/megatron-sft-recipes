# Qwen2.5-14B × 三后端性能对比测试计划

> 目标：在 8×H100 80GB 上用三套后端（FSDP2+compile / DeepSpeed ZeRO-3 / Megatron TP=4+SP）对 **Qwen2.5-14B-Instruct** 做 `GBS=256 × MAX_LEN=16384` 的实战 SFT 性能对比，产出 throughput / MFU / DCGM TC active / nsys profile，并写入 [`docs/qwen2_5_14b_benchmark_report.md`](./qwen2_5_14b_benchmark_report.md)。
>
> 本文是在替代 [`docs/qwen3_5_9b_benchmark_plan.md`](./qwen3_5_9b_benchmark_plan.md) 原 Qwen3.5-9B 方案的**修订版**，保留原 plan 不覆盖，以便跨代对比和踩坑记录。

---

## 0 · 为什么换模型：从 Qwen3.5-9B 到 Qwen2.5-14B-Instruct

原 plan 选 **Qwen3.5-9B**，实测过程中发现三个结构性阻塞，必须换模型：

### 0.1 Gated DeltaNet workspace 在 seq=16384 单卡装不下

Qwen3.5-9B 的 text backbone 是 **24 层 Gated DeltaNet (linear_attention) + 8 层 full attention** 的混合结构。24 层 GDN 走 fla 库（flash-linear-attention 0.4.2）的 `chunk_gated_delta_rule` kernel，backward 时需要把所有 chunk 的中间状态同时保留：

```python
# fla/ops/common/chunk_delta_h.py:690
h = k.new_empty(B, NT, H, K, V)         # [1, 256, 16, 128, 128]
v_new = torch.empty_like(u)              # + backward 专用 scratch
```

**实测**（见 §1 附 seq 扫描数据）：**FSDP 单卡 `seq=16384 × MBS=1` 最高只能到 seq=4096**；seq ≥ 5120 `GRAD_CKPT=false` 即撞 80GB 上限，seq ≥ 8192 即使开 `GRAD_CKPT=true` 也 OOM，seq ≥ 14336 Triton autotune 的瞬时 spike 直接把 Megatron TP=2 也打爆。

**推理 vs 训练不是一回事**：HF 模型卡和社区分析（Sebastian Raschka / AIBusinessDispatch 等）反复强调 "GDN state 固定尺寸、能跑 262k 上下文"，但那是**推理时的 KV-state**；训练时 chunk-parallel backward 需要把所有 chunk 状态落盘，workspace 仍然 ∝ seq。

### 0.2 社区 evidence 一致证实此路不通

搜到的直接相关 issue / 官方示例：

1. **ms-swift 官方 [`examples/models/qwen3_5/mcore.sh`](https://github.com/modelscope/ms-swift/blob/main/examples/models/qwen3_5/mcore.sh)** 对 Qwen3.5-35B 也只敢用 `max_length=2048 + LoRA rank=8 + recompute_granularity=full + sequence_parallel=true`；dense 9B 的官方推荐同样是 LoRA + 短 seq，从没声称过 "full-parameter + seq=16384" 能跑。
2. **HF 模型卡自白**："If you encounter out-of-memory (OOM) errors, consider reducing the context window."
3. **ms-swift #8181**：Qwen3.5 + `sequence_parallel_size` 在 transformers 后端直接挂，官方回复 "please use megatron-swift"。
4. **fla #241**（Gated DeltaNet shared memory 溢出，需手动降 block_size `BV`）+ **fla #790**（`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 在 Blackwell 上 autotune 选错 config 会 silent 精度发散）—— fla 库在长序列 GDN 上本身就是还在快速迭代、时不时出 silent bug 的区域。

### 0.3 尝试 **Ministral-3-14B-Instruct-2512** 作为替代，但 `mcore_bridge` 不支持

读过 model card 后首选过 mistralai/Ministral-3-14B，dense transformer 没 GDN 问题，但：
- `Instruct` 只发 **FP8** 格式，需要 `FineGrainedFP8Config(dequantize=True)` 反量化到 BF16 才能 SFT，多一道转换且上游刚支持；
- **`mcore_bridge 1.1.2` 没有 `mistral3` / `ministral` bridge**（实测 `mcore_bridge/model/mm_gpts/` 没对应文件），Megatron 这一栏会 DOA。

### 0.4 切到 **Qwen2.5-14B-Instruct**：所有约束解除

- **dense transformer**（48 层 × hidden=5120 × vocab=152064），标准 flash-attn，**无 GDN workspace 问题**；
- BF16 原生权重，无 FP8 反量化；
- 与 Qwen2.5-7B-Instruct **同架构家族**，跨代对比最干净（上一份 [`docs/fsdp_vs_megatron_report.md`](./fsdp_vs_megatron_report.md) 的基线就是 Qwen2.5-7B）；
- `mcore_bridge 1.1.2` 通用 LLM bridge 支持，ms-swift 4.1.2 全路径验证过；
- 参数 14.77 B（7.6B 的 1.94×，跨代把"模型变大"这个维度也覆盖进去）。

---

## 1 · 前置事实（已核对）

### 1.1 模型

```
Qwen/Qwen2.5-14B-Instruct
  num_params       : 14,770,033,664
  model_type       : qwen2
  num_hidden_layers: 48
  hidden_size      : 5120
  vocab_size       : 152064
  FSDP wrap cls    : Qwen2DecoderLayer
  raw bf16 size    : 27.5 GB
  shard (world=8)  : 3.44 GB/rank ← fsdp_diag 校验精确匹配
```

### 1.2 容器内软件栈

```
torch        2.10.0+cu129        transformers 5.5.4
accelerate   1.13.0              deepspeed    0.18.9
mcore_bridge 1.1.2               megatron.core 0.16.1
swift        4.1.2               flash_attn   2.8.3
```

均与上一份 Qwen2.5-7B 报告同镜像：`modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.0-modelscope1.35.4-swift4.1.2`

### 1.3 硬件

- 8 × NVIDIA H100 80GB HBM3（SXM）
- `dcgm-exporter-fast` 容器，host port 9500，`-c 1000` = 1 s collect interval
- 实测 `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` / `GR_ENGINE_ACTIVE` / `POWER_USAGE` 都可采

### 1.4 数据

- `sft-data/train.jsonl`：**18,819 条**多轮对话
- tokenizer：Qwen/Qwen2.5-14B 的 fast tokenizer

**实测 token 长度分布**（把 chat_template 套上去，逐 message tokenize）：

| 分位 | token 数 |
|---|---:|
| min / median / max | 16 / **1,464** / 17,011 |
| p75 / p90 | 2,996 / 7,726 |
| p95 / p99 | 12,041 / 13,641 |
| > 4096 | 3,611 (19.2%) |
| > 8192 | 1,750 (9.3%) |
| > 12288 | 886 (4.7%) |
| > 16384 | **3 条 (0.016%)** |

`max_length=16384` 下 **3 条样本**被 truncation（`truncation_strategy=delete` 或右截），其余 18,816 条完整训练。

---

## 2 · 关键设计决定

### 2.1 参数对齐（三后端一致）

| 参数 | 值 | 与 plan §1.1 (Qwen3.5-9B 原 plan) 对比 |
|---|---|---|
| MAX_LEN | 16,384 | 同 |
| MBS | 1 | 同 |
| **GBS** | **256**（GAS=32） | plan 原 384，**降** 1/3 以在 4 h 时间预算内完成三后端 |
| tokens/step | **4.19 M**（= 256 × 16384） | plan 原 6.29 M |
| TOTAL_STEPS | 15 | 同 |
| WARMUP_BENCH | 5 | 同 |
| seed / lr / warmup_ratio / bf16 | 42 / 1e-5 / 0.1 / on | 同 |
| ATTN_IMPL | flash_attention_2 | 同 |

### 2.2 后端 × ckpt 策略（三家差异）

实测（见 §3）得出的最终矩阵：

| 后端 | 并行策略 | 激活重算 | 原因 |
|---|---|---|---|
| **FSDP2 + compile** | FULL_SHARD × 8（ZeRO-3 风格） | **被迫** `GRAD_CKPT=true`，走 **transformers 原生 `gradient_checkpointing_enable`** (`FSDP_ACX=false`) | no-ckpt 撞 79 GB OOM；`fsdp_activation_checkpointing: true` 触发 flash_attn KV metadata 不匹配的 `CheckpointError` bug |
| **DeepSpeed ZeRO-3** | ZeRO-3 × 8，**强制 `compile=false`** | 被迫 `GRAD_CKPT=true` | DS 0.18 + torch.compile 组合已知不稳；no-ckpt 同样 OOM |
| **Megatron** | **TP=4 PP=1 + SP**（由 plan 原 TP=2 升级） | **`RECOMPUTE=none`**（相比 plan 原 selective/full 降档） | TP=4+SP 把 seq 切 4 份、每 TP rank 只看 seq=4096，50.8 GB/卡，**唯一能关 recompute 的后端** |

**和 plan 原本差异最大的一点**：
- plan 原本假设 "FSDP/DS 边缘 ~55-70 GB，Megatron TP=2+SP 宽裕"。实测**完全反过来** —— FSDP/DS 都撞墙必须开 ckpt，**Megatron TP=4+SP 才是唯一宽裕**的后端。

### 2.3 MFU 口径

`report.py` 的 `6N × tok/s / peak_TFLOPS` 是 Chinchilla 标准算法口径，但对 "被迫开 recompute" 的后端会**低估实际工作量**（recompute 把 forward 又跑一遍 —— 多做的 FLOPs 没进分母）。报告 §5 会**给三个口径并列**：

| 口径 | 含义 | 对今天的意义 |
|---|---|---|
| 6N 口径（算法） | `6 × N × tok/s / peak` | 跨后端可直接比；FSDP/DS 侧会显得"偏低"因为 recompute 不计 |
| **6N × ckpt 因子** | 对 FSDP/DS 乘 1.5（fwd 重算一遍），Megatron 不乘 | 近似真实 GPU 工作率 |
| **DCGM TC active** | `PIPE_TENSOR_ACTIVE` 硬件实测 | 最真实，不理会口径 |

### 2.4 `train.py` 的 gradient-accumulation 写法修正

**原 [`scripts/fsdp/train.py`](../scripts/fsdp/train.py) 用 `accelerator.accumulate(model)` 的写法有两个潜在问题**（在今天 bench 前发现并修复，commit 中）：

1. `Accelerator()` 没传 `gradient_accumulation_steps`，accelerate 内部 GAS=1 → `sync_gradients` 永远 True → `if not accelerator.sync_gradients: continue` 永不触发 → **`optimizer.step()` 每个 microbatch 都执行**。结果 plan 的 GBS=384 (GAS=48) 实际跑出来只是 GBS=8 (每 microbatch 一次 step)，**梯度根本没累加**。
2. 若改成 `Accelerator(gradient_accumulation_steps=N)` 走 `GradientAccumulationPlugin`，则在非同步 microbatch 走 `model.no_sync()`，**FSDP2 下 reduce-scatter 被跳过，每 rank 累 N 份全量未 shard 梯度**，对 14B 模型等于 ~7 GB/rank 多占。

**修正方案**（干净的、两个 bug 都避开）：**彻底不用 `accelerate.accumulate()` 上下文，手工 loop**：

```python
outputs = model(**batch)
loss = outputs.loss / gas         # 归一化
accelerator.backward(loss)         # 每次都 reduce-scatter (FSDP2 shard 保留)
_micro_in_step += 1
if _micro_in_step < gas:
    continue                       # 还没到 optimizer step
# 到 GBS 边界：
accelerator.clip_grad_norm_(...)
optimizer.step()
scheduler.step()
optimizer.zero_grad()
global_step += 1
_micro_in_step = 0
```

这样每次 `backward()` 都触发 FSDP2 reduce-scatter，gradient memory 恒定为 `params/world/rank = 3.44 GB/rank`，不会随 GAS 叠加。**也是本轮修正的一个副产品：上一份 Qwen2.5-7B 的 bench 在 GAS=1 下不受影响，但若换到 GAS>1 是有隐患的**。

### 2.5 其他实现细节

- **`sft-data/train.jsonl` 真实数据 × `PAD_TO_MAX=true`**：每批 pad 到 MAX_LEN=16384，torch.compile 不触发 re-compile；Megatron 侧走 `--packing true`。
- **`fsdp_activation_checkpointing: false` (FSDP 侧)**：避开 flash_attn_2 + Qwen2 KV 的 recompute metadata bug；由 `train.py` 直接调 `model.gradient_checkpointing_enable(use_reentrant=False)` 实现 ckpt。实测 FSDP2 sharding 在 transformers 5.5.4 下不会被 transformers 原生 ckpt 的模块重写破坏（这是相比 Qwen2.5-7B 报告记录的情况的一个新发现 —— 旧报告担心的 "TRANSFORMER_BASED_WRAP 匹配不到 Qwen2DecoderLayer" bug 已经不复现）。
- **Megatron `--recompute_method uniform --recompute_num_layers 1`**：在 `RECOMPUTE=full` 时必须给，不给会 raise ValueError；当前 `RECOMPUTE=none` 不会用到，但 `bench_megatron.sh` 已把开关做通用以防后续需要降档。
- **`scripts/benchmark/_inspect_model.py`**：加了 `AttributeError` fallback（从 Qwen3.5 VLM wrapper config 的 text_config 回退探测），兼容未来 VLM 模型。

---

## 3 · Smoke 诊断数据（实测）

### 3.1 FSDP `seq` 扫描（单卡容纳上限）

MBS=1 × GAS=1 × NPROC=8 × bf16 × flash_attention_2：

| SEQ | GRAD_CKPT=false | GRAD_CKPT=true (FSDP_ACX=false) |
|---:|---|---|
| 2,048 | ✅ OK (561 ms/step) | — |
| 4,096 | ✅ OK (1,324 ms/step) | — |
| 5,120 | ❌ OOM (64.12 + 4.74 GiB) | — |
| 6,144 | ❌ OOM (75.53 + 5.68 GiB) | — |
| 7,168 | ❌ OOM (74.61 + 6.63 GiB) | — |
| 8,192 | ❌ OOM (76.33 + 7.58 GiB) | ⚠ 需走 transformers 原生（FSDP plugin 路径踩 metadata bug） |
| 16,384 | ❌ OOM (78.80 + 432 MiB) | ✅ **67.9 GB peak**（compile=false）/ **53.8 GB**（compile=true） |

**结论**：FSDP 在 seq=16384 × MBS=1 × 14B 上 no-ckpt 结构性塞不下，差 ~400 MB；必须开 ckpt。

### 3.2 DeepSpeed `seq=16384` 诊断

| 配置 | 结果 | peak |
|---|---|---:|
| DS no-ckpt | ❌ OOM (78.80 + 432 MiB) | — |
| **DS ckpt=true** | ✅ | **76.3 GB** (95% 容量) |

DS ZeRO-3 的 memory profile 与 FSDP 类似但略高（FSDP2 `reshard_after_forward` 更激进）。

### 3.3 Megatron 调参扫描（**决定性实验**）

GBS=8 seq=16384 TOTAL_STEPS=4：

| Config | train_speed | memory (GiB) | 相对 plan baseline |
|---|---:|---:|---|
| **[plan baseline]** TP=2 full+uniform/1 MBS=1 | 14.85 s/it | 56.5 | — |
| TP=2 selective MBS=1 | — | — | ❌ OOM（selective 对 flash-attn 几乎不省 activation） |
| TP=4 full+uniform/1 MBS=1 | 17.42 s/it | 42.5 | 慢 17%（TP=4 通信 + full recompute 双重开销） |
| TP=4 selective MBS=1 | 13.81 s/it | 53.1 | 快 7% |
| **TP=4 none MBS=1** ← 采用 | **13.03 s/it** | **50.8** | **快 12%，最优** |
| TP=4 selective MBS=2 | — | — | ❌ OOM |

**关键发现**：
1. plan 假设 TP=2 是 sweet spot，实测在 seq=16384 下 **TP=4 + SP 才是**。SP 把 seq 切 4 份，每 TP rank 只看 seq=4096 的 activation + attention，**省 25 GB**，空间足以关掉 recompute。
2. TP=4 + full-recompute 反而比 TP=2 + full-recompute 慢（17.42 vs 14.85），说明 **TP=4 的通信开销只有在能换来关闭 recompute 时才划算**。
3. **和上一份 Qwen2.5-7B 报告的故事反转**：7B × seq=4096 时 FSDP 快 64%（那时 Megatron 的 TP=2 over-engineered）；**14B × seq=16384 时 Megatron 的 TP+SP 是唯一能关 recompute 的后端**，可能反超。

---

## 4 · 时间预算与资源估算

按 GBS=256 × seq=16384（4.19 M tokens/step）外推：

| 后端 | per-microbatch | microbatches/DP/step | per-step | 15 steps 主 bench |
|---|---:|---:|---:|---:|
| FSDP compile=true + ckpt=true | ~5,600 ms | 32 | ~180 s | **45 min** |
| DeepSpeed ZeRO-3 + ckpt=true | ~7,100 ms | 32 | ~230 s | **57 min** |
| Megatron TP=4 PP=1 no-recompute | ~3,300 ms × 2 (mbatch=2 per TP⇥DP) | — | ~420 s | **104 min** |
| **主 bench 合计** | | | | **~3.4 h** |
| Nsys profile × 4（GBS=8 × 10 步 synthetic） | | | | ~15 min |
| **总计 (含报告撰写 ~20 min)** | | | | **≤ 4 h** ✓ |

在 plan 原 4 小时停止线内。

---

## 5 · 实施步骤

### 5.1 阶段 0 · 准备（已完成）

- [x] 模型：`Qwen/Qwen2.5-14B-Instruct` HF 下载 28 GB → `/root/.cache/huggingface/models/Qwen/Qwen2.5-14B-Instruct`
- [x] 代码补丁：
  - [`scripts/fsdp/train.py`](../scripts/fsdp/train.py)：`AutoModelForCausalLM → AutoModelForImageTextToText` 回退；**手工 GAS accumulation**（替代 `accelerator.accumulate()`）
  - [`scripts/benchmark/bench_fsdp.sh`](../scripts/benchmark/bench_fsdp.sh)：`FREEZE_VISION` 开关、**`FSDP_ACX` 旁路**
  - [`scripts/benchmark/bench_megatron.sh`](../scripts/benchmark/bench_megatron.sh)：`FREEZE_VIT` 开关、`RECOMPUTE_METHOD / RECOMPUTE_NUM_LAYERS` 参数
  - 新 [`scripts/benchmark/bench_deepspeed.sh`](../scripts/benchmark/bench_deepspeed.sh)
  - 新 [`scripts/fsdp/accelerate_ds_zero3.yaml`](../scripts/fsdp/accelerate_ds_zero3.yaml)（`gradient_accumulation_steps: 1`）
  - [`scripts/benchmark/_inspect_model.py`](../scripts/benchmark/_inspect_model.py)：加 `AttributeError` fallback
  - [`scripts/benchmark/fsdp_diag.py`](../scripts/benchmark/fsdp_diag.py)：同样加 VLM 回退
- [x] `fsdp_diag.py` 校验：`ratio local/full = 0.1250 (=1/8)`，allocated = 3.44 GB/rank
- [x] DCGM exporter：`dcgm-exporter-fast` 容器，host:9500，`-c 1000`
- [x] Smoke × 6：三后端各 2 组（ckpt × compile 组合）+ Megatron 6 组调参
- [x] 最优配置锁定（本文件 §2.2）

### 5.2 阶段 2 · 主 Bench（~3.4 h）

每组启动前起后台 DCGM scraper：

```bash
docker exec -d fsdp_sft bash -c "python scripts/benchmark/dcgm_scrape.py \
  megatron_output/benchmark/<outdir>/dcgm_tc.tsv http://localhost:9500/metrics"
```

#### 5.2.1 FSDP compile=true × GBS=256 × seq=16384

```bash
docker exec fsdp_sft bash -c '
  cd /home/ubuntu/fyh/megatron-sft-recipes && \
  MODEL=/root/.cache/huggingface/models/Qwen/Qwen2.5-14B-Instruct \
  MBS=1 GAS=32 MAX_LEN=16384 TOTAL_STEPS=15 WARMUP_BENCH=5 \
  COMPILE=true GRAD_CKPT=true FSDP_ACX=false \
  SYNTHETIC=false PAD_TO_MAX=true ATTN_IMPL=flash_attention_2 \
  bash scripts/benchmark/bench_fsdp.sh'
# → megatron_output/benchmark/fsdp/
```

#### 5.2.2 DeepSpeed ZeRO-3 × GBS=256 × seq=16384

```bash
docker exec fsdp_sft bash -c '
  cd /home/ubuntu/fyh/megatron-sft-recipes && \
  MODEL=/root/.cache/huggingface/models/Qwen/Qwen2.5-14B-Instruct \
  MBS=1 GAS=32 MAX_LEN=16384 TOTAL_STEPS=15 WARMUP_BENCH=5 \
  GRAD_CKPT=true SYNTHETIC=false PAD_TO_MAX=true \
  ATTN_IMPL=flash_attention_2 \
  bash scripts/benchmark/bench_deepspeed.sh'
# → megatron_output/benchmark/deepspeed/
```

#### 5.2.3 Megatron TP=4 PP=1 × GBS=256 × seq=16384

```bash
docker exec fsdp_sft bash -c '
  cd /home/ubuntu/fyh/megatron-sft-recipes && \
  USE_MEGATRON_BACKEND=true \
  MODEL=/root/.cache/huggingface/models/Qwen/Qwen2.5-14B-Instruct \
  TP=4 PP=1 MBS=1 GBS=256 MAX_LEN=16384 \
  TOTAL_STEPS=15 WARMUP_BENCH=5 RECOMPUTE=none \
  bash scripts/benchmark/bench_megatron.sh'
# → megatron_output/benchmark/megatron/
```

每组结束：`kill` 对应 scraper；收集 `bench.jsonl` / `train.log` / `report.json` / `gpu_metrics.jsonl` / `dcgm_tc.tsv`。

### 5.3 阶段 3 · nsys profile（在 GBS=8 × seq=16384 × 10 步 × synthetic 上，~15 min）

- [ ] **FSDP compile=true** + 同 ckpt 配置 → `megatron_output/benchmark/fsdp/profile.nsys-rep`
- [ ] **FSDP compile=false** 作 kernel 对照 → `fsdp_no_compile/profile.nsys-rep`
- [ ] **DeepSpeed** → `deepspeed/profile.nsys-rep`
- [ ] **Megatron TP=4 no-recompute** → `megatron/profile.nsys-rep`
- [ ] 四份过 [`nsys_analyze.py`](../scripts/benchmark/nsys_analyze.py) → `profile_analysis/{fsdp,fsdp_no_compile,deepspeed,megatron}_nsys.txt`

### 5.4 阶段 4 · 写报告 [`docs/qwen2_5_14b_benchmark_report.md`](./qwen2_5_14b_benchmark_report.md)

结构（照搬 [`docs/fsdp_vs_megatron_report.md`](./fsdp_vs_megatron_report.md) §1–§7 布局，加 §8 与 Qwen2.5-7B 的跨代对比）：

- **摘要表**：FSDP / DeepSpeed / Megatron 三栏并排，step_time / throughput / MFU(6N) / MFU(with-ckpt) / TC_active / peak_mem / 稳态功耗，两两 delta
- **§1 测试配置**：三栏参数对齐表 + 各自默认差异表
- **§2 核心结果**：GBS=256 × seq=16384 下 throughput / MFU 对比
- **§3 nsys kernel 拆解**：GEMM / NCCL / flash-attn / elementwise / FSDP split-cat / other，wall-clock vs 活跃累计（overlap 率）
- **§4 DCGM TC active 硬件交叉验证**：TC / DRAM / GR active 均值 / median / p95，配合功耗
- **§5 MFU 三口径**：6N / 6N×ckpt / TC_active 并排
- **§6 与 Qwen2.5-7B（上一份报告）跨代对比**：同 FSDP / Megatron，参数从 7.6→14.77B × seq 从 4k→16k，**验证或推翻"FSDP 7-13B 甜蜜点"假设**
- **§7 选型建议**：14B × 长序列 × 实战 GBS 下的三后端推荐矩阵（把结论从"7-13B FSDP"推广到"7-13B FSDP，14B+ × 长序列可能 Megatron"）
- **§8 可复现命令**：三个 bash 片段 + nsys 片段

---

## 6 · 与原 plan ([Qwen3.5-9B](./qwen3_5_9b_benchmark_plan.md)) 的差异汇总

| 维度 | 原 plan | 修订后 | 原因 |
|---|---|---|---|
| 模型 | Qwen3.5-9B VLM (hybrid GDN) | **Qwen2.5-14B-Instruct dense** | GDN 在 seq=16384 结构性塞不下 + Mistral3 无 mcore bridge |
| GBS | 384 (GAS=48) | **256 (GAS=32)** | 被迫开 ckpt/recompute 后，plan 时间预算爆表；降 GBS 仍保 4M tokens/step 量级 |
| FSDP GRAD_CKPT | false 默认 | **true 强制** (FSDP_ACX=false) | seq=16384 × MBS=1 装不下 |
| FSDP vision 处理 | `--freeze_vision` | 不需要（Qwen2.5-14B 是纯 LLM） | N/A |
| Megatron TP | TP=2 PP=1 | **TP=4 PP=1 + SP** | TP=2 + no-recompute OOM；TP=4+SP 省 25 GB 能关 recompute |
| Megatron RECOMPUTE | none 默认 | **none（TP=4 下能保持）** | TP=4 让 "none 仍能跑"成立 |
| 预期 step time | FSDP 85s / DS 120s / Megatron 105s | **FSDP 180s / DS 230s / Megatron 420s** | 参数 2× + seq 4× + 强制 ckpt 1.5× + tokens/step 从 32K→131K 综合 |
| 预期总时 | ~2 h | **~3.4 h** | 4 h 硬上限内 |

---

## 7 · 可能踩到的坑（已预处理）

1. **`accelerator.accumulate()` 的 no_sync FSDP2 内存 blow-up**：已手工替换为 `loss/gas + backward()` 模式，每 microbatch reduce-scatter 保持 shard 梯度。见 [`train.py`](../scripts/fsdp/train.py) 注释。
2. **`fsdp_activation_checkpointing: true` × flash_attn × Qwen2 的 `CheckpointError`**：`bench_fsdp.sh` 加 `FSDP_ACX=false` 旁路，走 transformers 原生 `gradient_checkpointing_enable()`。
3. **DS `gradient_accumulation_steps: auto`**：硬编码 `1`（我们手工管 GAS），避免 `ValueError: invalid literal 'auto'`。
4. **Megatron `RECOMPUTE=full` 不给 method 会 raise ValueError**：`bench_megatron.sh` 加 `RECOMPUTE_METHOD_ARGS` 自动拼 `--recompute_method uniform --recompute_num_layers 1`。
5. **Zombie rank 进程占 GPU 显存**：每次 smoke 结束后主动 `pkill -9 -f train.py`；主 bench 正常退出不会留。
6. **DCGM `PIPE_TENSOR_ACTIVE` 需要 1 s interval**：现有 `dcgm-exporter` 用默认 30 s interval 稀释稳态；独立启 `dcgm-exporter-fast` 容器到 host:9500 解决。
7. **Megatron 首次跑会重建 hf→mcore ckpt（5-10 min）**：主 bench 开跑前用 smoke 把 cache 预热过了。
8. **DCGM 共享采样污染**：主 bench 期间避免其他容器跑 GPU 任务，stuck 时用 `nvidia-smi --query-compute-apps` 清 zombie。

---

## 8 · 输出交付物

1. `megatron_output/benchmark/fsdp/{bench.jsonl,report.json,train.log,gpu_metrics.jsonl,dcgm_tc.tsv,accelerate_config.rendered.yaml}`
2. `megatron_output/benchmark/deepspeed/{bench.jsonl,report.json,train.log,gpu_metrics.jsonl,dcgm_tc.tsv,accelerate_ds_zero3.rendered.yaml}`
3. `megatron_output/benchmark/megatron/{train.log,report.json,gpu_metrics.jsonl,dcgm_tc.tsv}`
4. `megatron_output/benchmark/{fsdp,fsdp_no_compile,deepspeed,megatron}/profile.nsys-rep`
5. `megatron_output/benchmark/profile_analysis/{fsdp,fsdp_no_compile,deepspeed,megatron}_nsys.txt`
6. [`docs/qwen2_5_14b_benchmark_report.md`](./qwen2_5_14b_benchmark_report.md)
7. 代码补丁（一个 commit）：
   - [`scripts/fsdp/train.py`](../scripts/fsdp/train.py) 手工 GAS + ImageTextToText 回退
   - [`scripts/benchmark/bench_fsdp.sh`](../scripts/benchmark/bench_fsdp.sh) `FREEZE_VISION` + `FSDP_ACX`
   - [`scripts/benchmark/bench_megatron.sh`](../scripts/benchmark/bench_megatron.sh) `FREEZE_VIT` + `RECOMPUTE_METHOD_ARGS`
   - 新 [`scripts/benchmark/bench_deepspeed.sh`](../scripts/benchmark/bench_deepspeed.sh)
   - 新 [`scripts/fsdp/accelerate_ds_zero3.yaml`](../scripts/fsdp/accelerate_ds_zero3.yaml)
   - [`scripts/benchmark/_inspect_model.py`](../scripts/benchmark/_inspect_model.py) + [`scripts/benchmark/fsdp_diag.py`](../scripts/benchmark/fsdp_diag.py) `AttributeError` / ImageTextToText 回退
8. 报告（另一个 commit）
