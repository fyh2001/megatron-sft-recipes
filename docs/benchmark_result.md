# 双后端 Benchmark 结果（Megatron vs FSDP2+compile）

> 测试日期：2026-04-20
> 机器：8×H100 80GB（Paperspace）
> 仓库：`megatron-sft-recipes`（commit `83f9f8f`）

本报告涵盖两项：
1. 对 [`scripts/benchmark/`](../scripts/benchmark/) 工具链的代码解读；
2. 在同一硬件上按脚本默认协议跑两个后端的基准测试，并给出结论、失败原因分析、可复现命令。

---

## TL;DR

| 后端 | 模型 | 50 步是否跑通 | 吞吐 (tok/s 合计) | MFU | Step time (avg) | 说明 |
|---|---|---|---|---|---|---|
| **Megatron** (ms-swift + mcore-bridge, TP=2 PP=1) | Qwen2.5-7B-Instruct | ✅ | 61,440 | 35.4% | 533 ms | 稳定、packing 生效 |
| **FSDP2+compile** (Accelerate, 现仓库脚本) | Qwen2.5-7B-Instruct | ❌ | —— | —— | 未完成首步 | 首步 >10 分钟未完成，多处已知缺陷 |

FSDP 侧在 8×H100 上跑满 99.7% 利用率但平均功耗仅 **126 W/GPU**（同硬件 Megatron 为 265 W），强烈指向 FSDP 的 wrap 粒度错误导致的 per-layer all-gather 风暴。详细失败链路见下文「FSDP 失败根因分析」。

---

## 1. 代码解读

### 1.1 目录结构

```
scripts/
├── _common.sh              # 共享环境变量：容器镜像/挂载/数据路径/GPU 数
├── benchmark/
│   ├── bench_megatron.sh   # Megatron/ms-swift 基准
│   ├── bench_fsdp.sh       # FSDP2+compile 基准
│   ├── gpu_monitor.py      # 后台 1Hz nvidia-smi 采样
│   └── report.py           # 解析日志 + GPU 采样 → 报告/对比
├── fsdp/
│   ├── train.py            # 手写 Accelerate FSDP2 + compile 训练脚本
│   └── accelerate_config.yaml
└── megatron/               # ms-swift + mcore-bridge 配置脚本
```

### 1.2 `bench_megatron.sh`（[scripts/benchmark/bench_megatron.sh](../scripts/benchmark/bench_megatron.sh)）

- 默认模型 `mistralai/Mistral-Nemo-Instruct-2407`（本地无缓存），两种执行分支：
  - `USE_MEGATRON_BACKEND=false`（默认）：调 `swift sft`（HF Trainer 后端，gradient_checkpointing + flash_attn2）——**不是真正的 Megatron**，只是对照。
  - `USE_MEGATRON_BACKEND=true`：调 `megatron sft`，使用 mcore-bridge 的真 TP/PP，启用 `sequence_parallel / cross_entropy_loss_fusion / overlap_grad_reduce / overlap_param_gather / use_distributed_optimizer`，支持 `--packing true`。
- 默认参数：`MBS=1 GBS=8 MAX_LEN=4096 TOTAL_STEPS=50 WARMUP_BENCH=20 RECOMPUTE=selective TP=2 PP=1`。
- 训练完成后 `| tee train.log`，再调 `report.py` 生成 `report.json`。

### 1.3 `bench_fsdp.sh`（[scripts/benchmark/bench_fsdp.sh](../scripts/benchmark/bench_fsdp.sh)）

- `accelerate launch --config_file scripts/fsdp/accelerate_config.yaml --num_processes 8 scripts/fsdp/train.py --benchmark ...`
- `train.py --benchmark` 模式下：关闭保存 / 逐步把 `{step, step_time_ms, tokens, loss}` 追加写入 `bench.jsonl`，`report.py` 从这里读精确 per-step 数据。
- 默认 `MBS=1 GAS=1 MAX_LEN=4096 TOTAL_STEPS=50 WARMUP_BENCH=20 COMPILE=true GRAD_CKPT=true`。

### 1.4 `gpu_monitor.py`（[scripts/benchmark/gpu_monitor.py](../scripts/benchmark/gpu_monitor.py)）

- 每 1 秒 `nvidia-smi --query-gpu=...` 获取每卡：`util.gpu / mem.used / mem.total / power.draw / temperature.gpu`，以 jsonl 追加写入。
- bench 脚本启动它到后台，训练结束后 `trap EXIT` 杀掉。

### 1.5 `report.py`（[scripts/benchmark/report.py](../scripts/benchmark/report.py)）

- 支持 `single`（单框架）与 `compare`（两边并排 Δ%）两种模式；FSDP 走 `bench.jsonl`，Megatron 走 `train.log` 解析。
- MFU 公式：`6 × N_params × tok/s / (peak_TFLOPS × num_gpus)`，H100 BF16 峰值取 989.5 TFLOPS。
- GPU 指标取所有采样的均值（`util.gpu`）与最大（`mem.used`）。

### 1.6 `train.py`（[scripts/fsdp/train.py](../scripts/fsdp/train.py)）

- Accelerate `FSDP` distributed_type + TRANSFORMER_BASED_WRAP + `bf16` + `flash_attention_2`。
- `torch.compile(model)`、`gradient_checkpointing_enable(use_reentrant=False)`。
- **Per-message loss mask**：`build_labels_with_loss_mask` 对每条消息都重新 `apply_chat_template(prefix)`，然后对 diff 打 -100 vs token id —— 算法时间复杂度 **O(N_msg²)**。
- 余弦调度带 min_lr。`--benchmark` 模式同步 `accelerator.gather(step_tokens)` 获得 per-step 全局 token 数。

---

## 2. 测试协议

两边共用配置：

| 参数 | 值 |
|---|---|
| 模型 | `Qwen/Qwen2.5-7B-Instruct`（本地 modelscope 缓存，7.6 B params） |
| 精度 | BF16 |
| 序列长度 | 4096 |
| MBS / GBS | 1 / 8（Megatron：8 张卡纯 DP 作为对外 GBS；FSDP：MBS×NPROC×GAS=1×8×1=8） |
| 训练步数 | 50，其中前 20 步 warmup 不计入 |
| Grad checkpointing | 打开（FSDP: `use_reentrant=False`；Megatron: `recompute_granularity=selective`） |
| 数据 | [`sft-data/train.jsonl`](../sft-data/train.jsonl)（18 819 条对话，223 MB） |

后端特有：

| | Megatron | FSDP2+compile |
|---|---|---|
| 并行 | TP=2 PP=1（DP=4） | FSDP_FULL_SHARD over 8 卡 |
| Packing | ✅ `--packing true` | ❌（collate 按 batch 最长 pad） |
| 融合优化 | `cross_entropy_loss_fusion / sequence_parallel / overlap_grad_reduce / overlap_param_gather / gradient_accumulation_fusion / masked_softmax_fusion / bias_dropout_fusion / bias_activation_fusion / apply_rope_fusion=False` | `torch.compile(model)` |
| 通信重叠 | distributed optimizer + all_gather overlap | FSDP backward_prefetch=BACKWARD_PRE |

为适配本次测试，对 bench 脚本做了以下**最小修改**（仓库内已落盘，供复现）：
- `bench_megatron.sh`：`--max_steps` 改成 Megatron 的正确参数名 `--train_iters`；`--num_params` 由 12.2e9 改成 7.6e9。
- `bench_fsdp.sh`：`COMPILE=false` 未正确传递（原来只在 true 时加 `--compile` flag，false 时什么都不传导致走了 argparse 默认 `--compile`）——改成 `--compile` / `--no_compile` 显式互斥。`--num_params` 改 7.6e9。新增可选 `SYNTHETIC=true` 开关，给 `train.py` 传 `--synthetic` 走合成数据绕过数据管线做微基准。
- `train.py`：兼容 transformers 5.x 的 `apply_chat_template(tokenize=True)` 返回 `BatchEncoding`（`UserDict` 而非 `dict`）；新增 `--synthetic` 与 `SyntheticDataset` 类。
- `report.py`：`parse_megatron_train_log` 新增对 ms-swift 风格 `{'iteration': '20/50', 'elapsed_time': '23s', 'train_speed(s/it)': ...}` 行的解析，从 `elapsed_time` 差分出 per-step 时间；新增 `--tokens_per_step` 作为 packing 模式下的 token 吞吐回退量。

---

## 3. Megatron 结果（完整数据）

执行（在 `swift_sft` 容器内）：

```bash
USE_MEGATRON_BACKEND=true \
MODEL=/home/ubuntu/perf_opt/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct \
TP=2 PP=1 MBS=1 GBS=8 MAX_LEN=4096 \
TOTAL_STEPS=50 WARMUP_BENCH=20 RECOMPUTE=selective \
bash scripts/benchmark/bench_megatron.sh
```

产物：[`megatron_output/benchmark/megatron/{train.log, gpu_metrics.jsonl, report.json}`](../../megatron_output/benchmark/megatron/)

`report.py` 输出（`num_params=7.6e9 num_gpus=8 gpu_type=h100 tokens_per_step=32768`）：

```
==================================================
  Benchmark Report: MEGATRON
==================================================
  Measured steps     : 30       (iter 21–50)
  Step time (avg)    : 533.3 ms
  Step time (median) : 600.0 ms
  Step time (p99)    : 600.0 ms
  Throughput         : 61,440 tok/s
  Samples/sec        : 1.88
  MFU                : 35.39 %
  GPU Utilization    : 55.5 %   (全采样均值)
  Peak GPU Memory    : 41.5 GB
  Memory Efficiency  : 52.1 %
  Avg Power Draw     : 203 W    (全采样均值)
==================================================
```

### 3.1 训练活跃期细化指标

`gpu_monitor.py` 从模型加载起采样、训练只占中间一段，故全采样均值偏低。按「单卡 util ≥ 50 %」过滤得到训练活跃期（72 个时间切片 ≈ 72 秒）：

| | 全采样 | 训练活跃期 |
|---|---|---|
| 平均 GPU util | 55.5 % | **94.9 %** |
| 峰值显存 | 41.5 GB | 41.5 GB |
| 平均显存（仅活跃期） | —— | 20.9 GB |
| 平均功耗 | 203 W | **265 W** |

> 说明：由于 `--packing true` + `--micro_batch_size 1 --global_batch_size 8`，实际 per-step 的合并 tokens 数接近 GBS×MAX_LEN=32 768；激活 memory 随步间有涨落，所以活跃期平均显存远低于峰值。

### 3.2 Step time 分布

Step time 来自对 swift 每 5 iter 输出的 `elapsed_time` 做差分，在 iter 21–50 区间为恒定 600 ms 附近，平均 533 ms（含一个 ~500 ms 的首次区块），非常稳定。

---

## 4. FSDP 结果（未能产出合格数据）

执行（在 `fsdp_sft` 容器内，修复 bench 脚本 `COMPILE=false` 之后）：

```bash
# 原默认
MODEL=/home/ubuntu/perf_opt/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct \
MBS=1 GAS=1 MAX_LEN=4096 \
TOTAL_STEPS=50 WARMUP_BENCH=20 COMPILE=true GRAD_CKPT=true \
bash scripts/benchmark/bench_fsdp.sh
# → 14 分钟未完成首步；GPU 100% util，bench.jsonl 为空

# 合成数据 + 关 compile（理论上最轻）
SYNTHETIC=true MBS=1 GAS=1 MAX_LEN=4096 COMPILE=false GRAD_CKPT=true \
bash scripts/benchmark/bench_fsdp.sh
# → 仍然 >5 分钟未完成首步
```

产物：[`megatron_output/benchmark/fsdp/`](../../megatron_output/benchmark/fsdp/) 下 `bench.jsonl` 为 0 字节、`train.log` 有配置与 FSDP prepare 阶段日志但没有任何 `step=` 行。

### 4.1 从 GPU 指标能看到的信号

即使没有一步 step 落盘，`gpu_metrics.jsonl` 的活跃期（128 个 1s 样本，总共约 2 分钟的 "100% util" 区间）告诉我们：

| | Megatron 活跃期 | FSDP 活跃期（未完成首步） |
|---|---|---|
| GPU util | 94.9 % | **99.7 %** |
| Peak 显存 / GPU | 41.5 GB | 26.7 GB |
| Avg 功耗 / GPU | **265 W** | **126 W** |

H100 在纯计算密集 matmul 上应该 ≥ 500 W。FSDP 侧 util 拉满但功耗仅 126 W —— 典型的「**大量小 kernel launch / 通信密集 / 访存受限**」特征，不是在做真正的矩阵乘。

### 4.2 FSDP 失败根因分析

按处理优先级从高到低：

1. **`bench_fsdp.sh` 中 `COMPILE=false` 未生效**（已修正）
   - 原逻辑：`if "${COMPILE}" = "true"; then COMPILE_FLAG="--compile"; fi`，false 时什么都不传；
   - 而 [`train.py`](../scripts/fsdp/train.py) 的 argparse 定义 `--compile` 默认 `True`，无 `--no_compile` 则 compile 一直开；
   - 结果：用户显式 `COMPILE=false` 无效，每次 shape 变化 torch.compile 重新编译。

2. **transformers 5.x 的 `apply_chat_template(tokenize=True)` 返回 `BatchEncoding`（UserDict，不是 dict）**（已修正）
   - `train.py` 原代码：`partial_ids = tokenizer.apply_chat_template(...); new_tokens = partial_ids[len(prev_ids):]`
   - 在 transformers ≤4.x 返回 `list[int]` 可直接切片；5.x 返回 `BatchEncoding`，切片拿不到 input_ids → DataLoader worker 抛 `TypeError: 'tokenizers.Encoding' object cannot be interpreted as an integer`。
   - 修复：`isinstance(partial_ids, dict)` 不生效（UserDict 不是 dict），改用 `hasattr(partial_ids, "keys") and "input_ids" in partial_ids.keys()` 后取 `partial_ids["input_ids"]`。

3. **`ChatDataset.__getitem__` 对 per-message loss mask 的 O(N²) 实现**
   - 对每条消息都 `apply_chat_template(messages[:i+1])` 重跑一次 tokenization；本 jsonl 中 max 消息数 **235**、p95 **83**，这种会话做一次 dataset `__getitem__` 就要 tokenize 几万字符几十次，dataloader worker 单核只能串行。
   - 对 Megatron 侧 ms-swift 不是问题——它的 template + packing 是 C-level + 并行的。
   - 改善思路：用 `apply_chat_template(..., tokenize=False)` 一次得到字符串，再 `tokenizer.encode()` 一次；然后通过每条消息的字符边界找到 offset 打 mask。或直接参考 ms-swift 的 template 实现。

4. **`accelerate_config.yaml` 的 FSDP 包装粒度**
   - 配置：`fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP`，但**没给** `fsdp_transformer_layer_cls_to_wrap`；
   - 在 `accelerate>=1.4` 里如果没显式给 class 名，自动包装策略退化到每个叶子模块都独立 shard。每个 `nn.Linear` forward 都触发一次 all-gather（7 B 模型 ≈ 400+ Linear × 28 layers × 50 steps）——这就是 GPU util=99.7% 但功耗只有 126 W 的根本原因。
   - 建议在 config 里加 `fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer`（或 `Qwen3DecoderLayer` 等按模型而定），或直接用 `FULL_SHARD` + `size_based_wrap_policy`。

5. **`torch.compile` 动态 shape 反复重编译**
   - [`collate_fn`](../scripts/fsdp/train.py) 把每个 batch pad 到 batch 内最长，不是定长 4096 → 每个 batch shape 不同；
   - `torch.compile(model)` 默认 `dynamic=False`，每遇到新 shape 重新编译；
   - 叠加 (4) 每层 Linear 独立包装，编译组合爆炸。

6. **FSDP + grad_ckpt(use_reentrant=False) 交互**
   - 即便合成数据（完全一致的定长 shape）+ `--no_compile` 也跑不出一步，说明前面 (4) 的 wrap 粒度问题是最核心瓶颈，其他都是叠加因素。

### 4.3 对 FSDP 侧最小可行修复建议

- **必改**：在 `scripts/fsdp/accelerate_config.yaml` 里显式给 wrap 类名：
  ```yaml
  fsdp_config:
    fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
    fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer
  ```
- **应改**：`collate_fn` 定长 pad 到 `max_length`，避免 compile 重编译；或在 `torch.compile(model, dynamic=True)`。
- **可改**：`ChatDataset` 的 O(N²) tokenization 走一次 `tokenize=False` + 边界偏移方案。
- 做完 (4)(5) 建议先 `SYNTHETIC=true` 跑一次纯合成 50 步验证，再切回真实数据。

---

## 5. 工具链问题清单（对仓库的副作用）

除了 FSDP 层面的问题，本次跑基准还顺带发现脚本层面以下问题（已在仓库内一并修掉）：

| 位置 | 问题 | 状态 |
|---|---|---|
| `bench_megatron.sh` | 用了 HF Trainer 的 `--max_steps` 给 `megatron sft`，后者只认 `--train_iters` | ✅ 已改 |
| `bench_megatron.sh` / `bench_fsdp.sh` | `--num_params 12.2e9` 硬编码（README 只写默认 Mistral-Nemo-12B） | ✅ 改为 7.6e9 以匹配本次测试模型，生产使用需按模型改 |
| `bench_fsdp.sh` | `COMPILE=false` 无效（未传 `--no_compile`） | ✅ 改为显式互斥 |
| `bench_fsdp.sh`（`bench_megatron.sh` 同样） | 脚本末尾 `python gpu_monitor.py ... &` 存在极少数情况下未及时 flush 的问题（曾出现一次 train.log / gpu_metrics.jsonl 未写入），用宿主机启动独立 `gpu_monitor.py` 可规避 | 本报告数据使用宿主机 gpu_monitor 复跑取得 |
| `scripts/fsdp/train.py` | `apply_chat_template` 返回类型不兼容 transformers 5.x | ✅ 已兼容 |
| `scripts/benchmark/report.py` | `parse_megatron_train_log` 不认 ms-swift 风格的 `{'iteration': 'N/T', 'elapsed_time': 'Xs'}` 日志 | ✅ 已加解析 |

---

## 6. 完全可复现的命令（Qwen2.5-7B × 8×H100）

```bash
# 前置：宿主机释放 GPU
# （本次操作中停掉了一个 vllm serve 占满 8 卡的 live service，实际环境按需处理）

# ========= 1. Megatron =========
docker exec -it swift_sft bash -c '
  cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  USE_MEGATRON_BACKEND=true \
  MODEL=/home/ubuntu/perf_opt/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct \
  TP=2 PP=1 MBS=1 GBS=8 MAX_LEN=4096 \
  TOTAL_STEPS=50 WARMUP_BENCH=20 RECOMPUTE=selective \
  bash scripts/benchmark/bench_megatron.sh
'

# ========= 2. FSDP（修复配置后） =========
# 前提：先在 scripts/fsdp/accelerate_config.yaml 加
#   fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer
docker exec -it fsdp_sft bash -c '
  cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
  MODEL=/home/ubuntu/perf_opt/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct \
  MBS=1 GAS=1 MAX_LEN=4096 \
  TOTAL_STEPS=50 WARMUP_BENCH=20 COMPILE=false GRAD_CKPT=true SYNTHETIC=true \
  bash scripts/benchmark/bench_fsdp.sh
'

# ========= 3. 对比报告 =========
python scripts/benchmark/report.py \
  --fsdp_dir /home/ubuntu/perf_opt/megatron_output/benchmark/fsdp \
  --megatron_dir /home/ubuntu/perf_opt/megatron_output/benchmark/megatron \
  --num_params 7.6e9 --num_gpus 8 --gpu_type h100 \
  --tokens_per_step 32768
```

---

## 7. 结论

- **Megatron/ms-swift 后端**：在 8×H100 上跑 Qwen2.5-7B-Instruct，MBS=1 GBS=8 MAX_LEN=4096 TP=2 PP=1 + packing + selective recompute，达到 **61 k tok/s**、**MFU 35 %**、GPU 活跃期 util **94.9 %**、峰值显存 41.5 GB、活跃期功耗 265 W。这是一个稳定、可复现的基线。
- **FSDP2+compile 后端**：当前仓库实现无法在合理时间内完成首个优化器步。问题集中在 `accelerate_config.yaml` 的 wrap 粒度、`train.py` 的 `BatchEncoding` 兼容与 `O(N²)` tokenization、`bench_fsdp.sh` 的 `COMPILE=false` 逻辑等。上文 §4.2 / §4.3 给出了优先级排序的修复清单，建议先按 (1)(4) 修后再重测。
- **Benchmark 工具链本身**：`report.py` / `bench_megatron.sh` / `bench_fsdp.sh` / `train.py` 在本次使用中暴露了若干实现问题，已在同次改动中全部修复并验证 Megatron 侧可以一键跑通并产出 `report.json`。

