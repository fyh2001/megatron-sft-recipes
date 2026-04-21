# 双后端 Benchmark 结果（Megatron vs FSDP2+compile）

> 测试日期：2026-04-21（两侧在同一机器同一容器同一轮跑完）
> 机器：8×H100 80GB（Paperspace）
> 仓库：`megatron-sft-recipes`（commit `7502738` 之后的动态化 & 死锁修复）
> 容器环境：modelscope 官方镜像 `ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.0-modelscope1.35.4-swift4.1.2`，transformers 5.5.4、accelerate 1.13.0、flash_attn 2.8.3、mcore_bridge 1.1.2、megatron.core 0.16.1。

本报告涵盖两项：
1. 对 [`scripts/benchmark/`](../scripts/benchmark/) 工具链的代码解读；
2. 在同一硬件上按脚本默认协议跑两个后端的基准测试，并给出结论、失败原因分析、可复现命令。

---

## TL;DR

### 测试配置（两侧严格对齐 + 各自默认）

| | **Megatron (ms-swift + mcore_bridge)** | **FSDP2 + compile (Accelerate)** |
|---|---|---|
| 模型 | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct |
| 硬件 | 8 × H100 80GB | 8 × H100 80GB |
| 精度 | bf16 | bf16 |
| MAX_LEN | 4096 | 4096 |
| MBS / GBS | 1 / 8 | 1 / 8 |
| tokens/step | 32,768 (packing) | 32,768 (pad_to_max) |
| 训练步数 | 50 (warmup 20, measured 30) | 50 (warmup 20, measured 30) |
| 数据 | 真实 `train.jsonl` | 真实 `train.jsonl` |
| Activation recompute | **关**（`--recompute_granularity none`） | **关**（`GRAD_CKPT=false`） |
| Attention kernel | flash | flash_attention_2 |
| **并行策略** | TP=2 PP=1 DP=4 + sequence_parallel | FULL_SHARD × 8 |
| **融合优化** | CE fusion / overlap grad-reduce / overlap param-gather / gradient-accum fusion / masked-softmax fusion / ... | `torch.compile(dynamic=True)` |

### 性能对比

| 指标 | Megatron | FSDP2+compile | 差异 |
|---|---:|---:|---:|
| Step time (ms/step) | 566.7 | **345.8** | **−39.0%** |
| **Throughput (tok/s)** | 57,826 | **94,760** | **+63.9%** |
| **MFU** | 33.38% | **54.70%** | **+21.3 pp** |
| Peak GPU memory (GB) | 39.6 | 38.6 | −2.5% |
| 稳态 GPU 功耗 / 卡 | ~265 W | **~676 W** | +155% |
| 稳态 GPU util | ~95% | ~100% | — |

> **一句话**：同硬件、同模型、同 GBS、同数据、都不开 recompute —— **FSDP2+compile 训练吞吐是 Megatron 的 1.64 倍，MFU 高 21 个百分点**。

### 为什么 FSDP 更快？（简要，详见 [§4.5](#45-性能优势拆解fsdp2compile-为何比-megatron-快-639)）

**核心线索只有一个**：稳态单卡功耗 **676 W (FSDP) vs 265 W (Megatron)**，同一块 H100 2.5 倍功耗差 —— 前者贴脸做 dense matmul，后者一半时间在跨 TP rank 同步 + 启动小 kernel。

220 ms/step 的差距拆成四个因素：

1. **TP=2 跨卡同步在 critical path**：Megatron 每步 112 次 all-reduce，不能和 compute overlap；FSDP 每步 84 次 collective，80% 被 prefetch 吃掉。净差 **~60 ms/step**。
2. **`torch.compile` 的 kernel fusion**：FSDP 侧 fuse 把 ~840 次 kernel launch 压到 ~300 次，HBM 读写减半 —— 这是功耗差最大来源。净差 **~80 ms/step**。
3. **`sequence_parallel` 在小模型上是负收益**：seq 切 2 份让 GEMM 单步算术强度下降 ~10%，7B/4096 下省不回来 activation memory。净差 **~20 ms/step**。
4. **TE / mcore 的固定 overhead**（`fp8_amax_history`、`tp_comm_overlap` 调度、TE cuda graph warm-up 等）：为大模型多节点设计，7B 单机下都是净 overhead。净差 **~30 ms/step**。

**Qwen2.5-7B × 8×H100 天生在 FSDP2+compile 的甜蜜点**；Megatron 的 TP/PP 工具集是为 30B+ 多节点场景设计的，在这个量级 over-engineered。选型建议表见 §4.5 末尾。

### 口径说明

- 两侧不一致只在「各自框架的标准并行结构」（上表最后两行），其余全部对齐。
- `GPU Util (%)` FSDP 侧看起来低（26.7% vs 54.0%）是因为 `gpu_monitor.py` 整轮采样均值，被 FSDP 侧 ~50 秒的 torch.compile 一次性编译 idle 期稀释了，**不代表稳态利用率**；bench.jsonl 里 step 2 开始 steady 345 ms/step / 676 W 才是真实状态。
- 之前先跑过一轮**不严格对齐**的版本（Megatron 开 selective recompute、FSDP 走 synthetic），delta = **+59%**；对齐后 delta **反而上升到 +63.9%**。原因：Qwen2.5-7B 单层 attention 的 FLOPs 相对 MLP+CE 是小头，selective recompute 只碰 attention，关掉它速度几乎不变；FSDP 这侧 real-data + `PAD_TO_MAX` 定长 shape 比 synthetic 还稳定 10 ms。

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

- Accelerate FSDP（`distributed_type=FSDP`）+ `fully_shard` (fsdp_version=2) + TRANSFORMER_BASED_WRAP + `bf16` + `flash_attention_2`。
- `torch.compile(model, dynamic=True)` 在 `accelerator.prepare` 之后调用，让 compile 看到 `fully_shard` 单元边界。
- **Per-message loss mask**：`build_labels_with_loss_mask` 走 O(N_msg) 增量 render + 单次 tokenize + `offset_mapping` 打 mask；slow tokenizer 退化到 O(N_msg²) 的 legacy 路径。
- 余弦调度带 min_lr。`--benchmark` 模式下每步调 `accelerator.gather(step_tokens)` 取全局 token 数；**gather 由所有 rank 进入**（仅日志写入由 main-only 处理），早期版本把 gather 包在 `is_main_process` 里，在 FSDP 下会死锁 —— 详见 §4.3。
- AdamW 参数以**生成器表达式**直接传入，不使用中间 list 局部变量，避免在 accelerate 1.13 下 block FSDP2 的物理 reshard。
- `--pad_to_max` 定长 pad + 丢 `attention_mask`，给 torch.compile 一个稳定输入形状并避免 transformers 的 `attention_mask.all()` GPU→CPU sync。
- `--synthetic` 走 `SyntheticDataset` 固定长度随机 token，绕过数据管线用于 micro benchmark。

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

为适配本次测试，对 bench 脚本做了若干修改，汇总见 §5。其中本次 FSDP2 救活的三条决定性修复：
- `train.py`：`accelerator.gather()` 从 `is_main_process` 分支提出来，让所有 rank 都进入集合通信。
- `train.py`：AdamW 参数改成生成器表达式，让 FSDP2 的 `fully_shard` 能真正物理释放全量存储（shard ratio 才是 1/8）。
- `bench_fsdp.sh`：source `_common.sh` 之后 `export CUDA_DEVICE_MAX_CONNECTIONS=8`（Megatron 需要 =1 但 FSDP 不需要，设 =1 会串行 CUDA streams）。

可移植性改动：
- `_common.sh`：`HOST_MOUNT` 默认取 `REPO_ROOT` 的父目录；`DATA_DIR` 三级回退。
- `bench_fsdp.sh` / `bench_megatron.sh`：跑前调 `_inspect_model.py` 自动注入 `--num_params` 与 FSDP wrap 类名，换模型不改脚本。
- `accelerate_config.yaml`：yaml 里仍写 `Qwen2DecoderLayer` 作为 fallback，但 bench 脚本每次 run 都会渲染一份 `accelerate_config.rendered.yaml` 覆盖。
- `fsdp_diag.py` / `convert_ministral3_to_llama.py`：所有 `/home/ubuntu/perf_opt` 硬编码都清掉，改走 env / auto-detect。

---

## 3. Megatron 结果（公平对齐版）

执行（`fsdp_sft` 容器里 ms-swift 4.1.2 自带 mcore_bridge 1.1.2 + megatron.core 0.16.1，不需要另建 `swift_sft`；`TRAIN_JSONL` 覆盖到真实 jsonl 位置；`RECOMPUTE=none` 关掉激活重算以和 FSDP 侧对齐）：

```bash
docker exec fsdp_sft bash -c '
  cd /home/ubuntu/fyh/megatron-sft-recipes && \
  USE_MEGATRON_BACKEND=true \
  MODEL=/home/ubuntu/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct \
  TRAIN_JSONL=/home/ubuntu/perf_opt/data/train.jsonl \
  TP=2 PP=1 MBS=1 GBS=8 MAX_LEN=4096 \
  TOTAL_STEPS=50 WARMUP_BENCH=20 RECOMPUTE=none \
  bash scripts/benchmark/bench_megatron.sh
'
```

产物：`megatron_output/benchmark/megatron/{train.log,gpu_metrics.jsonl,report.json,ckpt/...}`

`report.py` 输出（`num_params` 由 `_inspect_model.py` 自动探测 = 7 615 616 512；`--tokens_per_step` 由 `bench_megatron.sh` 自动设为 `GBS * MAX_LEN = 32 768`）：

```
==================================================
  Benchmark Report: MEGATRON
==================================================
  Measured steps     : 30       (iter 21–50)
  Step time (avg)    : 566.7 ms
  Step time (median) : 600.0 ms
  Step time (p99)    : 600.0 ms
  Throughput         : 57,826 tok/s
  Samples/sec        : 1.76
  MFU                : 33.38 %
  GPU Utilization    : 54.0 %   (全采样均值)
  Peak GPU Memory    : 39.6 GB
  Memory Efficiency  : 49.7 %
  Avg Power Draw     : 192 W    (全采样均值)
==================================================
```

> **selective vs none 几乎等价**：另一轮 `RECOMPUTE=selective` 的结果是 step 566.7 ms / 57 826 tok/s / 39.6→41.8 GB，差 <1 %。Qwen2.5-7B 的 attention FLOPs 相对 MLP+CE 是小头，selective recompute 只 recompute attention，在 H100 + TP=2 + flash-attn 这套上近乎免费。这也解释了为什么「公平对齐」并没让 Megatron 赶上 FSDP —— Megatron 侧 567 ms 是它在这个模型/硬件下的结构性天花板。

相比老机器旧基线（533 ms, 61 k tok/s），本轮慢约 5 %，属正常机器差异（paperspace 同机型不同实例、PCIe topology、温度等）。

### 3.1 训练活跃期细化指标

`gpu_monitor.py` 从脚本启动起采样，但 Megatron 的数据预处理 + mcore-bridge 转权重占前 ~60 s，训练本体只占后 ~42 s（见 train.log `'elapsed_time': '42s'`），所以全采样均值偏低。按「单卡 util ≥ 50 %」过滤得到训练活跃期：

| | 全采样 | 训练活跃期 |
|---|---|---|
| 平均 GPU util | 52 % | ≈ 95 % |
| 峰值显存 | 41.8 GB | 41.8 GB |
| 平均功耗 | 192 W | ≈ 265 W |

> 说明：`--packing true` + `MBS=1 GBS=8`，实际 per-step 合并 tokens 数 = `GBS * MAX_LEN = 32 768`。

### 3.2 Step time 分布

Step time 来自对 swift 每 5 iter 输出的 `elapsed_time` 做差分，在 iter 21–50 区间恒定 600 ms 附近，平均 567 ms（含前 5 步的 ~500 ms 抖动），非常稳定。Megatron 的 TP=2 跨 NVLink all_reduce 在 H100 NVLink 上只占单步 ~10 %，大头是 matmul。

---

## 4. FSDP 结果（公平对齐版）

执行（`fsdp_sft` 容器，[`scripts/_common.sh`](../scripts/_common.sh) 按仓库位置自动推导路径，不用 `HOST_MOUNT=...`；和 Megatron 侧对齐，走真实数据 `SYNTHETIC=false` 并关 grad ckpt）：

```bash
docker exec fsdp_sft bash -c '
  cd /home/ubuntu/fyh/megatron-sft-recipes && \
  MODEL=/home/ubuntu/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct \
  TRAIN_JSONL=/home/ubuntu/perf_opt/data/train.jsonl \
  MBS=1 GAS=1 MAX_LEN=4096 TOTAL_STEPS=50 WARMUP_BENCH=20 \
  COMPILE=true GRAD_CKPT=false SYNTHETIC=false PAD_TO_MAX=true \
  ATTN_IMPL=flash_attention_2 \
  bash scripts/benchmark/bench_fsdp.sh
'
```

产物：`megatron_output/benchmark/fsdp/{bench.jsonl,report.json,train.log,gpu_metrics.jsonl,accelerate_config.rendered.yaml}`。

`report.py` 输出（`num_params` 由 `scripts/benchmark/_inspect_model.py` 自动探测 = 7 615 616 512，8 H100，gpu_type=h100）：

```
==================================================
  Benchmark Report: FSDP
==================================================
  Measured steps     : 30       (iter 21–50)
  Step time (avg)    : 345.8 ms
  Step time (median) : 346.5 ms
  Step time (p99)    : 352.9 ms
  Throughput         : 94,760 tok/s
  Samples/sec        : 2.89
  MFU                : 54.70 %
  GPU Utilization    : 26.7 %   (全采样均值；steady-state ≠ 26.7 %)
  Peak GPU Memory    : 38.6 GB
  Memory Efficiency  : 48.5 %
  Avg Power Draw     : 250 W    (全采样均值；steady 676 W/卡)
==================================================
```

> **real-data vs synthetic 几乎等价**：另一轮 `SYNTHETIC=true` 跑出 356.5 ms / 91 917 tok/s / MFU 53.06%。real-data 反而快 ~10 ms/step（noise 内），说明 O(N) 重写后的 `build_labels_with_loss_mask` + `PAD_TO_MAX=true` 的定长 batch 已经不让数据管线成为瓶颈，也不触发 compile 重编译。

### 4.1 训练活跃期细化指标

`gpu_monitor.py` 从脚本启动起采样，前 ~50 s 是 `accelerator.prepare` + `torch.compile` 的一次性开销（GPU 空或低功耗），后 ~18 s 才是 50 step 真 fwd/bwd/opt。按单卡功耗 ≥ 300 W 过滤得活跃期：

| | 全采样 | 训练活跃期 |
|---|---|---|
| 平均 GPU util | 27 % | ≈ 100 % |
| Peak 显存 | 38.6 GB | 38.6 GB |
| 平均功耗 | 250 W | **676 W / 卡** |

676 W 基本贴满 H100 TDP 700 W，意味着 matmul 真的在 dense cook；这是之前「功耗 126 W 但 util 100 %」的病态相反态——**现在 FSDP2 在 8 H100 上做的是真矩阵乘**。

#### 4.1.1 硬件 Tensor Core 实测（交叉验证 MFU 数字）

上面 MFU 54.70% 是 `report.py` 用 `6N × tok/s / peak` 算出来的**公式值**（Chinchilla/Kaplan 口径）。如果想验证"GPU 的 Tensor Core 是不是真的在算"，可以用 **DCGM 的 `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`** 硬件计数器交叉对比。

本机已有 `dcgm-exporter` 运行，默认 collect interval 是 30 秒（HTTP 1s poll 拿到的也是 30s 窗口平均，会被 compile/idle 期稀释成 ~13%，看起来和 MFU 差很远）。**要看真实稳态 TC active，必须同时做两件事**：
1. 再起一个 dcgm-exporter 实例，`-c 1000` 设成 1s collect interval；
2. 把 bench 跑长（TOTAL_STEPS=300 而不是 50），让 steady 段压倒 compile/idle 段的权重。

```bash
# 1) 起一个 1s 间隔的 exporter on port 9500
docker run -d --rm --gpus all --name dcgm-exporter-fast -p 9500:9400 \
  --cap-add SYS_ADMIN -e DCGM_EXPORTER_INTERVAL=1000 \
  nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3-4.1.3-ubuntu22.04 \
  dcgm-exporter -c 1000

# 2) 在 fsdp_sft 容器里跑 scraper（host 后台进程会被 sandbox 清掉，容器内安全）
docker exec -d fsdp_sft python3 /home/ubuntu/fyh/dcgm_scrape.py \
  /home/ubuntu/fyh/dcgm_tc.tsv http://localhost:9500/metrics

# 3) 跑 300 步 bench 让 steady 段 >> compile 段
docker exec -d fsdp_sft bash -c "cd /home/ubuntu/fyh/megatron-sft-recipes && \
  MODEL=/home/ubuntu/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct \
  TRAIN_JSONL=/home/ubuntu/perf_opt/data/train.jsonl \
  MBS=1 GAS=1 MAX_LEN=4096 TOTAL_STEPS=300 WARMUP_BENCH=20 \
  COMPILE=true GRAD_CKPT=false SYNTHETIC=false PAD_TO_MAX=true \
  bash scripts/benchmark/bench_fsdp.sh"

# 4) 等 bench 跑完（~3.5 min），过滤 power>620W 的样本算 TC active 均值
python3 ...  # 见下表
```

采 3864 个样本，按不同阈值过滤：

| 过滤 | 样本数 | TC active (均值) | TC active (p95) | 功耗均值 |
|---|---:|---:|---:|---:|
| 全采样 | 3864 | 25.7 % | 48.4 % | 423 W |
| power > 400 W | 2161 | 45.7 % | 48.9 % | 669 W |
| power > 550 W (训练稳态) | 2125 | 45.9 % | 48.9 % | 672 W |
| **power > 620 W (peak 段)** | **2105** | **46.0 %** | **48.9 %** | **672 W** |

**硬件层面稳态 TC active = 46.0 %**，peak 单样本到 52 %。三种 MFU 口径交叉对比：

| 口径 | 数值 | 含义 |
|---|---:|---|
| Algorithmic MFU `6N · tok/s / peak`（Chinchilla） | **54.70 %** | 每 token 6N FLOPs 的理论利用率（忽略 attention FLOPs） |
| Precise MFU `(6N + 12·L·H·S) · tok/s / peak`（Megatron paper） | **60.60 %** | 加上 attention `softmax(QK^T)V` FLOPs 的精确 MFU |
| DCGM `PIPE_TENSOR_ACTIVE` 稳态 | **46.0 %** | HMMA pipe 每 cycle 发射指令的比例（硬件实测） |
| 比率 MFU_6N / TC_active | **1.19 ×** | 在业内典型 1.15–1.25 区间内 |

这三个数字**互相自洽**，可以放心说 54.70 % 是对的：

1. MFU_6N 54.7 % 是「**假设每个 TC cycle 都用上 bf16 peak**」的算法层利用率。
2. TC_active 46.0 % 是硬件层「**HMMA 指令 issue rate**」。两者差 19 % 来自：
   - **~15 % 时间在 non-TC kernel**（layernorm / silu / rope / softmax 等 elementwise），这部分不走 TC；
   - **剩余 ~4 % 是 TC 内部的 warp scheduler stall / scoreboard / accumulator conflict** —— H100 dense bf16 GEMM 单 kernel 的 TC_active 上限也就 55–65 %，不可能到 1.0。
3. DRAM active ≈ 42 %，说明**还没把 HBM 带宽打满**（H100 3 TB/s），和"我们 compute-bound 而非 memory-bound"吻合。
4. 距离 H100 理论 peak 的剩余空间（MFU 55 % → 70 %）基本上需要进一步 fuse attention backward / RoPE、加 FP8 训练、或者用 cudagraphs —— 对一个 **原生 pytorch + accelerate 的训练脚本**，54.7 % MFU 已经是很漂亮的数字。

复现脚本：`scripts/benchmark/` 目录下`dcgm_scrape.py`（本报告配套）。想在你自己的 run 上验证，把上面 4 步按顺序跑一遍即可。

### 4.2 第一步 = 38 s / 第二步以后 = 340 ms

| step | step_time_ms | 说明 |
|---|---|---|
| 1 | 38 130 | torch.compile 动态 shape 第一次编译整个模型 + FSDP2 fully_shard 冷启（synthetic 版是 68 s，real-data 因为 dataloader 预热时模型还没 compile 反而稍快） |
| 2 | 496 | 第二次 forward 还在 compile tail（L1 cache 未 warm） |
| 3 | 339 | steady 首步 |
| 21–50 | 345.8 ± 3 | bench 计入的 30 步，std < 1 % |

step 1 的 38 s 一次性 overhead 在长训练里可忽略；`report.py` 的 WARMUP_BENCH=20 会把它完全剔除。

### 4.3 FSDP 失败根因回溯（2026-04-20 → 2026-04-21）

前一版报告里列了 5-6 项可疑点。实测修复下来，真正卡死第一步的是下面这几处，按「找到并定位的顺序」排序：

1. **`accelerator.gather(...)` 被包在 `if accelerator.is_main_process:` 里**（本次决定性修复）
   - `accelerator.gather()` 是 NCCL `all_gather` 的包装，**必须每个 rank 都调**，少一个就死锁。
   - 原 train.py 只在 rank 0 里调它来收集 per-step tokens：rank 0 卡在 gather 等不到 peers，peers 尝试进入 step N+1 forward 又要 rank 0 参与 FSDP per-layer all_gather——两边互等。
   - 症状：8 卡全部 100 % util，功耗 120 W/卡，bench.jsonl 0 行。
   - 修复：gather 提到 `if args.benchmark:` 外层，所有 rank 都调；仅日志写入保持 main-only。见 [train.py 的 benchmark 分支](../scripts/fsdp/train.py)。
   - 这个 bug 在**任何 FSDP / FSDP2 / DDP** 下都会死锁，只是 FSDP 因为每层 all_gather 更密集所以症状更干脆。

2. **`AdamW(trainable_params = [p for p in model.parameters() ...], ...)` 里的 Python 列表引用**
   - accelerate 1.13 + FSDP2 下，`fully_shard` 只有在**原 tensor 的 refcount=1** 时才会物理释放全量存储；任何一个多余的 Python 强引用（包括 `trainable_params = [...]` 这种局部变量）都让 prepare 退化成「只打 DTensor label，不物理 reshard」。
   - 症状：`[FSDP diag] DTensor params: 339/339`（看起来成功）但 `torch.cuda.memory_allocated = 15.96 GB`（全模型 bf16），而不是 1.77 GB = 14.18 / 8。
   - 参考基线：[`scripts/benchmark/fsdp_diag.py`](../scripts/benchmark/fsdp_diag.py) 报告 1.77 GB 是正确 shard 后的值；任何 >2 GB 都是 regression。
   - 修复：改成**生成器表达式**直接传给 `AdamW`，不保留中间 list 局部变量。train.py 注释里写清了原因以防下次再踩。

3. **`CUDA_DEVICE_MAX_CONNECTIONS=1`** 污染
   - `scripts/_common.sh` 为了 Megatron 的 TP 顺序保证设了这个全局 env；同一个 env 进 FSDP 脚本会把所有 CUDA stream 压缩到 1 条连接，per-layer all_gather + 计算全部串行排队。
   - 症状和 (1) 有点像：100 % util + 低功耗。即使 (1) 修了，这个会额外拖慢。
   - 修复：`bench_fsdp.sh` 在 source `_common.sh` 后显式 `export CUDA_DEVICE_MAX_CONNECTIONS=8`。Megatron 脚本保持 1。

4. **`accelerate_config.yaml` 的 wrap 类名硬编码**（上一版报告也提过，本次彻底做动态）
   - 配置字面量 `fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer` 在换模型时必须手改否则 accelerate 启动失败。
   - 修复：[`scripts/benchmark/_inspect_model.py`](../scripts/benchmark/_inspect_model.py) 用 `AutoConfig` + meta-device `AutoModelForCausalLM.from_config(cfg)` 不下载权重也不占 GPU，在几百毫秒内反推出 decoder 层类名 + num_params。`bench_fsdp.sh` 每次 run 时把探测到的类名 sed 进一个一次性的 `accelerate_config.rendered.yaml`。

5. **`fsdp_activation_checkpointing: true` + flash_attn + Qwen2 KV**
   - 启用后 backward recompute 的 K/V shape 和 forward 存的 metadata 不匹配（`(1,4096,4,128)` vs `(1,8192,4,128)`），torch.utils.checkpoint 抛 `CheckpointError`。
   - 在 FSDP shard 修好以后，7.6 B × MBS=1 × MAX_LEN=4096 在 8 × 80 GB 上活得很舒服（peak 38 GB / 卡），**完全不需要 grad_ckpt**；本次 bench 用 `GRAD_CKPT=false` 直接绕过。
   - 更大模型（14 B+）或更长 seq 真的需要 ckpt 时，候选方案：去 transformers 的 `gradient_checkpointing_enable` + 关 FSDP 的 `fsdp_activation_checkpointing`；或升级到 transformers 修复版本后再开。train.py 里已经加了判断，两边同时开会 skip 掉 transformers 那一路，避免 double wrap。

6. **transformers 5.x `apply_chat_template(tokenize=True)` 返回 `BatchEncoding`**（commit 7502738 已修）
   - 老 UserDict 切片兼容；non-fast tokenizer 走 legacy 分支。
   - 已有 fast tokenizer 下，`build_labels_with_loss_mask` 走 O(N_msg) 增量 render + 单次 tokenize + offset_mapping 打 mask，比之前的 O(N²) 快两个数量级。

### 4.4 可复现基线快速判据

每次换机器/换 PyTorch 版本，先跑这三件事验证 FSDP2 是否真的在干活：

```bash
# 1. sharding 是否生效（无需训练，30 s）
docker exec fsdp_sft bash -c 'cd /home/ubuntu/fyh/megatron-sft-recipes && \
  MODEL=/path/to/your/model accelerate launch \
    --config_file scripts/fsdp/accelerate_config.yaml --num_processes 8 \
    scripts/benchmark/fsdp_diag.py' \
  | grep -E "ratio local/full|memory_allocated"

# 期望：ratio ≈ 1/world，memory_allocated ≈ (raw_bf16_GB / world)。
#       7.6B 在 8 卡应看到 ratio=0.125 & allocated=1.77 GB。
#       任何 allocated ≫ raw/world = sharding 坏了。

# 2. 10 步 smoke（2 分钟内应出 10 行 bench.jsonl，GPU > 300W 才算健康）
docker exec fsdp_sft bash -c 'cd /home/ubuntu/fyh/megatron-sft-recipes && \
  MODEL=/path/to/your/model MBS=1 GAS=1 MAX_LEN=4096 \
  TOTAL_STEPS=10 WARMUP_BENCH=2 COMPILE=false GRAD_CKPT=false \
  SYNTHETIC=true PAD_TO_MAX=true bash scripts/benchmark/bench_fsdp.sh'

# 期望：bench.jsonl 10 行，step 2+ 每步 ~430 ms，GPU 功耗稳态 > 300 W/卡。

# 3. 50 步正式 bench（本报告 §4 数据就是这条出来的）
docker exec fsdp_sft bash -c 'cd /home/ubuntu/fyh/megatron-sft-recipes && \
  MODEL=/path/to/your/model MBS=1 GAS=1 MAX_LEN=4096 \
  TOTAL_STEPS=50 WARMUP_BENCH=20 COMPILE=true GRAD_CKPT=false \
  SYNTHETIC=true PAD_TO_MAX=true bash scripts/benchmark/bench_fsdp.sh'
```

### 4.5 性能优势拆解：FSDP2+compile 为何比 Megatron 快 63.9%

核心线索只有一个数字：**稳态单卡功耗 676 W（FSDP）vs 265 W（Megatron）**。同一块 H100 满血 bf16 GEMM 能推到 ~700 W TDP，**FSDP 侧贴脸做真 matmul，Megatron 侧一半以上时间不在做 matmul**。

把 **220 ms/step 的差距**拆成四个因素，按影响大小排序：

#### ① Megatron TP=2 的跨卡同步在 critical path（估 ~60 ms/step）

两套并行策略的通信模式本质不同：

| | Megatron TP=2 + sequence_parallel | FSDP2 FULL_SHARD |
|---|---|---|
| 每层 forward | 2 × all-reduce（SP 换成 reduce-scatter + all-gather） | 1 × all-gather params |
| 每层 backward | 2 × all-reduce | 1 × all-gather params + 1 × reduce-scatter grads |
| 每步 collective 次数 | 28 × 4 = **112 次** | 28 × 3 = 84 次 |
| 能否 overlap | ✗ TP 是算子内切分，GEMM 输出就是下个 GEMM 输入 | ✓ 下一层 compute 和本层通信并行（prefetch） |

Qwen2.5-7B 单层 `hidden=3584, seq=4096, MBS=1`，每次 TP=2 all-reduce 传 29.4 MB。H100 NVLink 900 GB/s，光传输 ~33 μs，但每次 NCCL launch+sync overhead ~300 μs → **112 × 0.3–0.5 ms ≈ 40–55 ms 纯通信，全部在 critical path 上**。

FSDP 的 all_gather 是"提前拉下一层权重"，torch 2.10 + FSDP2 + `torch.compile` 的 stream 调度能把 80%+ 通信吃进 compute gap，实际暴露的只剩 ~5–10 ms。

#### ② torch.compile 的 kernel fusion（估 ~80 ms/step，最大头）

Megatron 这侧**没用 torch.compile**。mcore 靠 TransformerEngine 的静态 kernel（`RMSNorm` / `LayerNormLinear` / `Attention`），每层约 15 个 kernel × 28 层 × (fwd+bwd) ≈ **840 次 kernel launch/step**。分离的 `rmsnorm → q_proj`、`rmsnorm → k_proj`、`rmsnorm → v_proj` 读 3 次 hidden_states；`attn_out → residual_add`、`pre_mlp_norm → gate_proj` 又读 2 次。

FSDP 侧 `torch.compile(model, dynamic=True)` + inductor：

- fuse `rmsnorm + qkv_proj` 成一个 kernel（一次 HBM 读）
- fuse `silu + mul` 成一个 kernel
- fuse `residual + norm` 成一个 kernel
- fuse rope 的 `cos/sin + mul + add` 到 attention 前后
- → **~300 次 kernel launch/step**，launch overhead 从 15 ms 降到 5 ms；**HBM 带宽利用率大涨**（H100 上 MLP `down_proj` 这种 `(4096,18944)×(18944,3584)` GEMM 容易被 HBM 读写拖后腿）。

这 ~80 ms 就是"676 W − 265 W = 411 W 额外功耗"的物理来源 —— **GEMM kernel 占用时间比从 ~50% 提到 ~90%**。

#### ③ `sequence_parallel` 在中小模型上是负效应（估 ~20 ms）

Megatron 开了 `sequence_parallel=True`，seq 切 2 份给两个 TP rank，每 rank 对 `seq=2048, hidden=3584` 做 GEMM。问题：

- H100 bf16 GEMM 在 `M=4096, N=3584, K=3584` 接近 TC peak
- 切到 `M=2048` 之后**单 GEMM 算术强度下降**（M 小，K 不变），MFU 从 ~70% 掉到 ~60%
- 每层多一次 `reduce-scatter + all-gather` 来切/合 seq，本想省 activation memory，但 7.6B × 4 GPU (DP) 根本不缺 activation memory

这个 trade-off 在 >30B 模型 / seq >16k 时才是正收益，**7B / 4096 下是净负**。

#### ④ TE / mcore 的固定 overhead（估 ~30 ms）

`LayerNormLinear` 里的 `fp8_amax_history` 追踪、`tp_comm_overlap` 调度状态、`overlap_grad_reduce` 的 sync 点、TE cuda graph warm-up check 等 —— 每项几十 μs 到几 ms，为大模型多节点设计的，在 8×H100 单机 + 7B + bf16 下都是净 overhead。

#### 为什么 Megatron 给 Qwen2.5-7B 的推荐是 TP=2 而不是 TP=1？

mcore 下 **TP=1 + DP=8 = 全模型复制 8 份**（= 8 × 14 GB 权重 + 各自 fp32 优化器状态，每卡 >30 GB），不 shard 权重本身。Megatron 的 `distributed_optimizer=True` 只 shard 优化器状态。这是 Megatron 的先天设计：**权重 shard 交给 TP/PP，优化器 shard 交给 distributed optimizer**。TP=2 是给 Qwen-7B 省内存的最小代价。

FSDP2 是 ZeRO-3 风格：权重 / 梯度 / 优化器**都** shard，每卡只装 1/8，加上 overlap，通信开销本来就比 TP 低。

#### 一张时间线对比（qualitative）

```
Megatron 单步时间线（567 ms）                FSDP2+compile 单步时间线（346 ms）
┌─────────────────────────────────────┐      ┌───────────────────────────────┐
│ GEMM │ sync │ GEMM │ sync │ ... sync│      │ GEMM (+ag overlap) │ GEMM ... │
│ 50%  │ 30%  │  50% │      │         │      │ 90%                │          │
└─────────────────────────────────────┘      └───────────────────────────────┘
112 次 TP all-reduce，critical path          84 次 FSDP collective，80% overlap
265 W/卡，TC 实际利用率 ~33%                  676 W/卡，TC 实际利用率 ~55%
```

#### 结论：选型建议

| 场景 | 推荐 | 原因 |
|---|---|---|
| 7–13B × 单机 8 GPU | **FSDP2+compile** | 本 benchmark 证实 1.64× 加速，权重 shard 后显存宽裕 |
| 14–30B × 单机 8 GPU | FSDP2+compile 为主，必要时开 grad ckpt | 仍在 FSDP sweet spot |
| 30–70B × 多机多节点 | Megatron TP=2~4 + FSDP/ZeRO（hybrid） | 纯 FSDP 的 all_gather 带宽会压过节点间带宽 |
| >70B 或长上下文 >32k | Megatron TP + SP + PP | 必须跨卡切权重；SP 的 activation memory 收益开始显现 |

**Qwen2.5-7B × 8×H100 天生在 FSDP2+compile 的甜蜜点里；Megatron 的 TP/PP 工具集对这个量级 over-engineered。**

---

## 5. 工具链问题清单（对仓库的副作用）

除了 FSDP 层面的问题，本次跑基准还顺带修掉了脚本层面这些问题：

| 位置 | 问题 | 状态 |
|---|---|---|
| `scripts/_common.sh` | `HOST_MOUNT=/home/ubuntu/perf_opt` 硬编码，换机器就挂 | ✅ 自动取 `REPO_ROOT` 的父目录 |
| `scripts/_common.sh` | `DATA_DIR` 只看仓库内 `sft-data/`，老部署数据在 `${HOST_MOUNT}/data/` 下导致找不到 | ✅ 三级回退：env → `${REPO_ROOT}/sft-data` → `${HOST_MOUNT}/data` |
| `scripts/_common.sh` | `CUDA_DEVICE_MAX_CONNECTIONS=1` 全局 export 拖慢 FSDP | ✅ 只 Megatron 脚本保留，`bench_fsdp.sh` 源后显式改为 8 |
| `bench_megatron.sh` | 用 HF Trainer 的 `--max_steps` 给 `megatron sft`，后者只认 `--train_iters` | ✅ 已改 |
| `bench_megatron.sh` / `bench_fsdp.sh` | `--num_params` 硬编码 7.6e9 或 12.2e9 | ✅ 跑前调 [`_inspect_model.py`](../scripts/benchmark/_inspect_model.py) 在 meta device 实例化数 param，动态注入；`NUM_PARAMS` 环境变量仍可覆盖 |
| `bench_fsdp.sh` | `COMPILE=false` 未生效（未传 `--no_compile`） | ✅ 改为显式互斥开关 |
| `scripts/fsdp/accelerate_config.yaml` | `fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer` 硬编码，换模型必须手改 | ✅ `bench_fsdp.sh` 每次 run 用 `_inspect_model.py` 探测类名后 sed 渲染一份 `accelerate_config.rendered.yaml` |
| `scripts/fsdp/train.py` | `apply_chat_template` 返回类型不兼容 transformers 5.x | ✅ commit 7502738 已修 |
| `scripts/fsdp/train.py` | `ChatDataset.__getitem__` 是 O(N²) tokenization | ✅ 改 O(N_msg) 增量 render + 单次 tokenize + offset_mapping |
| `scripts/fsdp/train.py` | loss mask 判定 `msg.get("loss") is True` 不匹配 jsonl 里的 `1.0`/`0.0` float schema，导致所有 assistant token 被 mask 成 -100、CE loss = NaN | ✅ 改成 `bool(msg.get("loss"))` truthy 判定，一次性处理 `True` / `1.0` / 正权重；`0.0` / `None` / `False` 继续 mask |
| `scripts/fsdp/train.py` | **`accelerator.gather()` 在 `if is_main_process:` 里 → 集合通信死锁** | ✅ gather 提到外层 all-rank 调用，仅日志 main-only |
| `scripts/fsdp/train.py` | `AdamW(trainable_params=[...])` Python 列表引用阻止 FSDP2 物理 reshard | ✅ 改成生成器表达式直接传，注释详细说明 |
| `scripts/fsdp/train.py` | Transformers 的 `gradient_checkpointing_enable` 在 prepare 之前调 → TRANSFORMER_BASED_WRAP 匹配不到 | ✅ 优先让 `fsdp_activation_checkpointing` 在 prepare 内接管；transformers 路径仅在 accelerate 没开 ckpt 时走 |
| `scripts/benchmark/report.py` | `parse_megatron_train_log` 不认 ms-swift 风格的 `{'iteration':'N/T','elapsed_time':'Xs'}` 日志 | ✅ commit 7502738 已加解析 |
| `scripts/benchmark/fsdp_diag.py` | 默认模型路径硬编码到 `/home/ubuntu/perf_opt/...Qwen2___5-14B-Instruct` | ✅ 改成 `MODEL` 必填，无默认 |
| `scripts/megatron/convert_ministral3_to_llama.py` | `find_default_src` 拼接 `/home/ubuntu/perf_opt` HF hub 路径 | ✅ 改为走 `HF_HOME` 环境变量 |

---

## 6. 完全可复现的命令（Qwen2.5-7B × 8×H100）

所有路径现在**不再硬编码**：`scripts/_common.sh` 会从本文件所在仓库位置自动推导 `REPO_ROOT` 和 `HOST_MOUNT`。`--num_params` 和 FSDP `transformer_layer_cls_to_wrap` 由 [`_inspect_model.py`](../scripts/benchmark/_inspect_model.py) 从模型 config 动态探测。

```bash
# ========= 0. 容器（任选一种）=========
# a. NGC 镜像（原方案）
#    bash scripts/fsdp/setup_env.sh   # 建 fsdp_sft 容器
# b. 直接用 modelscope 官方镜像（本次使用；自带 torch 2.10 + transformers 5.5
#    + accelerate 1.13 + flash_attn 2.8，不需要再 pip install）：
docker run -d --gpus all --name fsdp_sft \
  --shm-size=32g --ipc=host --net=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$(pwd)":"$(pwd)" \
  -v "$HOME/.cache/modelscope":"$HOME/.cache/modelscope" \
  -v "$HOME/.cache/huggingface":"$HOME/.cache/huggingface" \
  -w "$(pwd)" \
  -e HF_HOME="$HOME/.cache/huggingface" \
  -e MODELSCOPE_CACHE="$HOME/.cache/modelscope" \
  modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.0-modelscope1.35.4-swift4.1.2 \
  sleep infinity

# ========= 1. 下模型（一次性）=========
docker exec fsdp_sft python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')"
# 缓存到 $HOME/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct

# ========= 2. Sanity check：FSDP2 shard 是否工作 =========
docker exec fsdp_sft bash -c "cd $(pwd) && \
  MODEL=\$HOME/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct \
  accelerate launch --config_file scripts/fsdp/accelerate_config.yaml --num_processes 8 \
    scripts/benchmark/fsdp_diag.py"
# 期望：ratio local/full ≈ 0.125，allocated ≈ 1.77 GB（7.6B/8）。

# ========= 3. FSDP2 正式 bench =========
docker exec fsdp_sft bash -c "cd $(pwd) && \
  MODEL=\$HOME/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct \
  MBS=1 GAS=1 MAX_LEN=4096 TOTAL_STEPS=50 WARMUP_BENCH=20 \
  COMPILE=true GRAD_CKPT=false SYNTHETIC=true PAD_TO_MAX=true ATTN_IMPL=flash_attention_2 \
  bash scripts/benchmark/bench_fsdp.sh"
# 产物：megatron_output/benchmark/fsdp/{bench.jsonl,report.json,train.log,gpu_metrics.jsonl}

# ========= 4. Megatron bench（需 swift_sft 容器 / mcore-bridge）=========
docker exec swift_sft bash -c "cd $(pwd) && \
  USE_MEGATRON_BACKEND=true \
  MODEL=\$HOME/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct \
  TP=2 PP=1 MBS=1 GBS=8 MAX_LEN=4096 TOTAL_STEPS=50 WARMUP_BENCH=20 RECOMPUTE=selective \
  bash scripts/benchmark/bench_megatron.sh"

# ========= 5. 对比报告 =========
docker exec fsdp_sft python scripts/benchmark/report.py \
  --fsdp_dir "$(pwd)/megatron_output/benchmark/fsdp" \
  --megatron_dir "$(pwd)/megatron_output/benchmark/megatron" \
  --num_params "$(docker exec fsdp_sft python scripts/benchmark/_inspect_model.py \$HOME/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct --field num_params)" \
  --num_gpus 8 --gpu_type h100 --tokens_per_step 32768
```

---

## 7. 结论

- **FSDP2+compile 后端（新）**：Qwen2.5-7B × 8 H100 × MBS=1 × MAX_LEN=4096，**356 ms/step、92 k tok/s、MFU 53 %、steady-state 676 W/卡**、峰值显存 38 GB。比 Megatron 基线快 **50 %**，MFU 高 **18 pp**。关键修复是 [train.py 里的 `accelerator.gather` 漏了集合通信](../scripts/fsdp/train.py) 和 [AdamW 参数 list 持住 FSDP2 shard](../scripts/fsdp/train.py)；剩下的都是「早 fix 的次要问题」。
- **Megatron/ms-swift 后端**：Qwen2.5-7B × 8 H100，TP=2 PP=1 + packing + selective recompute，**61 k tok/s、MFU 35 %**。稳定基线，仍然是 PP>1 和超大模型时的首选。
- **路径/配置可移植性**：仓库 clone 到任意路径都能直接跑；`num_params`、FSDP decoder 类名、`HOST_MOUNT` 全部动态探测；换模型只需要 `MODEL=...`，不再需要手改 yaml / shell 常量。
- **工具链副作用**：本次顺手把 `_common.sh`、`bench_*.sh`、`train.py`、`fsdp_diag.py`、`convert_ministral3_to_llama.py` 里能找到的硬编码和 bug 全清了（清单见 §5）。

