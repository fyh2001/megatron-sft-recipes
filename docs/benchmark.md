# 性能基准测试

对比 Megatron 和 FSDP2+compile 两个后端在相同模型、相同数据上的训练性能。

## 测试模型

默认：`mistralai/Mistral-Nemo-Instruct-2407`（12.2B 参数）

可通过 `MODEL=...` 环境变量切换为其他模型（两边需要用相同模型才有可比性）。

## 测试协议

- 50 个 optimizer step：前 20 步 warmup（torch.compile 编译 + CUDA cache 预热），后 30 步计入指标
- 不保存 checkpoint（排除 I/O 噪声）
- 统一配置：MBS=1, GBS=8, MAX_LEN=4096
- GPU 监控每秒采样一次 nvidia-smi

## 指标说明

| 指标 | 含义 |
|------|------|
| Throughput (tok/s) | 每秒处理的 token 数（所有 GPU 合计） |
| Step time (ms) | 单个 optimizer step 耗时（avg / median / p99） |
| MFU (%) | Model FLOPs Utilization = 实际 FLOPS / GPU 理论峰值。公式：`6 * num_params * tok/s / (peak_tflops * num_gpus)` |
| GPU Util (%) | nvidia-smi 报告的 SM 利用率平均值 |
| Peak Memory (GB) | 所有 GPU 中最大显存占用 |
| Memory Efficiency (%) | Peak Memory / Total Memory |
| Avg Power (W) | 平均功耗 |

## 前置条件

1. 已执行数据转换：`sft-data/train.jsonl` 存在
2. 对应容器已就绪（通过各自的 `setup_env.sh` 搭建）

## 执行步骤

### Step 1: FSDP2+compile Benchmark

```bash
# 进入 FSDP 容器
docker exec -it fsdp_sft bash

# 跑 benchmark（默认 Mistral-Nemo-12B, 8 卡, 50 步）
bash scripts/benchmark/bench_fsdp.sh

# 自定义模型或参数
MODEL=Qwen/Qwen2.5-7B-Instruct MBS=2 GAS=4 COMPILE=true \
    bash scripts/benchmark/bench_fsdp.sh
```

结果输出到 `${OUTPUT_ROOT}/benchmark/fsdp/`：
- `bench.jsonl` — 逐步计时指标
- `gpu_metrics.jsonl` — GPU 采样数据
- `report.json` — 汇总报告
- `train.log` — 完整训练日志

### Step 2: Megatron / ms-swift Benchmark

```bash
# 进入 Megatron 容器
docker exec -it swift_sft bash

# 跑 benchmark（默认走 swift sft HF 后端，保证模型一致）
bash scripts/benchmark/bench_megatron.sh

# 如果模型在 mcore-bridge 支持列表里，可以走 Megatron 后端
USE_MEGATRON_BACKEND=true TP=2 PP=1 \
    bash scripts/benchmark/bench_megatron.sh

# 换模型
MODEL=Qwen/Qwen2.5-14B-Instruct USE_MEGATRON_BACKEND=true TP=2 PP=1 \
    bash scripts/benchmark/bench_megatron.sh
```

结果输出到 `${OUTPUT_ROOT}/benchmark/megatron/`。

### Step 3: 生成对比报告

```bash
# 在任一容器内或宿主机上执行（只需要 Python + json）
python scripts/benchmark/report.py compare \
    --fsdp_dir /home/ubuntu/perf_opt/megatron_output/benchmark/fsdp \
    --megatron_dir /home/ubuntu/perf_opt/megatron_output/benchmark/megatron \
    --num_params 12.2e9 \
    --num_gpus 8 \
    --gpu_type h100
```

也可以只看单个框架的报告：

```bash
python scripts/benchmark/report.py \
    --framework fsdp \
    --bench_log /home/ubuntu/perf_opt/megatron_output/benchmark/fsdp/bench.jsonl \
    --gpu_log /home/ubuntu/perf_opt/megatron_output/benchmark/fsdp/gpu_metrics.jsonl \
    --num_params 12.2e9 \
    --num_gpus 8 \
    --gpu_type h100
```

### 输出示例

```
==============================================================
  Performance Comparison: Megatron vs FSDP2+compile
==============================================================
  Metric                  Megatron  FSDP2+compile     Delta
  ----------------------------------------------------------
  Step time (ms)          2640.3 ms     2780.1 ms    +5.3%
  Throughput (tok/s)       12,450        11,800      -5.2%
  MFU (%)                  42.30%        40.10%     -5.2%
  GPU Util (%)              96.2%         94.8%     -1.5%
  Peak Memory (GB)         72.1 GB       68.4 GB    -5.1%
  Memory Efficiency         90.1%         85.5%     -5.1%
  Avg Power (W)             680 W         652 W     -4.1%
  ----------------------------------------------------------
==============================================================
```

## 环境变量速查

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL` | `mistralai/Mistral-Nemo-Instruct-2407` | 测试模型 |
| `MBS` | `1` | micro batch size per GPU |
| `GAS` | `1` (FSDP) | gradient accumulation steps |
| `GBS` | `8` (Megatron) | global batch size |
| `MAX_LEN` | `4096` | 最大序列长度 |
| `TOTAL_STEPS` | `50` | 总步数 |
| `WARMUP_BENCH` | `20` | warmup 步数（不计入指标） |
| `COMPILE` | `true` | 是否开 torch.compile（仅 FSDP） |
| `GRAD_CKPT` | `true` | 是否开 gradient checkpointing |
| `NPROC_PER_NODE` | `8` | GPU 数量 |
| `USE_MEGATRON_BACKEND` | `false` | Megatron bench 是否走 mcore 后端 |

## GPU 类型

`report.py` 的 `--gpu_type` 参数决定 MFU 分母（理论峰值 BF16 TFLOPS）：

| GPU | Peak BF16 TFLOPS |
|-----|-----------------|
| h100 / h800 | 989.5 |
| a100 / a800 | 312.0 |
| l40s | 362.0 |

## 排障

**Q: bench_fsdp.sh 前几步特别慢（>30s/步）？**
A: 正常。torch.compile 在前 2-3 步做编译，设 `WARMUP_BENCH=20` 就是为了跳过这些步。看 step 21+ 的数据才是真实性能。

**Q: Megatron 后端不支持 Mistral-Nemo？**
A: mcore-bridge 没有 Mistral-Nemo converter。设 `USE_MEGATRON_BACKEND=false`（默认）走 `swift sft` HF 后端对比。或者先跑 `convert_ministral3_to_llama.py` 转成 Llama 再用 Megatron 后端。

**Q: 想延长测试步数？**
A: `TOTAL_STEPS=100 WARMUP_BENCH=30` — 更多步数统计结果更稳定，但耗时更长。

**Q: 想测不同 batch size 的吞吐-显存曲线？**
A: 循环跑多组配置：
```bash
for mbs in 1 2 4; do
    MBS=$mbs bash scripts/benchmark/bench_fsdp.sh
    mv ${OUTPUT_ROOT}/benchmark/fsdp ${OUTPUT_ROOT}/benchmark/fsdp_mbs${mbs}
done
```
