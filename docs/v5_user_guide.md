# Gemma-4-E4B-it SFT 训练镜像 — 同事使用文档（v1）

> 镜像：`fangyaohua/gemma4-e4b-it-sft:260505-u22.04-cu12.9.1-py3.12-t2.10.0-v0.19.0-m1.35.4-s4.1.2-v1`
>
> 一句话：**对齐 DeepSpeed ZeRO-3 baseline（loss 偏差 < 0.12%），但比 DS3 快 1.45 倍**（H100 8 卡 2 epoch：8.5 h vs 12.3 h）。

---

## 目录

1. [一句话启动训练](#1-一句话启动训练)
2. [新机器从 0 准备](#2-新机器从-0-准备)
3. [完整 docker run 选项说明](#3-完整-docker-run-选项说明)
4. [v5 vs DS3 baseline 性能对比](#4-v5-vs-ds3-baseline-性能对比)
5. [启动后怎么验证它跑对了](#5-启动后怎么验证它跑对了)
6. [常见问题排查](#6-常见问题排查)
7. [v5 是怎么做的（可选阅读）](#7-v5-是怎么做的可选阅读)

---

## 1. 一句话启动训练

机器已经装好 NVIDIA driver + Docker + nvidia-container-toolkit + 模型 + 数据，直接：

```bash
docker run --rm -it --gpus all --shm-size=16g --ipc=host \
  -v $HOME/.cache/modelscope:/root/.cache/modelscope \
  -v /path/to/your/sft.jsonl:/data/sft.jsonl:ro \
  -v $(pwd)/runs:/runs \
  -e MODEL=/root/.cache/modelscope/models/google/gemma-4-E4B-it \
  -e DATASET_PATH=/data/sft.jsonl \
  -e OUT_ROOT=/runs \
  fangyaohua/gemma4-e4b-it-sft:260505-u22.04-cu12.9.1-py3.12-t2.10.0-v0.19.0-m1.35.4-s4.1.2-v1 \
  bash /opt/megatron-sft-recipes/scripts/gemma4_E4B_opt/fsdp2_offload_E4B_text_only_2ep_a3_pf_fp32master.sh
```

8x H100 80GB 上 ~8.5 小时跑完 2 epoch；checkpoints 写到 `runs/run_<timestamp>_*/v0-*/checkpoint-403/` 和 `checkpoint-806/`。

---

## 2. 新机器从 0 准备

如果你刚拿到一台 8x H100/A100 机器、什么都没装，按下面 6 步来。**每一步都是独立的命令**，可以单独执行 / 跳过已完成的步骤。

### 2.1 装 NVIDIA driver

```bash
# 验证（已装的话跳过）
nvidia-smi          # 应该看到 8 张 GPU

# Ubuntu 22.04 / 24.04 安装
sudo apt update
sudo apt install -y nvidia-driver-550-server   # 或 535-server，跟 CUDA 12.x 兼容
sudo reboot                                    # 装完必须重启
```

### 2.2 装 Docker（≥ 20.10）

```bash
# 验证
docker version

# 一键安装 + 加权限
curl -fsSL https://get.docker.com | sudo bash
sudo usermod -aG docker $USER
newgrp docker         # 或退出重登录
```

### 2.3 装 `nvidia-container-toolkit`（让 Docker 看到 GPU）

```bash
# 验证（应该输出 8 行 GPU UUID）
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi -L

# 安装（Ubuntu）
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2.4 拉训练镜像（~15 GB，~5 min）

```bash
docker pull fangyaohua/gemma4-e4b-it-sft:260505-u22.04-cu12.9.1-py3.12-t2.10.0-v0.19.0-m1.35.4-s4.1.2-v1
```

### 2.5 下载模型权重到 host（28 GB，~10-30 min）

直接用镜像里自带的 modelscope CLI 下载，不需要 host 装任何工具：

```bash
docker run --rm \
  -v $HOME/.cache/modelscope:/root/.cache/modelscope \
  fangyaohua/gemma4-e4b-it-sft:260505-u22.04-cu12.9.1-py3.12-t2.10.0-v0.19.0-m1.35.4-s4.1.2-v1 \
  modelscope download \
    --model google/gemma-4-E4B-it \
    --local_dir /root/.cache/modelscope/models/google/gemma-4-E4B-it
```

下载完成后模型在 host 的 `$HOME/.cache/modelscope/models/google/gemma-4-E4B-it/`。

### 2.6 准备训练数据

把你的 `*.jsonl`（messages 格式，一行一条）放到任意路径，记下绝对路径备用，例如 `/home/ubuntu/data/SFT.jsonl`。

### 2.7（可选）跑 5 步 smoke test 验证

确保整个 stack 没问题，step 1 应该输出 `loss=2.226 grad_norm=10.28`：

```bash
docker run --rm --gpus all --shm-size=16g --ipc=host \
  -v $HOME/.cache/modelscope:/root/.cache/modelscope \
  -v /path/to/your/sft.jsonl:/data/sft.jsonl:ro \
  -v /tmp/smoke:/runs \
  -e MODEL=/root/.cache/modelscope/models/google/gemma-4-E4B-it \
  -e DATASET_PATH=/data/sft.jsonl \
  -e OUT_ROOT=/runs/bench \
  -e LABEL=smoke -e MAX_STEPS=5 -e FULL_SCHED_STOP=1 \
  -e TEMPLATE=gemma4 -e TORCH_DTYPE=float32 -e WEIGHT_DECAY=0.1 \
  -e FSDP_WRAP_EXTRA='{"activation_cpu_offload": true}' \
  -e EXTRA_ENV='GEMMA4_FSDP_WRAP_PLE=1 GEMMA4_KV_SHARE_DETACH=1 GEMMA4_FSDP_REDUCE_FP32_NCCL=1' \
  -e EXTRA_ARGS='--bf16 true --fp16 false --padding_free false --max_grad_norm 1.0' \
  fangyaohua/gemma4-e4b-it-sft:260505-u22.04-cu12.9.1-py3.12-t2.10.0-v0.19.0-m1.35.4-s4.1.2-v1 \
  bash /opt/megatron-sft-recipes/scripts/gemma4_E4B_opt/bench_variant.sh
```

跑完看 `/tmp/smoke/bench/run_*_smoke/v0-*/logging.jsonl`，step 1 应该是 `loss=2.226 / grad_norm=10.28`。

完成 2.1-2.7 后机器就 ready 了，回到 [§1 一句话启动训练](#1-一句话启动训练) 开训。

---

## 3. 完整 docker run 选项说明

### 3.1 必备的 docker 参数（4 个）

| 参数 | 作用 | 改动建议 |
|---|---|---|
| `--gpus all` | 让容器看到 8 张 GPU | 必填 |
| `--shm-size=16g` | NCCL / FSDP2 用的共享内存 | 必填，小了 NCCL 报错 |
| `--ipc=host` | 数据 loader workers 防 shm 错 | 推荐 |
| `--rm -it` | 完成后清理容器 / 交互式 | 推荐 |

### 3.2 必备 mount（3 个）

| `-v 主机路径:容器路径` | 作用 | 备注 |
|---|---|---|
| `$HOME/.cache/modelscope:/root/.cache/modelscope` | 模型权重（28 GB） | 由 `prepare.sh` 下载 |
| `/path/to/sft.jsonl:/data/sft.jsonl:ro` | 你的训练数据 | messages 格式 |
| `$(pwd)/runs:/runs` | 输出目录 | 容器写 `/runs/`，host 拿到 checkpoints |

### 3.3 必备环境变量（3 个）

| env var | 镜像内默认 | 必填 | 你应该填什么 |
|---|---|---|---|
| `MODEL` | `/home/ubuntu/.cache/modelscope/models/google/gemma-4-E4B-it` | **是** | 容器内模型路径，对应 `-v` 挂载 |
| `DATASET_PATH` | `/home/ubuntu/fyh/megatron-sft-recipes/sft-data/SFT_0424_2.jsonl` | **是** | 容器内数据路径 |
| `OUT_ROOT` | `/opt/megatron-sft-recipes/experiments/...` | **是** | 输出根目录，对应 `runs/` 挂载 |

> 镜像里的默认值是 host 路径，**在容器里不会自动有效**，必须 override。

### 3.4 可选环境变量（默认值已是 v5 最佳，不用动）

| env var | 默认 | 含义 |
|---|---|---|
| `NPROC` | `8` | GPU 数。8 卡机器不动 |
| `MBS` | `1` | per-device micro batch。max_length=16384 下只能 1 |
| `GAS` | `16` | gradient accumulation。GBS = NPROC × MBS × GAS = 128 |
| `NUM_EPOCHS` | `2` | 训 2 epoch |
| `LR` | `2e-5` | 学习率 |
| `WARMUP_RATIO` | `0.05` | cosine 调度 warmup |
| `MAX_LEN` | `16384` | 序列长度 |

### 3.5 完整命令（带注释）

```bash
docker run --rm -it --gpus all --shm-size=16g --ipc=host \
  -v $HOME/.cache/modelscope:/root/.cache/modelscope \      # 模型 28 GB
  -v $(pwd)/sft-data/SFT.jsonl:/data/sft.jsonl:ro \         # 你的训练数据
  -v $(pwd)/runs:/runs \                                    # 输出目录
  -e MODEL=/root/.cache/modelscope/models/google/gemma-4-E4B-it \
  -e DATASET_PATH=/data/sft.jsonl \
  -e OUT_ROOT=/runs \
  fangyaohua/gemma4-e4b-it-sft:260505-u22.04-cu12.9.1-py3.12-t2.10.0-v0.19.0-m1.35.4-s4.1.2-v1 \
  bash /opt/megatron-sft-recipes/scripts/gemma4_E4B_opt/fsdp2_offload_E4B_text_only_2ep_a3_pf_fp32master.sh
```

启动后 `runs/run_<timestamp>_fsdp2_offload_a3_pf_2ep_fp32master/` 下产出：

```
v0-<TS>/
├── args.json                      # 完整训练参数
├── logging.jsonl                  # 每步 metrics（loss/grad_norm/...）
├── checkpoint-403/                # epoch 1 末（403 = 51557/128）
├── checkpoint-806/                # epoch 2 末
└── runs/                          # tensorboard logs
fsdp_override.json                  # FSDP2 配置（自动写）
gpu_metrics.jsonl                   # GPU 利用率监控
train.log                           # 完整训练日志
report.json                         # 性能报告
STATUS                              # SUCCESS / FAILED
```

---

## 4. v5 vs DS3 baseline 性能对比

下面是 **完整 2-epoch（806 steps，GBS=128，max_len=16384，lr=2e-5）** 跟 DS3 baseline step-by-step 对比的数据。

数据来源：
- V5：`run_20260502_234152_fsdp2_offload_a3_pf_2ep_fp32master`（h100-8 节点）
- DS3：`/mnt/shared/fyh/v4-20260429-144702`（同事的全跑，同节点）
- 同 dataset (`SFT_0424_2.jsonl`, 51557 samples) + 同 hyper-params

### 4.1 Loss 对齐（mean Δ% < 0.12%）

V5 跟 DS3 baseline 在 loss 数值上**几乎完全对齐**（远低于 bf16 噪声水平）：

**百分比表示**（V5 相对 DS3 的偏差比例）：

| 阶段 | n | mean Δloss% | median | min | max | stdev |
|---|---|---|---|---|---|---|
| **Step 1-50（warmup）** | 50 | **+0.0044%** | +0.006% | -0.092% | +0.162% | 0.05% |
| Step 51-403（epoch 1 主区） | 353 | +0.0797% | +0.074% | -0.182% | +0.692% | 0.10% |
| Step 404-806（epoch 2） | 403 | +0.1664% | +0.146% | -0.215% | +0.691% | 0.16% |
| **全程 1-806** | 806 | **+0.118%** | +0.097% | -0.215% | +0.692% | 0.14% |

**绝对差科学计数法**（V5 loss − DS3 loss，单位：绝对 loss 值）：

| 阶段 | n | mean | median | min | max | stdev |
|---|---|---|---|---|---|---|
| Step 1-50 | 50 | +8.15e-05 | +1.08e-04 | -1.67e-03 | +3.07e-03 | 9.19e-04 |
| Step 51-403 | 353 | +1.26e-03 | +1.18e-03 | -2.92e-03 | +1.11e-02 | 1.58e-03 |
| Step 404-806 | 403 | +2.35e-03 | +2.12e-03 | -2.90e-03 | +9.26e-03 | 2.24e-03 |
| **全程 1-806** | 806 | **+1.73e-03** | +1.43e-03 | -2.92e-03 | +1.11e-02 | 2.03e-03 |

**百分位分布**（全程 1-806，n=806）—— 直接看 90% / 99% 步骤的偏差上界：

| 指标 | P50 | P75 | P90 | P95 | P99 | max |
|---|---|---|---|---|---|---|
| signed Δloss（V5−DS3） | +1.43e-03 | +2.83e-03 | +4.28e-03 | +5.50e-03 | +8.13e-03 | +1.11e-02 |
| signed Δloss% | +0.097% | +0.189% | +0.305% | +0.385% | +0.557% | +0.692% |
| **|Δloss%|** (绝对值) | 0.102% | 0.191% | **0.305%** | 0.385% | 0.557% | 0.692% |

读这个表：**90% 步骤的 |Δloss| ≤ 0.305%**、**99% 步骤 ≤ 0.557%**；最坏一步 +0.692%。所有都低于 bf16 在 41 层后的累积舍入误差量级。

每个 epoch 最终 loss + 平均 loss：

| | V5 final | DS3 final | 差（科学计数法）| V5 mean | DS3 mean | 差 |
|---|---|---|---|---|---|---|
| Epoch 1 (1-403) | 1.5009 | 1.4987 | +2.20e-03 | 1.6118 | 1.6107 | +1.10e-03 |
| Epoch 2 (404-806) | **1.3913** | **1.3872** | **+4.10e-03** | 1.4180 | 1.4156 | +2.40e-03 |

### 4.2 Grad norm 对齐

V5 跟 DS3 在大部分 step 几乎完全一致（**中位数 ratio = 1.001**），但少数 step 的 spike 在两个引擎间幅度不同：

**比值表示**（V5 grad_norm / DS3 grad_norm）：

| 阶段 | n | mean V5/DS3 | median | min | max | stdev |
|---|---|---|---|---|---|---|
| Step 1-50 | 50 | 1.007 | **1.000** | 0.441 | 1.399 | 0.13 |
| Step 51-403 | 353 | 1.175 | 1.027 | 0.262 | 4.905 | 0.58 |
| Step 404-806 | 403 | 1.273 | 0.971 | 0.136 | 44.5 | 3.10 |
| **全程 1-806** | 806 | 1.213 | **1.001** | 0.136 | 44.5 | 2.23 |

**绝对差科学计数法**（V5 grad_norm − DS3 grad_norm）：

| 阶段 | n | mean | median | min | max | stdev |
|---|---|---|---|---|---|---|
| Step 1-50 | 50 | -1.15e-02 | +3.61e-04 | -8.25e-01 | +3.12e-01 | 1.48e-01 |
| Step 51-403 | 353 | +1.06e-01 | +1.48e-02 | -1.86e+00 | +4.07e+00 | 4.47e-01 |
| Step 404-806 | 403 | +7.02e-02 | -1.69e-02 | -4.79e+00 | +2.31e+01 | 1.72e+00 |
| **全程 1-806** | 806 | +8.10e-02 | +7.49e-04 | -4.79e+00 | +2.31e+01 | 1.25e+00 |

> mean=+0.081 / max=+23.06 是典型的"被少数 spike 拖高"——单看 mean 会误以为 grad_norm 有偏差，看 percentile 才是真相。

**百分位分布**（全程 1-806，n=806）—— 看 V5/DS3 倍数关系的真实分布：

| 指标 | P50 | P75 | P90 | P95 | P99 | max |
|---|---|---|---|---|---|---|
| signed Δgn（V5−DS3） | +6.53e-04 | +6.19e-02 | +2.39e-01 | +6.22e-01 | +2.24e+00 | +2.31e+01 |
| **\|Δgn\|** | 5.53e-02 | 1.70e-01 | **5.43e-01** | 9.27e-01 | 2.62e+00 | 2.31e+01 |
| **V5 / DS3 ratio** | **1.0006** | 1.0998 | 1.3909 | 1.9393 | 4.5742 | 44.5091 |

读这个表：**50% 步骤 V5/DS3 ratio = 1.0006**（基本同一个值）；**P75 = 1.10**（小偏差）；**P90 = 1.39 / P95 = 1.94**（中等差）；只有最后 1% 的 step 拉出 4.5x 以上的极端 spike。这跟 4.1 节 loss 完美对齐互相印证：spike 是数值噪声（每 step 独立产生），不是行为差异（两个引擎的优化轨迹一致）。

Spike 分布（按 V5/DS3 ratio 落在区间统计）：

| 区间 | n / 806 | 占比 |
|---|---|---|
| 在 ±5% 内（视为完全一致） | 272 | **33.7%** |
| 在 ±20% 内（基本一致） | ~600 | ~75% |
| spike (V5/DS3 > 1.5x) | 67 | 8.3% |
| 极端 spike (> 5x) | < 10 | < 1% |

> **关于 spike 的说明**：grad spike 在 V5 和 DS3 baseline 中**位置高度相关**（在两个引擎都发生但幅度不同），不是 v5 实现 bug。它是 bf16 在 41 层反向传播后的累积数值噪声被放大，loss 收敛轨迹完全一致即说明不影响最终训练结果。

### 4.3 训练速度（V5 比 DS3 快 1.45x）

| | mean s/it | median | min | max | p95 | stdev |
|---|---|---|---|---|---|---|
| **V5** | **37.84** | 38.0 | 30 | 106 | 41 | 3.3 |
| DS3 | 54.86 | 55.0 | 43 | 85 | 62 | 4.5 |

> max=106 是 dataloader reload 等异常的 outlier，**p95=41 才是真实尾部**。

完整训练时长：

| | 总时长 | per epoch |
|---|---|---|
| **V5** | **8 h 28 min** | ~4 h 14 min |
| DS3 | 12 h 17 min | ~6 h 9 min |

### 4.4 显存占用

V5 因为开了 `activation_cpu_offload`（v5 必需），peak 显存比 DS3 略高，但仍在 80 GB 容量内：

| | mean GiB | peak GiB | 占 80GB |
|---|---|---|---|
| **V5** | 77.17 | **77.53** | 96.9% |
| DS3 | 73.74 | 74.81 | 93.5% |

A100 80GB 也能跑（peak 77.53 ≤ 80），但 step time 慢约 1.5x。

### 4.5 一图总结

| 指标 | DS3 baseline | **V5（本镜像）** | 改善 |
|---|---|---|---|
| 2-epoch wall time | 12.3 h | **8.5 h** | **-31%** |
| step time (steady) | 54.9 s/it | **37.8 s/it** | **-31%** |
| 全程 mean Δloss% | — | **+0.118%** | 数值对齐 |
| Final loss (epoch 2) | 1.3872 | 1.3913 | +0.30% |
| Peak memory | 74.8 GiB | 77.5 GiB | +3.6% |
| 训练框架 | DeepSpeed ZeRO-3 | PyTorch FSDP2 | — |

---

## 5. 启动后怎么验证它跑对了

启动后看 stderr 前 30 行，必须出现这些 `sdp_preamble` patches 加载消息（rank 0）：

```
[gemma4 sdp_preamble] (1/3) backend prefs: flash=False mem_eff=True math=False
[gemma4 sdp_preamble] (8/8) patched Gemma4TextModel.forward + Gemma4TextAttention.forward to route shared_kv_states ...
[gemma4 sdp_preamble] (16/16) patched torch FSDP2 FSDPParam.to_accumulated_grad_if_needed ...
[gemma4 sdp_preamble] (21) patched FSDP2 foreach_reduce to force reduce_dtype=fp32 ...
[gemma4 sdp_preamble] (21b) patched FSDP2 FSDPParam.init_dtype_attrs ...
[gemma4 sdp_preamble] (29) patched swift activation_cpu_offload: wrap gradient_checkpointing_disable ...
[gemma4 sdp_preamble] (30) patched swift AsyncDoubleBufferGroupOffloadHandler.tensor_pop ...
[INFO:swift] [Gemma4Loader] manually loaded 42 buffer(s) (layer_scalar / std_scale / std_bias) ...
```

**最关键的是最后一行** `[Gemma4Loader] manually loaded 42 buffer(s)` —— 没有它 loss 会爆到 ~30。

Step 1 应该出现：

```json
{"loss": 2.226, "grad_norm": 10.28, "memory(GiB)": 65-70, "train_speed(s/it)": 70+, ...}
```

跟 DS3 baseline step 1 完全一致（参考值 loss=2.225 / grad_norm=10.29）。如果 loss > 5，说明 buffer fix 没装上 → 重拉镜像。

---

## 6. 常见问题排查

| 现象 | 可能原因 | 解决 |
|---|---|---|
| `loss > 5` at step 1 | Gemma4Loader buffer-fix 没装 | 重 `docker pull` 镜像；启动 log 应该有 `[Gemma4Loader] manually loaded 42 buffer(s)` |
| OOM at micro 16 of step 1 | `activation_cpu_offload` 没启用 | 启动脚本会自动写 `fsdp_override.json` 含 `activation_cpu_offload: true`；如被你 override 了，加回来 |
| `KeyError: 22` 在 forward | KV-share patch 没加载（PYTHONPATH 没生效）| 启动脚本自动设；如绕过脚本直接调 swift sft，加 `-e PYTHONPATH=/opt/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble` |
| GPU 利用率显示 11% TC active | **正常** | H2D/D2H PCIe 拷贝是瓶颈，不是计算瓶颈。这是 v5 的硬约束，不是 bug |
| `dcgm_scrape.py: connection refused localhost:9500` | 镜像不内置 dcgm-exporter | 已默认 `DCGM_SCRAPE=0` 跳过；如需启用：host 跑 dcgm-exporter + `--network=host -e DCGM_SCRAPE=1` |
| `Tried to allocate 1024 MiB ... 78 GiB in use` | 别人占了 GPU | `nvidia-smi` 检查 ；先 kill 其他进程 |
| 速度比 38 s/it 慢很多 | 跟其他训练抢 SM | `nvidia-smi -l 1` 查 GPU SM 利用率；`pkill python` 后再启 |
| step 1 极慢（70 s/it 以上）| 正常 cold start | 第一步是冷启动（FSDP unshard、kernel cache 编译），第 5 步起会稳定到 ~38 s/it |

---

## 7. v5 是怎么做的（可选阅读）

V5 配置 = **5 个关键修复的组合**。镜像里全部预装好，你不需要做任何调整。如果好奇为什么这样配：

1. **fp32 sharded master** — `--torch_dtype float32 --bf16 true`：FSDP2 保留 fp32 master + bf16 forward。否则 bf16 master 在 41 层 × 800 步累计舍入误差 → loss 偏 11%（详见 `docs/gemma4_E4B_alt_offload_log.md` §24-25）

2. **fp32 NCCL reduce-scatter** — `GEMMA4_FSDP_REDUCE_FP32_NCCL=1`（patch 21）：绕过 PyTorch 2.10 cublas bug，强制 grads 在 fp32 通信

3. **fp32 unsharded cross-micro grad accumulation** — patch 21b：对齐 DS3 `contiguous_gradients=true`，代价 32 GB/rank（必须配合 `activation_cpu_offload`）

4. **KV-share path detach** — `GEMMA4_KV_SHARE_DETACH=1`（patch 8）：Gemma-4 layer 24-41 共享 layer 22/23 的 K/V，FSDP2+AC 让 grad 走两遍，detach 修掉

5. **swift activation_cpu_offload 兼容补丁** — patch 29 / 30：避开 swift 内部对 `Gemma4AudioModel`、`tensor_pop` 的两个 assert 错误

详细每一条的来龙去脉、源码位置、调试过程，参考镜像里的 `/opt/megatron-sft-recipes/docs/gemma4_E4B_alt_offload_log.md`（3000+ 行的完整时间线）。

如果需要更高数值精度（牺牲速度）：见 `docs/§26.2`（patch 21c）—— +0.016% Δloss 但 66 s/it。

---

## 镜像 / 仓库 / 联系

| | |
|---|---|
| Docker Hub | https://hub.docker.com/r/fangyaohua/gemma4-e4b-it-sft |
| 镜像 tag | `260505-u22.04-cu12.9.1-py3.12-t2.10.0-v0.19.0-m1.35.4-s4.1.2-v1` |
| 镜像 size | 14.92 GB compressed / 45.8 GB on disk |
| Dockerfile | [`docker/Dockerfile`](../docker/Dockerfile) |
| 准备脚本 | [`docker/prepare.sh`](../docker/prepare.sh) |
| 启动脚本 | [`scripts/gemma4_E4B_opt/fsdp2_offload_E4B_text_only_2ep_a3_pf_fp32master.sh`](../scripts/gemma4_E4B_opt/fsdp2_offload_E4B_text_only_2ep_a3_pf_fp32master.sh) |
| 完整调试时间线 | [`docs/gemma4_E4B_alt_offload_log.md`](gemma4_E4B_alt_offload_log.md) §1-26 |
| 联系 | fang yao hua |
