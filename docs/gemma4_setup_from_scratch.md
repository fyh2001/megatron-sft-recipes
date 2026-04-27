# Gemma-4-26B-A4B-it SFT — 从零搭建复现环境

> 目标读者：拿到一台干净的 8×H100 80GB 机器，想把整个项目从头跑一遍。
>
> **预计耗时**：硬件具备的前提下，环境搭建 ≈ 90 min（其中模型下载 50 GB 占大头），跑通 P0 sanity ≈ 25 min，跑完 P5 peak（30 step bench）≈ 15 min。

---

## 0. 前置 checklist（开干前检查）

| 项 | 要求 | 校验命令 |
|---|---|---|
| GPU | 8 × H100 80GB SXM（NVSwitch / NV18 / 477 GB/s） | `nvidia-smi -L` 看到 8 张 H100 |
| Driver | ≥ 575（CUDA 12.9 兼容） | `nvidia-smi` 顶上的 Driver Version |
| Docker | 已装且能用 GPU | `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` |
| 主机磁盘 | ≥ 200 GB 空闲（模型 50 GB + 容器 50 GB + outputs/log 余量） | `df -h /home` |
| 主机 RAM | ≥ 256 GB（CPU offload 用得到 200 GB） | `free -g` |
| 网络 | 能访问 ModelScope（或 HF Hub）下载模型 | `curl -I https://www.modelscope.cn` |

---

## 1. 拉镜像 + 起容器

```bash
# 镜像（其中包含 cuda 12.9 / torch 2.10 / py3.12 / vllm / modelscope / swift 4.1.2 base）
IMAGE="modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.0-modelscope1.35.4-swift4.1.2"
docker pull "${IMAGE}"   # ≈ 30 GB compressed, 60 GB uncompressed

# 起容器（命名 fsdp_sft，全部 8 卡，挂主机 /home/ubuntu）
docker run -d --name fsdp_sft \
    --gpus all --ipc=host --network host \
    --shm-size=128g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /home/ubuntu:/home/ubuntu \
    -w /home/ubuntu/fyh \
    "${IMAGE}" sleep infinity

# 验证
docker exec fsdp_sft nvidia-smi -L | head -3
docker exec fsdp_sft python3 -c "import torch; print(torch.__version__, torch.cuda.device_count())"
# 期望: 2.10.0+cu129  8
```

> 后续所有命令都会通过 `docker exec fsdp_sft bash -lc "..."` 进容器跑。容器名固定为 `fsdp_sft`，因为 phase 脚本里写死了这个名字。如果你想换名字，全局 `sed -i 's/fsdp_sft/your_name/g' scripts/gemma4_opt/*.sh` 即可。

---

## 2. Clone 两个仓库

```bash
mkdir -p /home/ubuntu/fyh && cd /home/ubuntu/fyh

# Recipes 仓（脚本 + 文档 + sitecustomize）
git clone https://github.com/fyh2001/megatron-sft-recipes.git

# ms-swift fork（gemma4-complete 分支，含 token stats + FA3 dispatch 三个改动）
git clone -b gemma4-complete https://github.com/fyh2001/ms-swift.git ms-swift-fork
```

最终目录长这样：
```
/home/ubuntu/fyh/
├── megatron-sft-recipes/    ← scripts/docs
├── ms-swift-fork/           ← swift 改造版
├── megatron_output/         ← 跑出来的 run 都存这里（脚本会自动建）
└── sft-data/                ← 数据集（下面 step 4）
```

---

## 3. 下载模型（51.6 GB）

容器内已有 modelscope CLI：
```bash
docker exec fsdp_sft bash -lc "
modelscope download \
    --model google/gemma-4-26B-A4B-it \
    --local_dir /home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it
"
```

**校验**（应该 ~51.6 GB，2 个 safetensors shard，1013 keys）：
```bash
docker exec fsdp_sft bash -lc "
du -sh /home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it
ls /home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it/*.safetensors | wc -l
"
# 期望: 51.6G ... \n 2
```

> 如果你的网络更适合走 HF Hub，把上面的 `modelscope download` 替换成 `huggingface-cli download google/gemma-4-26B-A4B-it`，但路径仍然落到 `/home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it`（脚本 hardcode 这个路径，要么按这个落，要么改 phase 脚本里的 `MODEL=...`）。

---

## 4. 准备数据集

**约定路径**：`/home/ubuntu/fyh/sft-data/train.jsonl`（18819 条多轮 VLM/text 对话）。

如果你有自己的 SFT 数据，直接放这里即可。**格式要求**：每行一个 JSON，必须有 `messages` 字段（OpenAI chat format），可选 `images`（VLM 训练）。最小示例：

```jsonl
{"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]}
{"messages": [{"role": "user", "content": "Explain SDPA"}, {"role": "assistant", "content": "..."}]}
```

如果只是想测吞吐而不在乎数据内容，可以拿任意 18k 行的 jsonl 凑数。**不要**用 18k 以下的小数据集 —— 30 步 bench 的 GBS=4 已经吃 120 个样本，过小数据集会被 epoch wraparound 影响计时。

校验：
```bash
wc -l /home/ubuntu/fyh/sft-data/train.jsonl
# 期望: 18819 (或 ≥ 1000，对吞吐测量没差)
```

---

## 5. 容器内安装代码包

镜像自带的 `swift 4.1.2` 不够（pypi 老版本有 SP bug，且 gemma4 dispatch 缺失）。**用我们的 fork 覆盖**：

```bash
docker exec fsdp_sft bash -lc "
set -e
# 用 gemma4-complete 分支的源码（editable install），覆盖容器自带的 4.1.2
pip install -e /home/ubuntu/fyh/ms-swift-fork --no-deps
# 校验
python3 -c 'import swift; print(swift.__version__)'
"
# 期望: 4.2.0.dev0
```

**Liger Kernel** 容器自带 0.7.0，足够（我们的 dispatch 是自带的 `liger_gemma4_patch.py`，不依赖 upstream gemma4 dispatch）：
```bash
docker exec fsdp_sft bash -lc "pip show liger-kernel | grep Version"
# 期望: Version: 0.7.0  (或更新)
```

如果输出 `not installed`：
```bash
docker exec fsdp_sft bash -lc "pip install liger-kernel==0.7.0"
```

---

## 6. 应用 modeling_gemma4 兼容补丁

镜像里 `transformers/models/gemma4/modeling_gemma4.py` 是 upstream 原版。我们对它有 4 处改动（FA fallback / TP plan / DTensor scalar / chunked CE），patch 文件就在 recipes 仓里：

```bash
docker exec fsdp_sft bash -lc "
cd /usr/local/lib/python3.12/site-packages/transformers/models/gemma4

# 备份原版（一旦想还原可以回来）
cp modeling_gemma4.py modeling_gemma4.py.orig

# 应用 patch（diff 格式是 ed 命令式，不是 unified diff）
patch < /home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/gemma4_modeling_compat.patch
"
```

**校验 md5（关键）**：
```bash
docker exec fsdp_sft md5sum /usr/local/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py
# 期望: 39ebf386a992fea9eac0883f459ac658
```

> 如果 md5 对不上，说明 patch 没干净 apply，**别继续往后跑** —— 后面 50% 的报错都因为这个。可以 `cp modeling_gemma4.py.orig modeling_gemma4.py` 还原后重新 patch。

---

## 7. （可选）链 nsys 2025 到 PATH

只在你想跑 `run_with_nsys.sh` 抓 kernel profile 时需要。镜像里默认有但路径不在 PATH：
```bash
docker exec fsdp_sft ln -sf \
    /opt/nvidia/nsight-compute/2025.2.1/host/target-linux-x64/nsys \
    /usr/local/bin/nsys
docker exec fsdp_sft nsys --version
# 期望: NVIDIA Nsight Systems version 2025.2.1
```

---

## 8. Smoke test（5 分钟内确认环境通）

跑 P0 default baseline 的最小化版本（FSDP2 + SP=2 + 5 step）：

```bash
cd /home/ubuntu/fyh/megatron-sft-recipes
# 临时压短，省时间
TOTAL_STEPS=5 WARMUP_BENCH=1 \
    bash scripts/gemma4_opt/p0_baseline_fsdp2.sh
```

成功标志（在 `megatron_output/gemma4_opt/p0_baseline_fsdp2/run_*/`）：
- `STATUS` 文件第一行 = `SUCCESS — see report.json`
- `report.json` 存在，`mean_step_time_ms` 在 1500-2500 范围
- `logging.jsonl` 至少 5 行带 `loss / tokens_this_step` 的记录

如果失败，按 STATUS 里的关键字到 [`docs/gemma4_debug_log.md`](gemma4_debug_log.md) 找。9 成都是这三个：
1. `Tensor has no attribute device_mesh` → `cpu_ram_efficient_loading=true` 没改 false（脚本里默认是 false，自动应该不会撞）
2. `head_dim=512` → modeling_gemma4 patch 没装好（重做 step 6）
3. `Could not find ... Gemma4AudioLayer` → `FSDP_TRANSFORMER_CLS_NAMES` 环境变量没传（脚本里默认带了，自动不会撞）

---

## 9. 跑生产 baseline（DS prod，~25 min）

确认 smoke 通了之后，跑 DeepSpeed 生产配置 baseline（这是优化基准点）：

```bash
cd /home/ubuntu/fyh/megatron-sft-recipes
bash scripts/gemma4_opt/p0_baseline_ds_prod.sh
```

跑完看 `megatron_output/gemma4_opt/p0_baseline_ds_prod/run_*/report.json`，关键字段：

```jsonc
{
  "mean_step_time_ms": ~86000,        // ≈ 86 秒/step（GAS=16，每 step = 16 micro）
  "tokens_per_sec_per_gpu": ~1521,    // padded 公式，DS prod baseline 期望
  "peak_mem_gib_from_swift_log": ~43, // 大幅低，因为 DS offload(opt+param) 到 CPU
  "actual_total_wall_min": ~20.0,     // 40 step 的实测 wall
}
```

这个数字记下来，后面所有优化都跟它对比。**理论 full-epoch 422 min**（18819 / 64 GBS × 86 sec / 60）。

---

## 10. 跑 P5 peak（30 step bench，~12 min）

终极优化配置：FSDP2 + packing + Liger + AC=off + GBS=4 native。

```bash
cd /home/ubuntu/fyh/megatron-sft-recipes
bash scripts/gemma4_opt/p5_liger.sh
```

跑完看 `megatron_output/gemma4_opt/p5_liger/run_*/report.json`，关键字段（**期望复现**）：

```jsonc
{
  "mean_step_time_ms": ~2781,         // 31× 比 DS 快
  "tokens_per_sec_per_gpu": ~5891,    // padded 公式，3.9× 比 DS
  "peak_mem_gib_from_swift_log": ~64.91,
  "mfu_pct_active_params": ~3.5,      // MoE-correct active-param MFU
}
```

`logging.jsonl` 里看 `tokens_global_per_sec` / `tokens_this_step`，得 real_TPS（packing 之后这个数和 padded 公式接近，因为每 micro 真填 16k）。

---

## 11. 跑全套 phase（可选，~2 小时）

如果你想完整复现 P0g → P8 9 期：

```bash
cd /home/ubuntu/fyh/megatron-sft-recipes

# P0g: FSDP2 ↔ DS 数值对齐（大约 30 min）
bash scripts/gemma4_opt/p0_train_align_fsdp2_gbs64.sh

# P1: GBS sweep（7 个配置点，约 70 min）
bash scripts/gemma4_opt/p1_gbs_sweep.sh

# P2-P5: 单期约 10-15 min
bash scripts/gemma4_opt/p2_no_ac.sh
bash scripts/gemma4_opt/p3_reshard.sh
bash scripts/gemma4_opt/p4_packing.sh
bash scripts/gemma4_opt/p5_liger.sh

# P6: torch.compile（会失败但保留 log，~3 min）
bash scripts/gemma4_opt/p6_compile.sh

# P8: GBS resweep on full stack（3 配置点，约 50 min）
bash scripts/gemma4_opt/p8_gbs_resweep.sh
```

跑完用 `attempts.md` + `report.json` 对照 [`gemma4_phase_delta_summary.md`](gemma4_phase_delta_summary.md) 校验数字。

---

## 12. 阅读路径

跑通后想理解为什么是这个配置：
1. 先 [`gemma4_README.md`](gemma4_README.md) — 5 分钟看总览
2. 再 [`gemma4_phase_delta_summary.md`](gemma4_phase_delta_summary.md) — 看每期增量
3. 想深入：[`gemma4_optimization_walkthrough.md`](gemma4_optimization_walkthrough.md) — 9 期详细
4. 想复盘错误：[`gemma4_debug_log.md`](gemma4_debug_log.md) — 4 段式 debug 集中册

---

## 常见坑（Setup 阶段最容易撞到的 5 个）

### Q1：`pip install -e ms-swift-fork` 报 `accelerate>=1.13.0 not satisfied`

**原因**：镜像里 accelerate 是 1.x。  
**修**：`docker exec fsdp_sft pip install -U accelerate` （别加 --no-deps）。

### Q2：跑 P0 时撞 `Tensor has no attribute device_mesh`

**原因**：FSDP_CPU_RAM_EFFICIENT 没设 false。gemma4 是 VLM，root-level 有 vision/embed 不被 auto_wrap 包住，cpu_ram_efficient=true 时这些 plain Tensor 和 DTensor 混合崩。  
**修**：phase 脚本默认已经传 `FSDP_CPU_RAM_EFFICIENT=false`。如果你自己写命令，别忘了加。

### Q3：跑 P4/P5 packing 时撞 `Template gemma4 does not support padding free or packing`

**原因**：sitecustomize.py 没生效。  
**修**：检查启动命令里有 `PYTHONPATH=<repo>/scripts/gemma4_opt/_sdp_preamble:$PYTHONPATH`，且**这个路径在 PYTHONPATH 最前面**。验证：
```bash
docker exec fsdp_sft bash -lc "
PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble \
python3 -c 'import sitecustomize; print(\"OK\")'
"
# 期望 stderr 出现 5 行 [sitecustomize] ... 然后 stdout: OK
```

### Q4：跑完报告 `tokens_this_step` 字段缺失

**原因**：用的不是 fyh2001/ms-swift gemma4-complete 分支，token stats patch 没装。  
**修**：重做 step 5（pip install -e ms-swift-fork）。验证：
```bash
docker exec fsdp_sft bash -lc "
python3 -c 'from swift.trainers.seq2seq_trainer import Seq2SeqTrainer; import inspect; src = inspect.getsource(Seq2SeqTrainer.training_step); assert \"_step_token_count\" in src, \"NOT patched!\"; print(\"OK\")'
"
```

### Q5：`docker exec fsdp_sft pkill ...` 报 `executable file not found`

**原因**：镜像精简版没 `procps`。  
**修**：`docker exec fsdp_sft apt-get install -y procps`。或者用 `docker exec fsdp_sft bash -c 'kill $(pgrep -f swift)'`。

---

## 一键 setup 脚本（可选）

如果你不想一步步走，把上面 step 1-7 合成一个脚本：

```bash
cat > /tmp/gemma4_setup.sh <<'EOF'
#!/bin/bash
set -euo pipefail
IMAGE="modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.0-modelscope1.35.4-swift4.1.2"

echo "[1/6] Pulling docker image..."
docker pull "${IMAGE}"

echo "[2/6] Starting container fsdp_sft..."
docker rm -f fsdp_sft 2>/dev/null || true
docker run -d --name fsdp_sft --gpus all --ipc=host --network host \
    --shm-size=128g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /home/ubuntu:/home/ubuntu -w /home/ubuntu/fyh \
    "${IMAGE}" sleep infinity
sleep 3

echo "[3/6] Cloning repos..."
mkdir -p /home/ubuntu/fyh && cd /home/ubuntu/fyh
[ -d megatron-sft-recipes ] || git clone https://github.com/fyh2001/megatron-sft-recipes.git
[ -d ms-swift-fork ] || git clone -b gemma4-complete https://github.com/fyh2001/ms-swift.git ms-swift-fork

echo "[4/6] Downloading model (51.6 GB, 5-30 min depending on bandwidth)..."
docker exec fsdp_sft bash -lc "
modelscope download --model google/gemma-4-26B-A4B-it \
    --local_dir /home/ubuntu/.cache/modelscope/models/google/gemma-4-26B-A4B-it
"

echo "[5/6] Installing ms-swift fork in container..."
docker exec fsdp_sft pip install -e /home/ubuntu/fyh/ms-swift-fork --no-deps

echo "[6/6] Applying modeling_gemma4 compat patch..."
docker exec fsdp_sft bash -lc "
cd /usr/local/lib/python3.12/site-packages/transformers/models/gemma4
cp modeling_gemma4.py modeling_gemma4.py.orig
patch < /home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/gemma4_modeling_compat.patch
md5sum modeling_gemma4.py
"

echo ""
echo "Setup done. Verify md5 above is: 39ebf386a992fea9eac0883f459ac658"
echo "Now place your training data at /home/ubuntu/fyh/sft-data/train.jsonl"
echo "Then run: bash /home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/p5_liger.sh"
EOF
chmod +x /tmp/gemma4_setup.sh
bash /tmp/gemma4_setup.sh
```

---

## 跑完后产出物在哪

```
/home/ubuntu/fyh/megatron_output/gemma4_opt/
├── p0_baseline_fsdp2/run_<TS>_*/
├── p0_baseline_ds_prod/run_<TS>_*/    ← DS prod baseline
├── p0_train_align/run_<TS>_*/
├── p1_gbs_sweep/run_<TS>_*/           ← 7 个配置
├── p2_no_ac/...
├── p3_reshard/...
├── p4_packing/...
├── p5_liger/run_<TS>_*/               ← peak ⭐
├── p6_compile/...
└── p8_gbs_resweep/...
```

每个 run 目录固定 5 件套：`cmd.sh / stdout.log / STATUS / fsdp_override.json / report.json`，外加 `logging.jsonl` symlink 和 `dcgm_tc.tsv` / `gpu_metrics.jsonl`。

详见 [`gemma4_README.md` "🔴 我要审计某次 run"](gemma4_README.md#-我要审计某次-run-的原始数据10-minrun) 节。

---

**好运。** 卡到的话先看 step 11 那 5 个常见坑，再去 `gemma4_debug_log.md` 翻 4 段式 debug。
