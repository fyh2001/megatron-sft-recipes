# SFT Recipes

多后端 SFT 训练方案，单机 8 卡 A100/H100。

| 后端 | 目录 | 适用场景 |
|---|---|---|
| **Megatron** (ms-swift + mcore-bridge) | `scripts/megatron/` | 最大吞吐，支持 TP/PP，需要 TE/Apex |
| **FSDP2 + compile** (Accelerate) | `scripts/fsdp/` | 轻量依赖，torch.compile 加速，任意 HF 模型 |

## 目录结构

```
sft-recipes/
├── pyproject.toml                uv 项目（公共 + gpu/fsdp 可选依赖组）
├── uv.lock
├── .python-version               Python 3.11
├── scripts/
│   ├── _common.sh                公共环境变量（两个后端共享）
│   ├── 02_convert_data.py        Arrow -> jsonl 数据转换（Mac 本地执行）
│   ├── megatron/                 Megatron 后端
│   │   ├── 01_setup_env.sh       NGC/modelscope 镜像 + ms-swift 生态
│   │   ├── 03_sft_qwen25_7b.sh
│   │   ├── 04_sft_gemma4_e4b.sh
│   │   ├── 05_sft_gemma4_31b.sh
│   │   ├── 06_infer.sh
│   │   ├── 07_sft_ministral3_3b.sh
│   │   ├── 08_sft_ministral3_text.sh
│   │   └── convert_ministral3_to_llama.py
│   └── fsdp/                     FSDP2 + compile 后端
│       ├── setup_env.sh          PyTorch 官方镜像 + accelerate
│       ├── train.py              手写训练脚本（FSDP2 + compile + per-message loss）
│       ├── accelerate_config.yaml
│       ├── sft_qwen25_7b.sh
│       ├── sft_gemma4_e4b.sh
│       ├── sft_gemma4_31b.sh
│       ├── sft_ministral3_3b.sh
│       └── sft_ministral3_text.sh
├── sft-data/
│   ├── train.jsonl               18819 条对话，223MB
│   └── valid.jsonl               190 条对话，2.7MB
└── docs/
    └── adapting-vlm-to-megatron.md
```

## 数据格式

两个后端共用同一份数据。jsonl 格式，每行一个对话：

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "loss": true},
    {"role": "assistant", "content": "...", "loss": false}
  ]
}
```

`loss` 字段控制该 assistant turn 是否参与训练 loss 计算。`true` = 训练，`false` = 不训练（仅作上下文）。

## 后端对比

| 特性 | Megatron | FSDP2 + compile |
|---|---|---|
| 并行策略 | TP + PP + DP | FSDP (纯数据并行分片) |
| torch.compile | 不支持（用 TE/Apex 替代） | 原生支持 |
| 依赖 | 重（TE, Apex, megatron-core, ms-swift） | 轻（accelerate, transformers） |
| Docker 镜像 | NGC PyTorch / modelscope（~20GB） | PyTorch 官方（~8GB） |
| Per-message loss | ms-swift 原生 `loss` 字段 | 手写 incremental tokenization + label mask |
| Packing | 内置 `--packing true` | 未实现（v2 计划） |
| 模型支持 | ms-swift MODEL_MAPPING 里的模型 | 任何 HF AutoModelForCausalLM |
| Checkpoint 格式 | safetensors（HF 兼容） | safetensors（HF 兼容） |

---

## Quick Start: FSDP2 + compile

### 0. 同步到 GPU 机器

```bash
rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='.ruff_cache' \
    /Users/huangye/Desktop/megatron/ ubuntu@gpu-host:/home/ubuntu/perf_opt/sft-recipes/
```

### 1. 启动容器 + 安装依赖

```bash
cd /home/ubuntu/perf_opt/sft-recipes
bash scripts/fsdp/setup_env.sh
```

默认用 PyTorch 官方 CUDA 镜像。也可以用 NGC 镜像（自带 flash_attn，省编译时间）：

```bash
FSDP_BASE_IMAGE=nvcr.io/nvidia/pytorch:25.03-py3 bash scripts/fsdp/setup_env.sh
```

### 2. 进入容器做训练

```bash
docker exec -it fsdp_sft bash

# Qwen2.5-7B
bash scripts/fsdp/sft_qwen25_7b.sh

# Gemma 4 E4B
bash scripts/fsdp/sft_gemma4_e4b.sh

# Gemma 4 31B（需要 CPU offload）
bash scripts/fsdp/sft_gemma4_31b.sh
```

### 3. 参数覆盖

```bash
MBS=1 GAS=8 LR=2e-5 EPOCHS=3 COMPILE=false \
    bash scripts/fsdp/sft_qwen25_7b.sh
```

### 4. 调试 tokenization（验证 per-message loss 正确性）

```bash
python scripts/fsdp/train.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --train_file sft-data/train.jsonl \
    --output_dir /tmp/debug \
    --debug_tokenization
```

---

## Quick Start: Megatron

### 1. 启动容器 + 安装依赖

```bash
cd /home/ubuntu/perf_opt/sft-recipes
bash scripts/megatron/01_setup_env.sh
```

国内机器可切阿里云镜像：

```bash
BASE_IMAGE=modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.10.1 \
    bash scripts/megatron/01_setup_env.sh
```

### 2. 进入容器做训练

```bash
docker exec -it swift_sft bash

bash scripts/megatron/03_sft_qwen25_7b.sh
bash scripts/megatron/04_sft_gemma4_e4b.sh
bash scripts/megatron/05_sft_gemma4_31b.sh
```

### 3. 推理验证

```bash
OUTPUT_SUB=qwen25_7b bash scripts/megatron/06_infer.sh
```

---

## 推荐配置（8x80GB）

### Megatron 后端

| 模型 | TP | PP | MBS | GBS | Recompute |
|---|---|---|---|---|---|
| Qwen2.5-7B | 2 | 1 | 2 | 64 | none |
| Gemma 4 E4B | 1 | 1 | 2 | 32 | selective |
| Gemma 4 31B | 4 | 2 | 1 | 16 | full |

### FSDP2 后端

| 模型 | MBS | GAS | GBS_eff | Compile | 备注 |
|---|---|---|---|---|---|
| Qwen2.5-7B | 2 | 4 | 64 | Yes | 宽裕 |
| Gemma 4 E4B | 4 | 2 | 64 | Yes | 宽裕 |
| Gemma 4 31B | 1 | 8 | 64 | Yes | 需 CPU offload |
| Ministral-3 3B (VLM) | 4 | 2 | 64 | Yes | freeze_vision |
| Ministral-3 3B (text) | 4 | 2 | 64 | Yes | 需先 convert_ministral3_to_llama |

## OOM 排查

### Megatron 后端

1. `MBS=1` → 2. `RECOMPUTE=selective` → `full` → 3. `MAX_LEN` 缩短 → 4. 加大 `TP` → 5. 加大 `PP` → 6. `--optimizer_cpu_offload true`

### FSDP2 后端

1. `MBS=1` → 2. 开 `GRAD_CKPT=true` → 3. `MAX_LEN` 缩短 → 4. `CPU_OFFLOAD=true` → 5. `COMPILE=false`（compile 期间有额外显存开销）

## 常见问题

**Q：两个后端可以在同一台机器上共存吗？**
A：可以。Megatron 用容器 `swift_sft`，FSDP 用容器 `fsdp_sft`，互不干扰。但不能同时训练（都要占满 8 卡）。

**Q：per-message loss 在 FSDP 后端怎么验证？**
A：用 `--debug_tokenization` 参数查看前 3 条样本的 token-message 对齐情况，确认 loss=true/false 的 token 边界正确。

**Q：torch.compile 第一步很慢？**
A：正常。前 2-3 步是编译期，之后每步速度会提升 15-40%。如果某个模型 compile 失败，用 `COMPILE=false` 回退。

**Q：容器里 `uv pip install` 报 "could not build wheels for apex" / "flash-attn"？**
A：说明在往 venv 装。必须用 `--system`。参考对应后端的 setup_env.sh。
