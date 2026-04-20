#!/usr/bin/env bash
# 公共环境变量与工具函数
# 被 01/03/04/05/06 等脚本 source 使用
#
# 所有可变参数支持通过环境变量覆盖，例如：
#   CONTAINER_NAME=my_sft HOST_MOUNT=/srv/megatron bash scripts/01_setup_env.sh

# ===== Docker 镜像 =====
# 默认用 NGC PyTorch 镜像（海外/NVIDIA 源，自带 TE/Apex/FlashAttention）。
# 如在国内机器，可切换到阿里云 modelscope 官方镜像（自带 ms-swift 生态），
# 首次安装会更快（只要覆盖升级 ms-swift 到 ≥4.1.0 满足 Gemma 4 要求）。
#
# 候选镜像（任选其一，通过环境变量 BASE_IMAGE=... 覆盖默认值）：
#   a) NGC PyTorch（默认）:
#      nvcr.io/nvidia/pytorch:25.03-py3
#   b) 阿里云 modelscope 杭州:
#      modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.10.1
#   c) 阿里云 modelscope 北京:
#      modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.10.1
#   d) 阿里云 modelscope 美西（海外直连友好）:
#      modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.10.1
: "${BASE_IMAGE:=nvcr.io/nvidia/pytorch:25.03-py3}"
# 兼容旧变量名
: "${NGC_IMAGE:=${BASE_IMAGE}}"
BASE_IMAGE="${NGC_IMAGE}"   # 统一走 BASE_IMAGE

: "${CONTAINER_NAME:=swift_sft}"

# 宿主机工作区路径（包含本仓库及 sft-data/ 等）
# 设计：宿主机和容器挂载使用同一绝对路径，方便容器内外互操作
# 覆盖示例：HOST_MOUNT=/srv/other_path bash scripts/01_setup_env.sh
: "${HOST_MOUNT:=/home/ubuntu/perf_opt}"

# 容器内挂载点：和宿主机保持一致
: "${CONTAINER_MOUNT:=${HOST_MOUNT}}"

# ===== uv / 缓存 相关 =====
# 本方案不再使用 venv，直接把业务包装进基础镜像自带的系统 Python。
# 原因：镜像预装的 transformer_engine / apex / flash_attn 是针对镜像自带 torch
# 编译的，一旦依赖链（如 megatron-core>=2.6）让 uv 在 venv 里重装 torch，
# 就会 ABI 失配；而 `uv pip install --system` 能把系统 site-packages 里
# 已有的 torch/TE/apex/flash_attn 识别为"已满足"，不会被重装。
# 对应 Python 大版本（仅供排障参考，不用手动指定）：
#   - NGC nvcr.io/nvidia/pytorch:25.03-py3  → Python 3.12
#   - 阿里云 modelscope ... py311 ...       → Python 3.11
#
# uv 自己的 cache（wheels/source dist），放挂载卷避免容器重启重下
: "${UV_CACHE_DIR:=${CONTAINER_MOUNT}/.cache/uv}"
export UV_CACHE_DIR

# ===== 训练路径 =====
# 仓库根目录（scripts/ 的上一级），用来定位随仓库一起同步过来的 sft-data/
# 这样无论 HOST_MOUNT / CONTAINER_MOUNT 怎么配，sft-data 都跟着仓库走。
_REPO_ROOT_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${REPO_ROOT:=${_REPO_ROOT_DEFAULT}}"
unset _REPO_ROOT_DEFAULT

: "${DATA_DIR:=${REPO_ROOT}/sft-data}"
: "${OUTPUT_ROOT:=${CONTAINER_MOUNT}/megatron_output}"
: "${TRAIN_JSONL:=${DATA_DIR}/train.jsonl}"
: "${VALID_JSONL:=${DATA_DIR}/valid.jsonl}"

# ===== 模型缓存（多机训练时建议指向共享存储）=====
: "${MODELSCOPE_CACHE:=${CONTAINER_MOUNT}/.cache/modelscope}"
: "${HF_HOME:=${CONTAINER_MOUNT}/.cache/huggingface}"

export MODELSCOPE_CACHE HF_HOME

# ===== 训练通用环境 =====
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 默认 8 卡，可被子脚本覆盖
: "${NPROC_PER_NODE:=8}"
export NPROC_PER_NODE

# 共享打印工具
log() { printf '\n[%(%F %T)T] %s\n' -1 "$*" >&2; }
