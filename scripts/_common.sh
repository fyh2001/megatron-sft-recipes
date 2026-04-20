#!/usr/bin/env bash
# 公共环境变量与工具函数
# 被 01/03/04/05/06 等脚本 source 使用
#
# 所有可变参数支持通过环境变量覆盖，例如：
#   CONTAINER_NAME=my_sft HOST_MOUNT=/srv/megatron bash scripts/01_setup_env.sh

# ===== Docker 相关 =====
# NGC PyTorch 镜像，自带 PyTorch + CUDA + TE + Apex + FlashAttention
: "${NGC_IMAGE:=nvcr.io/nvidia/pytorch:25.03-py3}"
: "${CONTAINER_NAME:=swift_sft}"

# 宿主机工作区路径（包含本仓库及 sft-data/ 等）
# 设计：宿主机和容器挂载使用同一绝对路径，方便容器内外互操作
# 覆盖示例：HOST_MOUNT=/srv/other_path bash scripts/01_setup_env.sh
: "${HOST_MOUNT:=/home/ubuntu/perf_opt}"

# 容器内挂载点：和宿主机保持一致
: "${CONTAINER_MOUNT:=${HOST_MOUNT}}"

# ===== uv / venv / 缓存 相关 =====
: "${PYTHON_VERSION:=3.11}"
# venv 放挂载卷，容器重启不丢
: "${VENV_DIR:=${CONTAINER_MOUNT}/.venv}"
# uv 自己的 cache（wheels/source dist），也放挂载卷避免重启重下
: "${UV_CACHE_DIR:=${CONTAINER_MOUNT}/.cache/uv}"
export UV_CACHE_DIR

# ===== 训练路径 =====
: "${DATA_DIR:=${CONTAINER_MOUNT}/sft-data}"
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
