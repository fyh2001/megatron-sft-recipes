#!/usr/bin/env bash
# scripts/fsdp/setup_env.sh
#   FSDP2 + torch.compile 训练环境搭建。
#   对比 Megatron，这里用更轻量的 PyTorch 官方镜像，不依赖 TE/Apex/megatron-core。
#
# 前置条件：
#   - 已安装 docker + NVIDIA Container Toolkit
#   - 已把工作区 rsync/clone 到 GPU 机器
#
# 用法：
#   bash scripts/fsdp/setup_env.sh
#
#   # 自定义镜像：
#   FSDP_BASE_IMAGE=nvcr.io/nvidia/pytorch:25.03-py3 bash scripts/fsdp/setup_env.sh
#
#   # 改挂载路径：
#   HOST_MOUNT=/data/sft bash scripts/fsdp/setup_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

# 默认用 PyTorch 官方 CUDA 镜像（轻量，不含 TE/Apex/megatron-core）
# 也可以直接用 NGC 镜像（更大但自带 flash_attn，省去编译时间）：
#   FSDP_BASE_IMAGE=nvcr.io/nvidia/pytorch:25.03-py3
: "${FSDP_BASE_IMAGE:=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel}"
: "${FSDP_CONTAINER_NAME:=fsdp_sft}"

log "Step 1/3 pulling image: ${FSDP_BASE_IMAGE}"
docker pull "${FSDP_BASE_IMAGE}"

log "Step 2/3 (re)starting container: ${FSDP_CONTAINER_NAME}"
if docker ps -a --format '{{.Names}}' | grep -qx "${FSDP_CONTAINER_NAME}"; then
    log "  container already exists, removing..."
    docker rm -f "${FSDP_CONTAINER_NAME}" >/dev/null
fi

docker run -d --gpus all \
    --name "${FSDP_CONTAINER_NAME}" \
    --shm-size=32g --ipc=host --net=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "${HOST_MOUNT}":"${CONTAINER_MOUNT}" \
    -w "${CONTAINER_MOUNT}" \
    "${FSDP_BASE_IMAGE}" sleep infinity

log "Step 3/3 installing uv + FSDP dependencies"
docker exec -e CONTAINER_MOUNT="${CONTAINER_MOUNT}" \
            -e UV_CACHE_DIR="${UV_CACHE_DIR}" \
            "${FSDP_CONTAINER_NAME}" bash -se <<'INNER_EOF'
set -euo pipefail

# 1. 装 uv
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# 2. 装 FSDP 训练依赖（轻量：accelerate + transformers，不含 megatron/TE/apex）
cd "${REPO_ROOT:-${CONTAINER_MOUNT}/sft-recipes}"
uv pip install --system --break-system-packages --inexact -e ".[fsdp]"

# 3. 装 flash-attn（PyTorch 官方镜像不预装，需要编译；NGC 镜像已有则跳过）
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "Installing flash-attn (may take 5-10 minutes to compile)..."
    uv pip install --system --break-system-packages flash-attn --no-build-isolation
fi

# 4. 验证关键依赖
python <<'PY'
import importlib, sys
mods = ['torch', 'accelerate', 'transformers', 'flash_attn', 'datasets']
print('=' * 60)
for name in mods:
    try:
        m = importlib.import_module(name)
        v = getattr(m, '__version__', '<no __version__>')
        print(f'  OK  {name:25s} {v}')
    except Exception as e:
        print(f'  FAIL {name:25s} {e}')
        sys.exit(1)

import torch
assert torch.cuda.is_available(), "CUDA not available"
print(f'  CUDA devices: {torch.cuda.device_count()}')
print('=' * 60)
print('FSDP2+compile stack ready.')
PY
INNER_EOF

log "Environment ready. Enter the container with:"
log "  docker exec -it ${FSDP_CONTAINER_NAME} bash"
