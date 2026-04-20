#!/usr/bin/env bash
# 01_setup_env.sh
#   在 GPU 服务器上准备训练环境：
#   1. 拉基础镜像（默认 NGC PyTorch，可切阿里云 modelscope 镜像）
#   2. 启动容器，挂载工作区（容器内外同路径）
#   3. 在容器内装 uv，然后 `uv pip install --system --inexact -e .[gpu]`
#      直接把业务包装进镜像自带的系统 Python，让 uv 把镜像预装的
#      torch / TE / apex / flash_attn 视作已满足，避免被依赖链强升 torch。
#
# 前置条件：
#   - 已安装 docker + NVIDIA Container Toolkit
#   - 已把整个工作区 rsync / clone 到 GPU 机器，例如 /home/ubuntu/perf_opt
#
# 用法（在 GPU 机器的宿主机上执行）：
#
#   # 默认：NGC PyTorch 镜像（海外源，自带 TE/Apex/FlashAttention）
#   bash scripts/01_setup_env.sh
#
#   # 切阿里云 us-west-1（海外直连）/ cn-hangzhou / cn-beijing：
#   BASE_IMAGE=modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.10.1 \
#       bash scripts/01_setup_env.sh
#
#   # 改挂载路径：
#   HOST_MOUNT=/data/megatron bash scripts/01_setup_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

log "Step 1/3 pulling image: ${BASE_IMAGE}"
docker pull "${BASE_IMAGE}"

log "Step 2/3 (re)starting container: ${CONTAINER_NAME}"
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    log "  container already exists, removing..."
    docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

docker run -d --gpus all \
    --name "${CONTAINER_NAME}" \
    --shm-size=32g --ipc=host --net=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "${HOST_MOUNT}":"${CONTAINER_MOUNT}" \
    -w "${CONTAINER_MOUNT}" \
    "${BASE_IMAGE}" sleep infinity

log "Step 3/3 installing uv + uv pip install --system --inexact -e .[gpu]"
# 把本脚本所需的环境变量通过 `-e` 传进容器 bash
docker exec -e CONTAINER_MOUNT="${CONTAINER_MOUNT}" \
            -e UV_CACHE_DIR="${UV_CACHE_DIR}" \
            "${CONTAINER_NAME}" bash -se <<'INNER_EOF'
set -euo pipefail

# 1. 装 uv（单文件二进制，秒装）
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# 2. 直接把业务包装进镜像自带的系统 Python
#    --system                : 装进容器的系统 site-packages，而不是 venv
#    --break-system-packages : 绕过 Debian PEP 668 的 EXTERNALLY-MANAGED 标记
#                              （NGC/modelscope 镜像都带这个标记；容器是一次性的，安全）
#    --inexact               : 允许保留已装好的系统包（torch / TE / apex / flash_attn）
#    --editable              : 方便直接改 scripts/ 下 python 代码
#  关键：--system 让 uv 看到系统里已经装了 torch 2.7（NGC）/ 2.8（modelscope），
#  依赖解析时把它视作满足 megatron-core 的 `torch>=2.6` 约束，不会去 pypi 重装，
#  从而保持镜像预装的 TE / apex / flash_attn 的 ABI 兼容性。
cd "${CONTAINER_MOUNT}/megatron-sft-recipes"
uv pip install --system --break-system-packages --inexact -e ".[gpu]"

# 3. 验证关键依赖（直接用系统 python）
python <<'PY'
import importlib, sys
mods = ['torch', 'swift', 'mcore_bridge', 'megatron.core',
        'transformer_engine', 'apex', 'flash_attn', 'datasets']
print('=' * 60)
for name in mods:
    try:
        m = importlib.import_module(name)
        v = getattr(m, '__version__', '<no __version__>')
        print(f'  OK  {name:25s} {v}')
    except Exception as e:
        print(f'  FAIL {name:25s} {e}')
        sys.exit(1)
print('=' * 60)
print('All GPU dependencies ready.')
PY
INNER_EOF

log "Environment ready. Enter the container with:"
log "  docker exec -it ${CONTAINER_NAME} bash"
log "  # 无需 venv，直接用容器里的 python / megatron 命令"
