#!/usr/bin/env bash
# 01_setup_env.sh
#   在 H100 服务器上准备训练环境：
#   1. 拉 NGC PyTorch 镜像（自带 TE/Apex/FlashAttention）
#   2. 启动容器，挂载工作区到 /home/ubuntu/perf_opt（容器内外同路径）
#   3. 在容器内装 uv + 建立继承系统包的 venv + uv pip install -e .[gpu]
#
# 前置条件：
#   - 已安装 docker + NVIDIA Container Toolkit
#   - 已把整个工作区 rsync 到 GPU 机器，比如 /home/ubuntu/perf_opt
#
# 用法（在 GPU 机器的宿主机上执行）：
#   HOST_MOUNT=/home/ubuntu/perf_opt bash scripts/01_setup_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

log "Step 1/3 pulling image: ${NGC_IMAGE}"
docker pull "${NGC_IMAGE}"

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
    "${NGC_IMAGE}" sleep infinity

log "Step 3/3 installing uv + creating venv + uv pip install -e .[gpu]"
# 把本脚本所需的环境变量通过 `-e` 传进容器 bash
docker exec -e VENV_DIR="${VENV_DIR}" \
            -e PYTHON_VERSION="${PYTHON_VERSION}" \
            -e CONTAINER_MOUNT="${CONTAINER_MOUNT}" \
            "${CONTAINER_NAME}" bash -se <<'INNER_EOF'
set -euo pipefail

# 1. 装 uv（单文件二进制，秒装）
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# 2. 创建 venv 并继承系统 site-packages（关键：继承 NGC 镜像预装的 TE/Apex/FA）
cd "${CONTAINER_MOUNT}"
if [ ! -d "${VENV_DIR}" ]; then
    uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}" --system-site-packages
fi

# 3. 激活 venv，装 gpu 可选依赖组
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# --editable 模式：方便后续直接改 scripts/ 下 python 代码
# --inexact 允许保留系统 site-packages 里已有的包，不去改动它们
uv pip install --inexact -e ".[gpu]"

# 4. 验证关键依赖
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
log "  source ${VENV_DIR}/bin/activate"
