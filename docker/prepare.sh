#!/usr/bin/env bash
# Bring a fresh Linux box (Ubuntu 22.04+ recommended) up to the point
# where you can `docker run` the gemma-4-E4B-it training image.
#
# What this script does, IDEMPOTENTLY (safe to re-run):
#   1) Sanity-check: linux + 8x H100/A100 80GB recommended
#   2) Check NVIDIA driver — print install command if missing (does NOT auto-install)
#   3) Check docker — print install command if missing
#   4) Check nvidia-container-toolkit — auto-install (apt) WITH user confirmation
#   5) docker pull fangyaohua/gemma4-e4b-it-sft:...-v2
#   6) Download model weights via modelscope CLI to $HOME/.cache/modelscope (28 GB)
#   7) Run a 5-step smoke test (requires --dataset for real data; skipped if missing)
#
# Use:
#   bash docker/prepare.sh                           # interactive — recommended
#   bash docker/prepare.sh --no-interactive          # CI mode, skip 'sudo apt install' confirmations
#   bash docker/prepare.sh --skip-model              # skip the 28 GB model download
#   bash docker/prepare.sh --skip-smoke              # skip 5-step smoke test
#   DATASET_PATH=/path/to/sft.jsonl bash docker/prepare.sh   # if you have data, also run smoke test

set -euo pipefail

IMAGE="${IMAGE:-fangyaohua/gemma4-e4b-it-sft:runtime-260506-u22.04-cu12.9.1-py3.12-t2.10.0-v0.19.0-m1.35.4-s4.1.2-r1}"
MODEL_LOCAL_DIR="${MODEL_LOCAL_DIR:-$HOME/.cache/modelscope/models/google/gemma-4-E4B-it}"
MODEL_ID="${MODEL_ID:-google/gemma-4-E4B-it}"
SKIP_MODEL="${SKIP_MODEL:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
INTERACTIVE="${INTERACTIVE:-1}"

for arg in "$@"; do
    case "${arg}" in
        --skip-model)   SKIP_MODEL=1 ;;
        --skip-smoke)   SKIP_SMOKE=1 ;;
        --no-interactive) INTERACTIVE=0 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown arg: ${arg}" >&2; exit 2 ;;
    esac
done

step() { echo; echo "=== [step $1/$2] $3 ==="; }
ok()   { echo "  [ ok ] $*"; }
warn() { echo "  [warn] $*" >&2; }
err()  { echo "  [FAIL] $*" >&2; }
ask_yes() {
    [ "${INTERACTIVE}" = "0" ] && return 0
    read -r -p "  $1 [y/N]: " ans
    case "${ans}" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

TOTAL=7
exit_with_summary() {
    echo
    echo "=================================================="
    if [ "${1:-1}" = "0" ]; then
        echo "All checks passed. Ready to train."
        echo
        echo "Sample full-run command (replace /path/to/your/sft.jsonl):"
        echo
        echo "  docker run --rm -it --gpus all --shm-size=16g --ipc=host \\"
        echo "    -v \$HOME/.cache/modelscope:/root/.cache/modelscope \\"
        echo "    -v /path/to/your/sft.jsonl:/data/sft.jsonl:ro \\"
        echo "    -v \$(pwd)/runs:/runs \\"
        echo "    -e MODEL=/root/.cache/modelscope/models/${MODEL_ID} \\"
        echo "    -e DATASET_PATH=/data/sft.jsonl \\"
        echo "    -e OUT_ROOT=/runs \\"
        echo "    ${IMAGE} \\"
        echo "    bash /opt/megatron-sft-recipes/scripts/gemma4_E4B_opt/fsdp2_offload_E4B_text_only_2ep_a3_pf_fp32master.sh"
        echo
        echo "Or quick 5-step smoke test:  DATASET_PATH=/path/to/your/sft.jsonl bash docker/prepare.sh --skip-model"
    else
        echo "Setup incomplete. Fix the [FAIL] items above and re-run this script."
    fi
    echo "=================================================="
    exit "${1:-1}"
}

# ─────────────────────────────────────────────────────────────────────
step 1 ${TOTAL} "Linux platform sanity"
if [ "$(uname -s)" != "Linux" ]; then
    err "This image is Linux-only ($(uname -s) detected)."
    exit_with_summary 1
fi
ok "Linux $(uname -r)"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    ok "Distro: ${PRETTY_NAME}"
fi

# ─────────────────────────────────────────────────────────────────────
step 2 ${TOTAL} "NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    err "nvidia-smi not found — NVIDIA driver is not installed."
    cat <<'NVIDIA_INSTALL_HINT'
  How to install:
    Ubuntu 22.04+: sudo apt install -y nvidia-driver-550-server  (or 535-server)
    Then reboot and re-run this script.
    See https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/ for details.
NVIDIA_INSTALL_HINT
    exit_with_summary 1
fi
DRIVER_VER="$(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader)"
GPU_NAME="$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader)"
GPU_MEM="$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits)"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
ok "Driver:   ${DRIVER_VER}"
ok "GPU:      ${GPU_NAME} (${GPU_COUNT}x, ${GPU_MEM} MiB each)"
if [ "${GPU_COUNT}" -lt 8 ]; then
    warn "v5 config is tuned for 8 GPUs — you have ${GPU_COUNT}. May still work but adjust GBS=8/GPU_COUNT × GAS."
fi
if [ "${GPU_MEM}" -lt 79000 ]; then
    warn "v5 config requires ~70 GiB peak GPU memory — your GPU has ${GPU_MEM} MiB. Likely OOM."
fi

# ─────────────────────────────────────────────────────────────────────
step 3 ${TOTAL} "Docker"
if ! command -v docker >/dev/null 2>&1; then
    err "docker not found."
    cat <<'DOCKER_INSTALL_HINT'
  How to install on Ubuntu:
    curl -fsSL https://get.docker.com | sudo bash
    sudo usermod -aG docker $USER
    newgrp docker          # or log out / log in
DOCKER_INSTALL_HINT
    exit_with_summary 1
fi
DOCKER_VER="$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo unknown)"
ok "Docker:   ${DOCKER_VER}"
if ! docker info >/dev/null 2>&1; then
    err "docker info failed — current user not in docker group, or daemon not running."
    cat <<'DOCKER_PERM_HINT'
  Fix:
    sudo usermod -aG docker $USER && newgrp docker
    sudo systemctl start docker
DOCKER_PERM_HINT
    exit_with_summary 1
fi

# ─────────────────────────────────────────────────────────────────────
step 4 ${TOTAL} "nvidia-container-toolkit (lets docker see GPUs)"
NVCT_OK=0
# Probe by running a tiny CUDA container with --gpus all
if docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi -L >/dev/null 2>&1; then
    ok "docker --gpus all works"
    NVCT_OK=1
else
    err "docker --gpus all failed — nvidia-container-toolkit not installed or not configured."
    cat <<'NVCT_INSTALL_HINT'
  Install on Ubuntu:
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sudo sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt update && sudo apt install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
NVCT_INSTALL_HINT
    exit_with_summary 1
fi

# ─────────────────────────────────────────────────────────────────────
step 5 ${TOTAL} "Pull training image"
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    ok "Image already cached locally: ${IMAGE}"
else
    echo "  Pulling ${IMAGE}  (~15 GB compressed, ~46 GB on disk)"
    if ask_yes "Proceed with docker pull?"; then
        docker pull "${IMAGE}"
        ok "Pulled."
    else
        warn "Skipped pull. Run 'docker pull ${IMAGE}' before training."
    fi
fi

# ─────────────────────────────────────────────────────────────────────
step 6 ${TOTAL} "Model weights"
if [ "${SKIP_MODEL}" = "1" ]; then
    warn "Skipping model download (--skip-model)."
elif [ -d "${MODEL_LOCAL_DIR}" ] && \
     [ "$(find "${MODEL_LOCAL_DIR}" -name 'model-*.safetensors' 2>/dev/null | wc -l)" -gt 0 ]; then
    SHARDS=$(find "${MODEL_LOCAL_DIR}" -name 'model-*.safetensors' | wc -l)
    SIZE=$(du -sh "${MODEL_LOCAL_DIR}" 2>/dev/null | awk '{print $1}')
    ok "Model already at ${MODEL_LOCAL_DIR} (${SHARDS} shards, ${SIZE})"
else
    echo "  Model not found at ${MODEL_LOCAL_DIR}"
    echo "  Will download via modelscope CLI inside the docker image (~28 GB, ~10-30 min)"
    if ask_yes "Download model now?"; then
        mkdir -p "$(dirname "${MODEL_LOCAL_DIR}")"
        # New image uses ENTRYPOINT that requires git clone before exec; for
        # one-shot modelscope CLI we override entrypoint to /bin/bash directly.
        docker run --rm \
            --entrypoint /bin/bash \
            -v "${HOME}/.cache/modelscope:/root/.cache/modelscope" \
            "${IMAGE}" \
            -c "modelscope download --model '${MODEL_ID}' --local_dir '/root/.cache/modelscope/models/${MODEL_ID}'"
        ok "Downloaded to ${MODEL_LOCAL_DIR}"
    else
        warn "Skipped model download. Set MODEL=/path/to/your/model when running training."
    fi
fi

# ─────────────────────────────────────────────────────────────────────
step 7 ${TOTAL} "Smoke test (5 steps, ~5 min)"
if [ "${SKIP_SMOKE}" = "1" ]; then
    warn "Skipping smoke test (--skip-smoke)."
elif [ -z "${DATASET_PATH:-}" ]; then
    warn "DATASET_PATH not set — skipping smoke test (need a real .jsonl to run training)."
    cat <<EOF
  To run smoke test manually after preparing your dataset:
    bash docker/run_smoke.sh /path/to/your/sft.jsonl
EOF
elif [ ! -f "${DATASET_PATH}" ]; then
    err "DATASET_PATH=${DATASET_PATH} does not exist."
    exit_with_summary 1
elif [ ! -d "${MODEL_LOCAL_DIR}" ]; then
    warn "Model dir ${MODEL_LOCAL_DIR} not found — skipping smoke test."
else
    ok "Use sft_v5.sh for the 5-step smoke test (it auto-pulls business code):"
    echo "    DATA_HOST_PATH='${DATASET_PATH}' \\"
    echo "    MODEL_HOST_DIR='${MODEL_LOCAL_DIR}' \\"
    echo "    bash scripts/gemma4_E4B_opt/sft_v5.sh smoke"
fi

exit_with_summary 0
