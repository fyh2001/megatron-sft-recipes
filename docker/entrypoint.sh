#!/usr/bin/env bash
# Container entrypoint for fangyaohua/gemma4-e4b-it-sft.
#
# 镜像只装依赖（PyTorch / ms-swift / CUDA），业务代码（sitecustomize.py、
# swift gemma.py patch、启动脚本）在容器启动时从 git 拉取，由 CODE_REPO /
# CODE_REF 控制版本。这样镜像不变，业务代码可独立迭代。
#
# 行为（按优先级）：
#   1) 如果 /workspace/code 已 mount（开发者模式）→ 用挂载版本，跳过 git
#   2) 否则按 CODE_REPO / CODE_REF git clone 到 /workspace/code
#   3) Apply swift gemma.py buffer-fix patch（幂等：已 apply 跳过）
#   4) 设置 PYTHONPATH 指向 sitecustomize 目录
#   5) exec 用户命令
#
# 用户传给 docker run 的最后一个 arg = 训练命令（通常是 sft_v5.sh）

set -euo pipefail

CODE_REPO="${CODE_REPO:-https://github.com/fyh2001/megatron-sft-recipes.git}"
CODE_REF="${CODE_REF:-v1.0}"
CODE_DIR="${CODE_DIR:-/workspace/code}"

log() { echo "[entrypoint] $*"; }

# ────────────────────────────────────────────────
# 1. Resolve code source
# ────────────────────────────────────────────────
HAS_MOUNTED_CODE=0
if [ -d "${CODE_DIR}/scripts/gemma4_opt/_sdp_preamble" ] \
   && [ -f "${CODE_DIR}/scripts/gemma4_opt/_sdp_preamble/sitecustomize.py" ]; then
    HAS_MOUNTED_CODE=1
    log "using mounted code at ${CODE_DIR} (skip git clone)"
else
    log "cloning code: ${CODE_REPO} @ ${CODE_REF} → ${CODE_DIR}"
    rm -rf "${CODE_DIR}"
    if ! git clone --depth 1 -b "${CODE_REF}" "${CODE_REPO}" "${CODE_DIR}" 2>&1; then
        log "ERROR: git clone failed"
        log "  CODE_REPO=${CODE_REPO}"
        log "  CODE_REF=${CODE_REF}"
        log "  Possible causes:"
        log "    - tag/branch '${CODE_REF}' doesn't exist on this remote"
        log "    - private repo without credentials (set GIT_ASKPASS or use https://USER:TOKEN@host/repo)"
        log "    - container has no network (try docker run --network=host)"
        exit 1
    fi
    cd "${CODE_DIR}"
    log "  resolved commit: $(git rev-parse HEAD)"
    log "  ref: ${CODE_REF}"
    cd - >/dev/null
fi

# ────────────────────────────────────────────────
# 2. Apply ms-swift Gemma4Loader buffer-fix patch (idempotent)
# ────────────────────────────────────────────────
PATCH_FILE="${CODE_DIR}/docker/patches/gemma4_loader_buffer_fix.patch"
MARKER="layer_scalar / std_scale / std_bias"  # 出现在 patched gemma.py 注释里

if [ ! -f "${PATCH_FILE}" ]; then
    log "WARN: patch file not found at ${PATCH_FILE} — skipping"
elif grep -q "${MARKER}" /opt/ms-swift/swift/model/models/gemma.py 2>/dev/null; then
    log "swift gemma.py buffer fix already applied — skip"
else
    log "applying swift gemma.py buffer fix-up patch"
    cd /opt/ms-swift
    if git apply --check "${PATCH_FILE}" 2>/dev/null; then
        git apply "${PATCH_FILE}"
        log "  done"
    else
        log "  WARN: patch dry-run failed — possibly already partially applied or ms-swift commit changed"
        log "       (training will likely fail with loss>5; fix CODE_REF or the patch file)"
    fi
    cd - >/dev/null
fi

# ────────────────────────────────────────────────
# 3. Configure PYTHONPATH for sitecustomize monkey-patches
# ────────────────────────────────────────────────
SITECUSTOM_DIR="${CODE_DIR}/scripts/gemma4_opt/_sdp_preamble"
if [ -f "${SITECUSTOM_DIR}/sitecustomize.py" ]; then
    export PYTHONPATH="${SITECUSTOM_DIR}:${PYTHONPATH:-}"
    log "PYTHONPATH += ${SITECUSTOM_DIR}"
else
    log "WARN: sitecustomize.py not found at ${SITECUSTOM_DIR}"
fi

# ────────────────────────────────────────────────
# 4. exec user command
# ────────────────────────────────────────────────
if [ "$#" -eq 0 ]; then
    log "no command given, dropping into shell"
    exec /bin/bash
fi

log "exec: $*"
exec "$@"
