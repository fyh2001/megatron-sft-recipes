#!/bin/bash
# Per-rank nsys wrapper. torchrun calls this with --no-python so it acts
# as a fork target; LOCAL_RANK is set by torchrun for each worker.
set -e
RANK="${LOCAL_RANK:-0}"
OUT_DIR="${NSYS_OUT_DIR:-/tmp/nsys_out}"
mkdir -p "${OUT_DIR}"

# Only profile rank 0 to avoid 8x trace overhead; that's the typical pattern
# (per-rank GEMM is symmetric for FSDP2 SP=2).
if [ "${RANK}" = "0" ]; then
    exec /usr/local/bin/nsys profile \
        --trace=cuda,nvtx,cublas,cudnn \
        --cuda-memory-usage=false --sample=none \
        --delay=80 --duration=10 \
        --output="${OUT_DIR}/rank0.nsys-rep" \
        --force-overwrite=true \
        /usr/local/bin/python "$@"
else
    exec /usr/local/bin/python "$@"
fi
