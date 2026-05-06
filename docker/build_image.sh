#!/usr/bin/env bash
# Build the Gemma-4-E4B-it text-only SFT training image.
#
# Convention (mirroring colleague's zyfncg/kaon-swift-train tag style):
#   <ACCOUNT>/<NAME>:<DATE>-<BASE_VERSIONS>-<RELEASE>
#
# Override any of ACCOUNT / NAME / DATE / RELEASE / BASE via env var.
# Run from repo root:
#   bash docker/build_image.sh
#   bash docker/build_image.sh --no-cache
#
# After successful build the script prints the resulting tag, ready to be
# fed to `docker push` or `docker save`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

ACCOUNT="${ACCOUNT:-fangyaohua}"
NAME="${NAME:-gemma4-e4b-it-sft}"
DATE="${DATE:-260506}"
BASE_VERSIONS="${BASE_VERSIONS:-u22.04-cu12.9.1-py3.12-t2.10.0-v0.19.0-m1.35.4-s4.1.2}"
RELEASE="${RELEASE:-v2}"

BASE_IMAGE="${BASE_IMAGE:-modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.0-modelscope1.35.4-swift4.1.2}"

TAG="${ACCOUNT}/${NAME}:${DATE}-${BASE_VERSIONS}-${RELEASE}"

echo "=================================================="
echo "Building image:    ${TAG}"
echo "  base image:      ${BASE_IMAGE}"
echo "  Dockerfile:      docker/Dockerfile"
echo "  context:         ${REPO_ROOT}"
echo "  extra args:      $*"
echo "=================================================="

# Sanity check: required input files
test -f docker/patches/gemma4_loader_buffer_fix.patch || {
    echo "ERROR: docker/patches/gemma4_loader_buffer_fix.patch not found." >&2
    echo "       Re-export with:" >&2
    echo "       docker exec fsdp_sft bash -c 'cd /opt/ms-swift && git diff swift/model/models/gemma.py' \\" >&2
    echo "         > docker/patches/gemma4_loader_buffer_fix.patch" >&2
    exit 1
}
test -f docker/README.md || {
    echo "ERROR: docker/README.md not found (image needs an embedded README)." >&2
    exit 1
}
test -f scripts/gemma4_opt/_sdp_preamble/sitecustomize.py || {
    echo "ERROR: scripts/gemma4_opt/_sdp_preamble/sitecustomize.py not found." >&2
    exit 1
}

docker build \
    --build-arg "BASE=${BASE_IMAGE}" \
    -f docker/Dockerfile \
    -t "${TAG}" \
    "$@" \
    .

echo
echo "=================================================="
echo "Build OK."
echo "Tag:         ${TAG}"
echo
echo "Local size:"
docker images --format '  {{.Repository}}:{{.Tag}}  {{.Size}}' "${ACCOUNT}/${NAME}" | head -3
echo
echo "Next steps:"
echo "  1) Smoke test (5-step bench, requires GPUs + model + data mounts):"
echo "     docker run --rm --gpus all --shm-size=16g \\"
echo "       -v \$HOME/.cache/modelscope:/root/.cache/modelscope \\"
echo "       -v \$(pwd)/sft-data:/data \\"
echo "       -v \$(pwd)/runs:/runs \\"
echo "       -e MODEL=/root/.cache/modelscope/models/google/gemma-4-E4B-it \\"
echo "       -e DATASET_PATH=/data/SFT_0424_2.jsonl \\"
echo "       -e OUT_ROOT=/runs/bench \\"
echo "       -e LABEL=smoke -e MAX_STEPS=5 -e FULL_SCHED_STOP=1 \\"
echo "       -e TORCH_DTYPE=float32 \\"
echo "       -e EXTRA_ENV='GEMMA4_FSDP_WRAP_PLE=1 GEMMA4_KV_SHARE_DETACH=1 GEMMA4_FSDP_REDUCE_FP32_NCCL=1' \\"
echo "       -e EXTRA_ARGS='--bf16 true --fp16 false --padding_free false --max_grad_norm 1.0' \\"
echo "       ${TAG} \\"
echo "       bash /opt/megatron-sft-recipes/scripts/gemma4_E4B_opt/bench_variant.sh"
echo
echo "  2) Push to Docker Hub (requires 'docker login' as ${ACCOUNT}):"
echo "     docker push ${TAG}"
echo
echo "  3) Or save to tar (no registry needed):"
echo "     docker save ${TAG} | gzip > ${NAME}-${DATE}-${RELEASE}.tar.gz"
echo "=================================================="
