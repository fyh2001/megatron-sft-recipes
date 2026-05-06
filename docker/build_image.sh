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
KIND="${KIND:-runtime}"          # runtime = 仅含依赖的镜像；业务代码靠 git clone
DATE="${DATE:-$(date +%y%m%d)}"
BASE_VERSIONS="${BASE_VERSIONS:-u22.04-cu12.9.1-py3.12-t2.10.0-v0.19.0-m1.35.4-s4.1.2}"
RELEASE="${RELEASE:-r1}"          # r1 = runtime release 1；只在依赖升级时 bump

BASE_IMAGE="${BASE_IMAGE:-modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.0-modelscope1.35.4-swift4.1.2}"

TAG="${ACCOUNT}/${NAME}:${KIND}-${DATE}-${BASE_VERSIONS}-${RELEASE}"

echo "=================================================="
echo "Building image:    ${TAG}"
echo "  base image:      ${BASE_IMAGE}"
echo "  Dockerfile:      docker/Dockerfile"
echo "  context:         ${REPO_ROOT}"
echo "  extra args:      $*"
echo "=================================================="

# Sanity check: required input files
test -f docker/Dockerfile      || { echo "ERROR: docker/Dockerfile not found"      >&2; exit 1; }
test -f docker/entrypoint.sh   || { echo "ERROR: docker/entrypoint.sh not found"   >&2; exit 1; }
# Note: business code (sitecustomize.py / patches / scripts) NOT required at
# build time — they are pulled by the entrypoint at container startup from
# CODE_REPO @ CODE_REF (see Dockerfile + docker/entrypoint.sh).

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
echo "  1) Smoke test using sft_v5.sh:"
echo "     IMAGE=${TAG} bash scripts/gemma4_E4B_opt/sft_v5.sh smoke"
echo
echo "  2) Push to Docker Hub (requires 'docker login' as ${ACCOUNT}):"
echo "     docker push ${TAG}"
echo
echo "  3) Or save to tar (no registry needed):"
echo "     docker save ${TAG} | gzip > ${NAME}-${KIND}-${DATE}-${RELEASE}.tar.gz"
echo "=================================================="
