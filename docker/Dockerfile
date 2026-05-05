# Gemma-4-E4B-it text-only SFT training image (v5 production stack).
#
# Self-contained runtime: PyTorch 2.10 + CUDA 12.9 + ms-swift 4.2.0.dev0 +
# all required monkey-patches (sitecustomize.py) + Gemma4Loader buffer fix.
# No model weights or training data baked in — mount via -v at runtime.
#
# Build:
#   bash docker/build_image.sh
# Run quickstart:
#   see docker/README.md
ARG BASE=modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.0-modelscope1.35.4-swift4.1.2
FROM ${BASE}

# 1) Install ms-swift 4.2.0.dev0 as an editable install at /opt/ms-swift,
# pinned to the fork commit that contains the Gemma4 SDPA fallback fix
# (b462182d).  The base image ships ms-swift 4.1.2 in site-packages;
# editable install at /opt/ms-swift takes precedence and is what every
# script and patch file in this image expects.
ARG SWIFT_REPO_URL=https://github.com/fyh2001/ms-swift.git
ARG SWIFT_REPO_COMMIT=b462182dacb5a6c8369b6dd846fafdb12e9c1cd0
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && git clone "${SWIFT_REPO_URL}" /opt/ms-swift \
 && cd /opt/ms-swift \
 && git checkout "${SWIFT_REPO_COMMIT}" \
 && pip install --no-deps -e . \
 && pip show ms-swift | grep -E 'Version|Editable' \
 && echo "ms-swift 4.2.0.dev0 editable install at /opt/ms-swift @ ${SWIFT_REPO_COMMIT}"

# 2) Apply the Gemma4Loader buffer fix-up.
# Without this, transformers 5.x re-initialises layer_scalar / std_scale /
# std_bias to torch.ones(1), causing loss to explode to ~27-38 at step 1
# (verified bug in §15 of docs/gemma4_E4B_alt_offload_log.md).
COPY docker/patches/gemma4_loader_buffer_fix.patch /tmp/gemma4_loader_buffer_fix.patch
RUN cd /opt/ms-swift \
 && git apply --check /tmp/gemma4_loader_buffer_fix.patch \
 && git apply /tmp/gemma4_loader_buffer_fix.patch \
 && rm /tmp/gemma4_loader_buffer_fix.patch \
 && echo "Gemma4Loader buffer fix applied to /opt/ms-swift"

# 3) Copy the v5 training stack into /opt/megatron-sft-recipes/.
WORKDIR /opt/megatron-sft-recipes

# Runtime monkey-patches (loaded via PYTHONPATH at swift sft startup time)
COPY scripts/gemma4_opt/_sdp_preamble/ scripts/gemma4_opt/_sdp_preamble/

# Production launch script + bench harness + alternative variants
COPY scripts/gemma4_E4B_opt/ scripts/gemma4_E4B_opt/

# Metrics / report helpers used by the launch scripts
COPY scripts/benchmark/gpu_monitor.py     scripts/benchmark/gpu_monitor.py
COPY scripts/benchmark/dcgm_scrape.py     scripts/benchmark/dcgm_scrape.py
COPY scripts/benchmark/report_swift_sp.py scripts/benchmark/report_swift_sp.py
COPY scripts/benchmark/extract_loss_curve.py scripts/benchmark/extract_loss_curve.py

# Reference docs (full debugging timeline) + image-specific README
COPY docs/gemma4_E4B_alt_offload_log.md docs/gemma4_E4B_alt_offload_log.md
COPY docker/README.md README.md

# 4) Default env so the launch script picks the in-image paths.
# User overrides MODEL / DATASET_PATH / OUT_ROOT at docker run time.
ENV REPO=/opt/megatron-sft-recipes
ENV DOCKER_EXEC_WRAP="bash -lc"
ENV GPU_MONITOR=1 \
    DCGM_SCRAPE=0 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/megatron-sft-recipes
CMD ["/bin/bash"]
