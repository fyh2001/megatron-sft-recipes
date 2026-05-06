# Gemma-4-E4B-it text-only SFT runtime image (依赖层).
#
# 设计哲学：镜像只装"依赖"，业务代码走 git pull。
#
#   依赖（在镜像里）:                  业务代码（容器启动时 git clone）:
#     PyTorch 2.10                      sitecustomize.py (运行时 monkey-patches)
#     CUDA 12.9                         swift gemma.py buffer-fix patch
#     ms-swift 4.2.0.dev0 (editable)    sft_fsdp.sh / bench_variant.sh 等启动脚本
#     transformers / accelerate         docs/
#     modelscope CLI                    plot_compare_*.py 等分析工具
#     git, patch (拉代码工具)
#     entrypoint.sh
#
# 容器启动时 entrypoint.sh 会:
#   1. 按 CODE_REPO / CODE_REF 拉代码到 /workspace/code
#   2. apply swift gemma.py buffer-fix patch（幂等）
#   3. 设 PYTHONPATH 指向 sitecustomize 目录
#   4. exec 用户命令
#
# 这样：依赖升级 → bump 镜像 tag (r1 → r2)；业务代码改 → git push + 改
# CODE_REF 即可，不需要 rebuild 镜像。
#
# Build:
#   bash docker/build_image.sh
ARG BASE=modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.0-modelscope1.35.4-swift4.1.2
FROM ${BASE}

# 1) Install ms-swift 4.2.0.dev0 as editable install at /opt/ms-swift,
# pinned to a specific commit. The base image ships ms-swift 4.1.2 in
# site-packages; editable install at /opt/ms-swift takes precedence.
ARG SWIFT_REPO_URL=https://github.com/fyh2001/ms-swift.git
ARG SWIFT_REPO_COMMIT=b462182dacb5a6c8369b6dd846fafdb12e9c1cd0
RUN apt-get update \
 && apt-get install -y --no-install-recommends git patch ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && git clone "${SWIFT_REPO_URL}" /opt/ms-swift \
 && cd /opt/ms-swift \
 && git checkout "${SWIFT_REPO_COMMIT}" \
 && pip install --no-deps -e . \
 && echo "ms-swift 4.2.0.dev0 editable install at /opt/ms-swift @ ${SWIFT_REPO_COMMIT}"

# Note: swift gemma.py buffer-fix patch is NOT applied at build time.
# entrypoint.sh applies it at runtime, sourced from the cloned business code.
# This way changes to the patch don't require rebuilding the image.

# 2) Copy entrypoint script (the only "code" baked into the image)
COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

# 3) Workspace dir (where business code gets cloned at runtime)
RUN mkdir -p /workspace/code /workspace/output /workspace/logs
WORKDIR /workspace

# 4) Default env (entrypoint reads these; user overrides at runtime via -e)
ENV CODE_REPO=https://github.com/fyh2001/megatron-sft-recipes.git \
    CODE_REF=v1.0 \
    CODE_DIR=/workspace/code \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["/bin/bash"]
