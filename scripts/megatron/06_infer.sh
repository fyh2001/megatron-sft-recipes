#!/usr/bin/env bash
# 06_infer.sh
#   用 swift infer 验证 SFT 训练后的模型。
#   ms-swift 保存时已经是 safetensors，可直接被 transformers / vLLM 加载。
#
# 用法示例：
#   # 自动找 qwen25_7b 下最新的 checkpoint
#   bash /home/ubuntu/perf_opt/scripts/06_infer.sh
#
#   # 指定某个 checkpoint
#   CKPT_DIR=/home/ubuntu/perf_opt/megatron_output/qwen25_7b/v0-.../checkpoint-500 \
#       bash /home/ubuntu/perf_opt/scripts/06_infer.sh
#
#   # 验证 Gemma 4
#   CKPT_DIR=/home/ubuntu/perf_opt/megatron_output/gemma4_e4b/v0-.../checkpoint-500 \
#       PROMPT='Explain entropy in one sentence' \
#       bash /home/ubuntu/perf_opt/scripts/06_infer.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/../_common.sh"

: "${OUTPUT_SUB:=qwen25_7b}"
: "${PROMPT:=你好，请用一句话介绍你自己。}"
: "${MAX_NEW_TOKENS:=512}"
: "${TEMPERATURE:=0.7}"

if [ -z "${CKPT_DIR:-}" ]; then
    # 自动找最新的 checkpoint
    BASE="${OUTPUT_ROOT}/${OUTPUT_SUB}"
    LATEST_RUN="$(ls -td "${BASE}"/v*/ 2>/dev/null | head -1 || true)"
    if [ -z "${LATEST_RUN}" ]; then
        echo "找不到训练输出：${BASE}/v*/，请显式设置 CKPT_DIR" >&2
        exit 1
    fi
    LATEST_CKPT="$(ls -d "${LATEST_RUN}"checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
    if [ -z "${LATEST_CKPT}" ]; then
        echo "checkpoint-* 目录不存在于 ${LATEST_RUN}" >&2
        exit 1
    fi
    CKPT_DIR="${LATEST_CKPT}"
fi

log "inference"
printf '  %-18s = %s\n' \
    checkpoint "${CKPT_DIR}" \
    prompt "${PROMPT}" \
    max_new_tokens "${MAX_NEW_TOKENS}" \
    temperature "${TEMPERATURE}"

# swift infer 的交互式模式从 stdin 读取 query，用 echo 喂一个 prompt 即可。
# 想要交互式对话：直接在容器里跑
#     swift infer --model <CKPT_DIR> --stream true
echo "${PROMPT}" | \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
swift infer \
    --model "${CKPT_DIR}" \
    --stream true \
    --temperature "${TEMPERATURE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"
