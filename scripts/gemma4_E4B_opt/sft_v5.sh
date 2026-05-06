#!/usr/bin/env bash
# Gemma-4-E4B-it text-only SFT — v5 launcher（FSDP2 + DS3-equivalent）
#
# 单文件 launcher，host / container 双模式 + 6 个 subcommand：
#
#   bash sft_v5.sh                # 默认：full 2-epoch 训练
#   bash sft_v5.sh train          # 同上
#   bash sft_v5.sh smoke          # 跑 5 步快速验证（~5 分钟）
#   bash sft_v5.sh status         # 查看最近一次 run 的容器状态
#   bash sft_v5.sh logs           # 实时跟随最近一次 run 的日志
#   bash sft_v5.sh stop           # 停掉最近一次启动的容器
#   bash sft_v5.sh --dry-run      # 打印 docker run 命令但不执行
#   bash sft_v5.sh --help         # 详细帮助
#
# 编辑下面 [USER CONFIG] 段然后直接跑。镜像里 entrypoint 自动拉业务代码，
# host 上除 docker 外没有任何依赖。

set -euo pipefail

# ============================================================
# [USER CONFIG] —— 必填三项 + 高级 + 内部，按需改
# ============================================================

# ---------- 必填（90% 用户只动这几个）----------

IMAGE="${IMAGE:-fangyaohua/gemma4-e4b-it-sft:runtime-260506-u22.04-cu12.9.1-py3.12-t2.10.0-v0.19.0-m1.35.4-s4.1.2-r1}"
MODEL_HOST_DIR="${MODEL_HOST_DIR:-${HOME}/.cache/modelscope/models/google/gemma-4-E4B-it}"
DATA_HOST_PATH="${DATA_HOST_PATH:-$(pwd)/sft.jsonl}"
OUTPUT_HOST_DIR="${OUTPUT_HOST_DIR:-$(pwd)/runs}"

# ---------- 高级（你知道在干嘛再动）----------

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES_VAL="${CUDA_VISIBLE_DEVICES_VAL:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29501}"

NUM_EPOCHS="${NUM_EPOCHS:-2}"
LR="${LR:-2e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"        # GBS = NPROC × MBS × GAS = 128
MAX_LEN="${MAX_LEN:-16384}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"           # epoch / steps / no
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"

SHM_SIZE="${SHM_SIZE:-16g}"
IPC_MODE="${IPC_MODE:-host}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-true}"

# ---------- 业务代码版本（默认拉 v1.0 tag）----------

CODE_REPO="${CODE_REPO:-https://github.com/fyh2001/megatron-sft-recipes.git}"
CODE_REF="${CODE_REF:-v1.0}"

# ---------- v5 内部（强烈不建议改）----------

TORCH_DTYPE="${TORCH_DTYPE:-float32}"             # fp32 master
USE_BF16_MP="${USE_BF16_MP:-true}"                # bf16 mixed precision compute
PADDING_FREE="${PADDING_FREE:-false}"             # Gemma-4 必须 false (§22)
USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-true}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-true}"
ACTIVATION_CPU_OFFLOAD="${ACTIVATION_CPU_OFFLOAD:-true}"  # patch 21b 需要
GEMMA4_FSDP_WRAP_PLE="${GEMMA4_FSDP_WRAP_PLE:-1}"
GEMMA4_KV_SHARE_DETACH="${GEMMA4_KV_SHARE_DETACH:-1}"
GEMMA4_FSDP_REDUCE_FP32_NCCL="${GEMMA4_FSDP_REDUCE_FP32_NCCL:-1}"

# ============================================================
# [INTERNAL] —— 下面是逻辑，一般不用改
# ============================================================

DOCKER_BIN="${DOCKER_BIN:-docker}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
THIS_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
HOST_SCRIPT_DIR="$(dirname "${THIS_SCRIPT}")"
META_DIR="${OUTPUT_HOST_DIR}/.meta"
META_FILE="${META_DIR}/latest_run.txt"

log()   { echo "[$1] ${@:2}"; }
info()  { log info "$@"; }
warn()  { log warn "$@" >&2; }
err()   { log error "$@" >&2; }
die()   { err "$@"; exit 1; }

usage() {
    cat <<'EOF'
Gemma-4-E4B-it SFT v5 launcher

用法:
  bash sft_v5.sh [SUBCOMMAND] [FLAGS]

Subcommand:
  train         (默认) full 训练（按 NUM_EPOCHS 配置，~9h on 8x H100）
  smoke         5-step smoke test（~5 min），验证 step 1 loss=2.226 / gn=10.28
  status        显示最近一次 run 的容器状态 + 输出目录
  logs          docker logs -f 跟随最近一次 run
  stop          docker stop 最近一次 run
  --dry-run     仅打印 docker run 命令，不执行
  --help, -h    本帮助

env override（例）:
  MODEL_HOST_DIR=/path/to/model bash sft_v5.sh
  NUM_EPOCHS=1 LR=1e-5 bash sft_v5.sh
  CODE_REF=main bash sft_v5.sh                # 拉 main 而非 v1.0
  CODE_REPO=https://gitlab.example.com/... bash sft_v5.sh

完整配置见脚本顶部 [USER CONFIG] 段。
EOF
}

read_meta() {
    [ -f "${META_FILE}" ] || return 1
    # shellcheck disable=SC1090
    source "${META_FILE}"
}

# ============================================================
# Subcommand dispatch
# ============================================================

SUBCMD="${1:-train}"
DRY_RUN=0

case "${SUBCMD}" in
    -h|--help|help) usage; exit 0 ;;
    --dry-run)  DRY_RUN=1; SUBCMD="train"; shift || true ;;
    train|smoke) shift || true ;;
    status)
        read_meta || die "no previous run found (META_FILE=${META_FILE})"
        echo "container_name : ${CONTAINER_NAME:-?}"
        echo "container_id   : ${CONTAINER_ID:-?}"
        STATUS=$("${DOCKER_BIN}" inspect --format '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo "missing/removed")
        echo "current status : ${STATUS}"
        echo "output_dir     : ${OUTPUT_DIR:-?}"
        echo "log_file       : ${LOG_FILE:-?}"
        echo "view logs      : docker logs -f ${CONTAINER_NAME}"
        echo "stop           : bash $0 stop"
        if [ -f "${OUTPUT_DIR:-/dev/null}" ] || [ -d "${OUTPUT_DIR:-/dev/null}" ]; then
            recent=$(ls -dt "${OUTPUT_DIR}"/v0-* 2>/dev/null | head -1 || true)
            if [ -n "${recent}" ] && [ -f "${recent}/logging.jsonl" ]; then
                last_step=$(grep -oE '"global_step/max_steps": "[0-9]+/[0-9]+"' "${recent}/logging.jsonl" | tail -1 || true)
                echo "last step      : ${last_step}"
            fi
        fi
        exit 0
        ;;
    logs)
        read_meta || die "no previous run found"
        exec "${DOCKER_BIN}" logs -f "${CONTAINER_NAME}"
        ;;
    stop)
        read_meta || die "no previous run found"
        info "stopping ${CONTAINER_NAME}"
        "${DOCKER_BIN}" stop "${CONTAINER_NAME}"
        exit $?
        ;;
    *)
        if [ "${SUBCMD}" = "${1:-}" ]; then
            err "unknown subcommand: ${SUBCMD}"
            echo
            usage
            exit 2
        fi
        ;;
esac

# ============================================================
# Smoke test config override
# ============================================================
if [ "${SUBCMD}" = "smoke" ]; then
    info "smoke test mode: MAX_STEPS=5, FULL_SCHED_STOP=1, RUN_IN_BACKGROUND=false"
    SMOKE_MAX_STEPS=5
    SMOKE_FULL_SCHED_STOP=1
    RUN_IN_BACKGROUND=false        # smoke 都是前台跑，方便看输出
fi

# ============================================================
# Branch on RUN_CONTEXT: host-side launches docker, container-side runs swift
# ============================================================

if [ "${RUN_CONTEXT:-host}" != "container" ]; then
    # ──────────── HOST MODE ────────────
    info "image       : ${IMAGE}"
    info "code repo   : ${CODE_REPO} @ ${CODE_REF}"
    info "model       : ${MODEL_HOST_DIR}"
    info "dataset     : ${DATA_HOST_PATH}"
    info "output_dir  : ${OUTPUT_HOST_DIR}/sft_v5_${RUN_TS}"
    info "GPUs        : ${NPROC_PER_NODE} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VAL})"
    info "subcommand  : ${SUBCMD}"

    # ── Sanity checks ──
    if ! command -v "${DOCKER_BIN}" >/dev/null 2>&1; then
        die "docker not found; install from https://get.docker.com"
    fi
    if ! "${DOCKER_BIN}" info >/dev/null 2>&1; then
        die "docker daemon not reachable; sudo systemctl start docker"
    fi
    if [ ! -d "${MODEL_HOST_DIR}" ]; then
        die "model dir not found: ${MODEL_HOST_DIR}"
    fi
    if [ ! -f "${DATA_HOST_PATH}" ]; then
        die "dataset file not found: ${DATA_HOST_PATH}"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
        if [ "${gpu_count}" -lt "${NPROC_PER_NODE}" ]; then
            warn "host has ${gpu_count} GPUs but NPROC_PER_NODE=${NPROC_PER_NODE}; OOM likely"
        fi
    fi

    mkdir -p "${OUTPUT_HOST_DIR}" "${META_DIR}"

    RUN_OUTPUT_DIR_HOST="${OUTPUT_HOST_DIR}/sft_v5_${RUN_TS}"
    LOG_FILE_HOST="${RUN_OUTPUT_DIR_HOST}/host.log"
    CONTAINER_NAME="gemma4-e4b-sft-v5-${SUBCMD}-${RUN_TS}"

    # Forward all relevant config to container via -e
    env_args=(
        -e RUN_CONTEXT=container
        -e RUN_TS="${RUN_TS}"
        -e SUBCMD="${SUBCMD}"
        -e CODE_REPO="${CODE_REPO}"
        -e CODE_REF="${CODE_REF}"
        -e NPROC_PER_NODE="${NPROC_PER_NODE}"
        -e CUDA_VISIBLE_DEVICES_VAL="${CUDA_VISIBLE_DEVICES_VAL}"
        -e MASTER_PORT="${MASTER_PORT}"
        -e NUM_EPOCHS="${NUM_EPOCHS}"
        -e LR="${LR}"
        -e WARMUP_RATIO="${WARMUP_RATIO}"
        -e WEIGHT_DECAY="${WEIGHT_DECAY}"
        -e MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}"
        -e GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}"
        -e MAX_LEN="${MAX_LEN}"
        -e SAVE_STRATEGY="${SAVE_STRATEGY}"
        -e SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}"
        -e SEED="${SEED}"
        -e DATA_SEED="${DATA_SEED}"
        -e TORCH_DTYPE="${TORCH_DTYPE}"
        -e USE_BF16_MP="${USE_BF16_MP}"
        -e PADDING_FREE="${PADDING_FREE}"
        -e USE_LIGER_KERNEL="${USE_LIGER_KERNEL}"
        -e ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING}"
        -e ACTIVATION_CPU_OFFLOAD="${ACTIVATION_CPU_OFFLOAD}"
        -e GEMMA4_FSDP_WRAP_PLE="${GEMMA4_FSDP_WRAP_PLE}"
        -e GEMMA4_KV_SHARE_DETACH="${GEMMA4_KV_SHARE_DETACH}"
        -e GEMMA4_FSDP_REDUCE_FP32_NCCL="${GEMMA4_FSDP_REDUCE_FP32_NCCL}"
    )
    if [ "${SUBCMD}" = "smoke" ]; then
        env_args+=(-e SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS}"
                   -e SMOKE_FULL_SCHED_STOP="${SMOKE_FULL_SCHED_STOP}")
    fi

    docker_args=(
        run --rm --init
        --gpus all
        "--shm-size=${SHM_SIZE}"
        "--ipc=${IPC_MODE}"
        --name "${CONTAINER_NAME}"
        -v "${MODEL_HOST_DIR}:/workspace/model:ro"
        -v "${DATA_HOST_PATH}:/workspace/data/sft.jsonl:ro"
        -v "${OUTPUT_HOST_DIR}:/workspace/output"
    )

    # Open up: if user is in a git checkout of the repo, mount it so iteration
    # is faster (entrypoint detects mounted code and skips git clone).
    if [ -d "${HOST_SCRIPT_DIR}/../.." ] \
       && [ -f "${HOST_SCRIPT_DIR}/../../scripts/gemma4_opt/_sdp_preamble/sitecustomize.py" ]; then
        REPO_ROOT="$(cd "${HOST_SCRIPT_DIR}/../.." && pwd)"
        docker_args+=(-v "${REPO_ROOT}:/workspace/code:ro")
        info "developer mode: mounting repo ${REPO_ROOT} → /workspace/code"
    else
        info "stand-alone mode: container will git clone CODE_REPO @ CODE_REF"
    fi

    docker_args+=("${env_args[@]}")

    if [ "${RUN_IN_BACKGROUND}" = true ]; then
        docker_args+=(-d)
    elif [ -t 0 ] && [ -t 1 ]; then
        docker_args+=(-it)
    fi

    docker_args+=(
        "${IMAGE}"
        bash -c "
            mkdir -p /workspace/output/sft_v5_${RUN_TS}
            exec bash /workspace/code/scripts/gemma4_E4B_opt/sft_v5.sh ${SUBCMD} \\
                2>&1 | tee /workspace/output/sft_v5_${RUN_TS}/train.log
        "
    )

    if [ "${DRY_RUN}" = "1" ]; then
        echo
        echo "== docker command (dry run) =="
        printf '%q ' "${DOCKER_BIN}" "${docker_args[@]}"
        echo
        exit 0
    fi

    if [ "${RUN_IN_BACKGROUND}" = true ]; then
        CONTAINER_ID=$("${DOCKER_BIN}" "${docker_args[@]}")
        cat > "${META_FILE}" <<EOF
RUN_TS="${RUN_TS}"
SUBCMD="${SUBCMD}"
CONTAINER_ID="${CONTAINER_ID}"
CONTAINER_NAME="${CONTAINER_NAME}"
OUTPUT_DIR="${RUN_OUTPUT_DIR_HOST}"
LOG_FILE="${LOG_FILE_HOST}"
EOF
        info "started in background: ${CONTAINER_NAME}"
        info "view logs: bash $0 logs    (or: docker logs -f ${CONTAINER_NAME})"
        info "status   : bash $0 status"
        info "stop     : bash $0 stop"
        exit 0
    fi

    exec "${DOCKER_BIN}" "${docker_args[@]}"
fi

# ────────────────────────────────────────────────
# CONTAINER MODE — by entrypoint after git clone + apply patch + set PYTHONPATH
# ────────────────────────────────────────────────

info "container: running v5 SFT, RUN_TS=${RUN_TS}, SUBCMD=${SUBCMD}"

RUN_OUTPUT_DIR="/workspace/output/sft_v5_${RUN_TS}"
mkdir -p "${RUN_OUTPUT_DIR}"

FSDP_OVERRIDE="${RUN_OUTPUT_DIR}/fsdp_override.json"
GBS=$(( MICRO_BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM_STEPS ))

# Write FSDP2 override JSON
cat > "${FSDP_OVERRIDE}" <<EOF
{
    "fsdp": "full_shard auto_wrap offload",
    "fsdp_config": {
        "fsdp_version": 2,
        "reshard_after_forward": true,
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "cpu_ram_efficient_loading": false,
        "state_dict_type": "FULL_STATE_DICT",
        "activation_checkpointing": ${ACTIVATION_CHECKPOINTING},
        "activation_cpu_offload": ${ACTIVATION_CPU_OFFLOAD},
        "transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer", "Gemma4AudioLayer"]
    }
}
EOF

# bf16 mixed precision flag
if [ "${USE_BF16_MP}" = "true" ]; then
    BF16_FLAGS=("--bf16" "true" "--fp16" "false")
else
    BF16_FLAGS=("--bf16" "false" "--fp16" "false")
fi

# Smoke test: stop after N steps via GEMMA4_STOP_AFTER_STEPS env (read by
# sitecustomize patch). Bash's syntactic env-prefix `KEY=VAL cmd` only works
# with literal KEY=VAL, not with variable expansion — so we use `env` to
# forward the variable into swift sft's environment.
SMOKE_PREFIX=()
if [ "${SUBCMD}" = "smoke" ]; then
    SMOKE_PREFIX=(env "GEMMA4_STOP_AFTER_STEPS=${SMOKE_MAX_STEPS:-5}")
fi

info "GBS=${GBS} (MBS=${MICRO_BATCH_SIZE} × NPROC=${NPROC_PER_NODE} × GAS=${GRAD_ACCUM_STEPS})"
info "max_length=${MAX_LEN}, num_epochs=${NUM_EPOCHS}, lr=${LR}"
info "FSDP_OVERRIDE=${FSDP_OVERRIDE}"

# Run swift sft. PYTHONPATH was set by entrypoint.sh.
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_DEVICE_MAX_CONNECTIONS=8 \
FSDP_STATE_DICT_TYPE=FULL_STATE_DICT \
GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
GEMMA4_FSDP_WRAP_PLE="${GEMMA4_FSDP_WRAP_PLE}" \
GEMMA4_KV_SHARE_DETACH="${GEMMA4_KV_SHARE_DETACH}" \
GEMMA4_FSDP_REDUCE_FP32_NCCL="${GEMMA4_FSDP_REDUCE_FP32_NCCL}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}" \
MASTER_PORT="${MASTER_PORT}" \
"${SMOKE_PREFIX[@]}" swift sft \
    --model /workspace/model \
    --model_type gemma4 \
    --template gemma4 \
    --dataset /workspace/data/sft.jsonl \
    --tuner_type full \
    --torch_dtype "${TORCH_DTYPE}" \
    "${BF16_FLAGS[@]}" \
    --attn_impl flash_attention_2 \
    --max_length "${MAX_LEN}" \
    --truncation_strategy right \
    --per_device_train_batch_size "${MICRO_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --learning_rate "${LR}" \
    --lr_scheduler_type cosine \
    --warmup_ratio "${WARMUP_RATIO}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --use_liger_kernel "${USE_LIGER_KERNEL}" \
    --gradient_checkpointing "${ACTIVATION_CHECKPOINTING}" \
    --freeze_vit true --freeze_aligner true \
    --fsdp "${FSDP_OVERRIDE}" \
    --sequence_parallel_size 1 \
    --packing false \
    --padding_free "${PADDING_FREE}" \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --logging_steps 1 \
    --report_to tensorboard \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --save_only_model true \
    --load_from_cache_file false \
    --split_dataset_ratio 0 \
    --output_dir "${RUN_OUTPUT_DIR}" \
    --logging_dir "${RUN_OUTPUT_DIR}/runs" \
    --seed "${SEED}" \
    --data_seed "${DATA_SEED}"

info "done. output_dir=${RUN_OUTPUT_DIR}"
