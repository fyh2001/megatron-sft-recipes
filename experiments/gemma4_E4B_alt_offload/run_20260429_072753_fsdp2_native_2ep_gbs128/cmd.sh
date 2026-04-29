#!/usr/bin/env bash
# E4B-it text-only SFT — FSDP2 native (no offload) — full 2-epoch training
#
# Origin: 用户基于 P1 alt-offload (DS-equivalent) 与同事 DS ZeRO-3 native 配置
# 比较后选择的 D 路径 + 4 处自定义修改。引擎用 FSDP2 native（不开 offload，
# 因为 E4B 只有 8B 参数，单 H100 80G 显存绰绰有余），GBS / lr / warmup / epochs
# 全部按用户要求落定。
#
# Effective config (real, not estimated):
#   * Engine            : FSDP2  (full_shard auto_wrap, NO offload, AC=on)
#   * NPROC_PER_NODE    : 8
#   * SP                : 1            (no sequence parallel; matches D)
#   * MBS               : 1
#   * GAS               : 16           (用户改：48 → 16)
#   * DP                : 8            (NPROC / SP)
#   * GBS               : 128          (= MBS × DP × GAS = 1 × 8 × 16)
#   * num_train_epochs  : 2            (用户改：1 → 2)
#   * learning_rate     : 2e-5         (用户改：1e-5 → 2e-5)
#   * warmup_ratio      : 0.05         (用户改：0.1/0.025 → 0.05)
#   * save_strategy     : epoch        (用户改：steps/no → epoch)
#   * max_length        : 16384
#   * truncation        : right
#   * dtype             : bfloat16
#   * attn_impl         : flash_attention_2  (modeling_gemma4 patch falls back
#                                              to mem_efficient SDPA on global
#                                              head_dim=512 layers)
#   * Liger             : true
#   * freeze_vit/aligner: true (model.vision_tower + model.audio_tower +
#                                model.embed_vision + model.embed_audio frozen)
#
# Steps total (实测样本 51 557):
#   per epoch ≈ ceil(51 557 / 128) = 403   →  2 epoch ≈ 806 optimizer steps
#
# Required preamble (already installed in container fsdp_sft):
#   * sitecustomize.py @ scripts/gemma4_opt/_sdp_preamble (mem_efficient SDPA pin)
#   * GEMMA4_FORCE_MEM_EFFICIENT_SDP=1
#   * modeling_gemma4.py patch (md5 39ebf386a992fea9eac0883f459ac658)
#
# Output dir: experiments/gemma4_E4B_alt_offload/run_<TS>_fsdp2_native_2ep_gbs128/

set -euo pipefail

REPO=/home/ubuntu/fyh/megatron-sft-recipes
SCRIPT_DIR=${REPO}/scripts/gemma4_E4B_opt

OUT_ROOT=${REPO}/experiments/gemma4_E4B_alt_offload
mkdir -p "${OUT_ROOT}"

LABEL="${LABEL:-fsdp2_native_2ep_gbs128}"
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUT_ROOT}/run_${TS}_${LABEL}"
mkdir -p "${RUN_DIR}"
cp "$0" "${RUN_DIR}/cmd.sh"

MODEL=/home/ubuntu/.cache/modelscope/models/google/gemma-4-E4B-it
DATASET_PATH=/home/ubuntu/fyh/megatron-sft-recipes/sft-data/SFT_0424_2.jsonl
DATASET_SIZE=51557

NPROC=8
SP=1
MBS=1
GAS=16
GBS=$(( MBS * NPROC * GAS / SP ))   # 128
NUM_EPOCHS=2
LR=2e-5
WARMUP_RATIO=0.05
MAX_LEN=16384
WARMUP_BENCH_FOR_REPORT=5  # 用于 report_swift_sp.py 跳过前 5 步算稳态指标

NUM_PARAMS=7.518e9   # text trainable: language_model + tied lm_head
NUM_ACTIVE_PARAMS=4.62e9  # compute-only (PLE 2.82B 是 lookup 不算 GEMM)

MASTER_PORT=$(( 29500 + (RANDOM % 200) ))

echo "=========================================="
echo "FSDP2 native — Gemma-4-E4B-it text-only 2-epoch"
echo "Run dir   : ${RUN_DIR}"
echo "Dataset   : ${DATASET_PATH} (${DATASET_SIZE} samples)"
echo "Engine    : FSDP2 (no offload), AC=on"
echo "MBS / GAS / SP / DP / GBS : ${MBS} / ${GAS} / ${SP} / $((NPROC/SP)) / ${GBS}"
echo "Epochs / LR / Warmup     : ${NUM_EPOCHS} / ${LR} / ${WARMUP_RATIO}"
echo "Steps (computed)         : per_epoch ≈ ceil(${DATASET_SIZE} / ${GBS}) ≈ $(( (DATASET_SIZE + GBS - 1) / GBS )) ; total ≈ $(( ((DATASET_SIZE + GBS - 1) / GBS) * NUM_EPOCHS ))"
echo "Port      : ${MASTER_PORT}"
echo "=========================================="

# Render fsdp_override.json — FSDP2 native (no offload)
FSDP_OVERRIDE="${RUN_DIR}/fsdp_override.json"
cat > "${FSDP_OVERRIDE}" <<'EOF'
{
    "fsdp": "full_shard auto_wrap",
    "fsdp_config": {
        "fsdp_version": 2,
        "reshard_after_forward": true,
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "cpu_ram_efficient_loading": false,
        "state_dict_type": "SHARDED_STATE_DICT",
        "activation_checkpointing": true,
        "transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer", "Gemma4AudioLayer"]
    }
}
EOF
echo "FSDP override written:"
cat "${FSDP_OVERRIDE}"

# Background monitors (running on host, not in container)
GPU_LOG="${RUN_DIR}/gpu_metrics.jsonl"
DCGM_LOG="${RUN_DIR}/dcgm_tc.tsv"
TRAIN_LOG="${RUN_DIR}/train.log"
STATUS_FILE="${RUN_DIR}/STATUS"

python3 "${REPO}/scripts/benchmark/gpu_monitor.py" --output "${GPU_LOG}" &
GPU_MON_PID=$!
python3 "${REPO}/scripts/benchmark/dcgm_scrape.py" "${DCGM_LOG}" "http://localhost:9500/metrics" &
DCGM_PID=$!
trap 'kill ${GPU_MON_PID} ${DCGM_PID} 2>/dev/null || true' EXIT

set +e
docker exec fsdp_sft bash -lc "
cd /home/ubuntu/fyh/megatron-sft-recipes && \
PYTHONPATH=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_opt/_sdp_preamble:\${PYTHONPATH:-} \
GEMMA4_FORCE_MEM_EFFICIENT_SDP=1 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_DEVICE_MAX_CONNECTIONS=8 \
NPROC_PER_NODE=${NPROC} MASTER_PORT=${MASTER_PORT} \
swift sft \
    --model ${MODEL} \
    --model_type gemma4 \
    --template gemma4_nothinking \
    --dataset ${DATASET_PATH} \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --max_length ${MAX_LEN} \
    --truncation_strategy right \
    --per_device_train_batch_size ${MBS} \
    --gradient_accumulation_steps ${GAS} \
    --num_train_epochs ${NUM_EPOCHS} \
    --learning_rate ${LR} \
    --lr_scheduler_type cosine \
    --warmup_ratio ${WARMUP_RATIO} \
    --weight_decay 0.0 \
    --use_liger_kernel true \
    --gradient_checkpointing true \
    --freeze_vit true --freeze_aligner true \
    --fsdp ${FSDP_OVERRIDE} \
    --sequence_parallel_size ${SP} \
    --packing false \
    --padding_free false \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --logging_steps 1 \
    --report_to tensorboard \
    --save_strategy epoch \
    --save_total_limit 3 \
    --save_only_model true \
    --load_from_cache_file false \
    --split_dataset_ratio 0 \
    --output_dir ${RUN_DIR} \
    --logging_dir ${RUN_DIR}/runs \
" 2>&1 | tee "${TRAIN_LOG}"
EXIT=${PIPESTATUS[0]}
set -e

kill "${GPU_MON_PID}" "${DCGM_PID}" 2>/dev/null || true
wait "${GPU_MON_PID}" 2>/dev/null || true
wait "${DCGM_PID}" 2>/dev/null || true

# Resolve swift's auto-named v*-* sub-dir which holds logging.jsonl
VDIR=$(ls -dt ${RUN_DIR}/v*-* 2>/dev/null | head -n 1 || true)
if [ -n "${VDIR}" ] && [ -f "${VDIR}/logging.jsonl" ]; then
    ln -sf "${VDIR}/logging.jsonl" "${RUN_DIR}/logging.jsonl"
fi

# Categorise outcome
if [ "${EXIT}" = "0" ]; then
    STATUS="SUCCESS"
    SUMMARY="see report.json"
else
    if grep -q "CUDA out of memory" "${TRAIN_LOG}" 2>/dev/null; then
        STATUS="FAILED"
        SUMMARY="OOM"
    else
        STATUS="FAILED"
        SUMMARY="exit=${EXIT}"
    fi
fi
echo "${STATUS} — ${SUMMARY}" > "${STATUS_FILE}"

# Report aggregation (steady-state metrics, skipping first 5 warmup steps)
if [ -f "${RUN_DIR}/logging.jsonl" ]; then
    python3 "${REPO}/scripts/benchmark/report_swift_sp.py" \
        --logging_jsonl "${RUN_DIR}/logging.jsonl" \
        --gpu_log "${GPU_LOG}" \
        --warmup_steps "${WARMUP_BENCH_FOR_REPORT}" \
        --num_gpus "${NPROC}" \
        --gbs "${GBS}" \
        --grad_accum "${GAS}" \
        --max_len "${MAX_LEN}" \
        --backend fsdp2 \
        --sp "${SP}" \
        --num_params "${NUM_PARAMS}" \
        --num_active_params "${NUM_ACTIVE_PARAMS}" \
        --dataset_size "${DATASET_SIZE}" \
        --bench_jsonl_out "${RUN_DIR}/bench.jsonl" \
        --report_out "${RUN_DIR}/report.json" || true

    python3 "${REPO}/scripts/benchmark/extract_loss_curve.py" \
        "${RUN_DIR}/logging.jsonl" "${RUN_DIR}" || true
fi

# Print final summary
if [ -f "${RUN_DIR}/report.json" ]; then
    echo "=========================================="
    echo "Final summary"
    echo "=========================================="
    python3 -c "
import json
d = json.load(open('${RUN_DIR}/report.json'))
keys = ['mean_step_time_ms','median_step_time_ms','p99_step_time_ms','micro_step_time_ms',
        'tokens_per_sec_per_gpu','achieved_tflops_per_gpu','achieved_tflops_per_gpu_active',
        'mfu_pct','mfu_pct_active_params',
        'peak_mem_gb','peak_mem_gib_from_swift_log',
        'avg_gpu_util_pct','avg_power_w','peak_power_w',
        'actual_total_wall_min','steps_per_epoch_for_dataset',
        'loss_first_step','loss_last_step']
for k in keys:
    print(f'  {k:32s} = {d.get(k)}')
"
fi

# Cleanup any straggler workers
docker exec fsdp_sft pkill -9 python 2>/dev/null || true

exit ${EXIT}
