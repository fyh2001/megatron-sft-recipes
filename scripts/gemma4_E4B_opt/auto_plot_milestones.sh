#!/usr/bin/env bash
# Watch the current 2-ep FSDP2 run; generate a comparison plot at
# each milestone (steps 200, 400, 600, 800/last).
# Saves each milestone PNG with a unique name and a "summary.txt".

set -u

RUN_GLOB="${RUN_GLOB:-run_*_fsdp2_offload_a3_pf_2ep_gbs128}"
RUN_DIR="${RUN_DIR:-$(ls -td /home/ubuntu/fyh/megatron-sft-recipes/experiments/gemma4_E4B_alt_offload/${RUN_GLOB}/ 2>/dev/null | head -1)}"
if [ -z "${RUN_DIR}" ]; then
    echo "[auto_plot] no run dir found, exit" >&2
    exit 1
fi
echo "[auto_plot] watching ${RUN_DIR}"

PLOT_PY=/home/ubuntu/fyh/megatron-sft-recipes/scripts/gemma4_E4B_opt/plot_compare_ds3_fsdp2.py
BASELINE="${BASELINE:-/home/ubuntu/fyh/megatron-sft-recipes/baselines/ds3_baseline_logging.jsonl}"
# When copying plots to /mnt/shared/fyh, optionally append a tag to filename
# so multiple runs don't overwrite each other (e.g. SHARED_TAG="_fp32rs"
# → /mnt/shared/fyh/compare_ds3_fsdp2_step50_fp32rs.png)
SHARED_TAG="${SHARED_TAG:-}"

MILESTONES=(${MILESTONES:-50 100 200 400 600 800})
NOTIFIED=()

is_notified() {
    local m="$1"
    for n in "${NOTIFIED[@]:-}"; do
        [ "$n" = "$m" ] && return 0
    done
    return 1
}

current_step() {
    local logfile=$(ls "${RUN_DIR}"/v0-*/logging.jsonl 2>/dev/null | head -1)
    [ -z "$logfile" ] && { echo 0; return; }
    grep -oE '"global_step/max_steps": "[0-9]+/' "$logfile" 2>/dev/null \
        | tail -1 | grep -oE '[0-9]+' | head -1
}

run_complete() {
    [ -f "${RUN_DIR}/STATUS" ]
}

generate_plot() {
    local milestone=$1
    local out="${RUN_DIR}/compare_ds3_fsdp2_step${milestone}.png"
    local summary="${RUN_DIR}/summary_step${milestone}.txt"
    local fsdp2_log=$(ls "${RUN_DIR}"/v0-*/logging.jsonl 2>/dev/null | head -1)

    echo "[auto_plot] === milestone step ${milestone} reached ==="
    local maxstep_flag=""
    if [ "${milestone}" != "FINAL" ]; then
        maxstep_flag="--max_step ${milestone}"
    fi
    docker exec fsdp_sft python "${PLOT_PY}" \
        --baseline "${BASELINE}" \
        --fsdp2 "${fsdp2_log}" \
        ${maxstep_flag} \
        --output "${out}" 2>&1 | tee "${summary}" | tail -20
    # If SHARED_TAG is set, insert it before .png in the shared filename
    local shared_name="$(basename ${out})"
    if [ -n "${SHARED_TAG}" ]; then
        shared_name="${shared_name%.png}${SHARED_TAG}.png"
    fi
    cp "${out}" "/mnt/shared/fyh/${shared_name}" 2>/dev/null && \
        echo "[auto_plot] also copied to /mnt/shared/fyh/${shared_name}"
}

# Initial plot if we're already past any milestone
INIT=$(current_step)
echo "[auto_plot] starting at step ${INIT} / 806"

while true; do
    STEP=$(current_step)
    if [ -z "$STEP" ] || [ "$STEP" = "0" ]; then
        sleep 30
        continue
    fi

    for m in "${MILESTONES[@]}"; do
        if [ "$STEP" -ge "$m" ] && ! is_notified "$m"; then
            generate_plot "$m"
            NOTIFIED+=("$m")
        fi
    done

    if run_complete; then
        echo "[auto_plot] STATUS file appeared, run finished."
        # Final plot with full data
        generate_plot "FINAL"
        # Also generate the canonical filename
        local fsdp2_log=$(ls "${RUN_DIR}"/v0-*/logging.jsonl 2>/dev/null | head -1)
        docker exec fsdp_sft python "${PLOT_PY}" \
            --baseline "${BASELINE}" \
            --fsdp2 "${fsdp2_log}" 2>&1 | tail -20
        echo "[auto_plot] === all done at step ${STEP} ==="
        exit 0
    fi

    sleep 60
done
