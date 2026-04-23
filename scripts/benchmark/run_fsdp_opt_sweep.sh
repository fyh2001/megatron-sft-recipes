#!/usr/bin/env bash
# run_fsdp_opt_sweep.sh
#   Executes the 10-config FSDP2 optimisation sweep defined in the plan
#   `fsdp2-sp-optimization-sweep`. Each config runs twice:
#     (a) 10-step real training under `bench_swift_sp_v2.sh` (DCGM @ 10 Hz)
#     (b) 5-step profile run under `bench_swift_sp_profile.sh`
#   Results land under `$BENCH_ROOT/$CFG_NAME/train/` and `.../profile/`.
#
# Selectors (env vars):
#   TIERS="1 2 3"           default; pick subset of "1", "2", "3"
#   ONLY="no_ac compile"    run only the named configs (space separated)
#   DRY_RUN=true            print what would run, don't execute
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

: "${BENCH_ROOT:=${OUTPUT_ROOT}/bench_fsdp_opt}"
: "${TRAIN_STEPS:=10}"
: "${PROFILE_STEPS:=5}"
: "${PROFILE_STEP:=3}"
: "${MASTER_PORT_BASE:=29533}"
: "${TIERS:=1 2 3}"
: "${ONLY:=}"
: "${DRY_RUN:=false}"

mkdir -p "${BENCH_ROOT}"

# ---- config matrix ----
# Format per line:
#   name tier BACKEND SP MBS TORCH_COMPILE NO_AC FSDP_RESHARD FSDP_WRAP_POLICY FSDP_MIN_NUM_PARAMS
# An empty field is represented by '-'.
read -r -d '' CONFIG_MATRIX <<'EOF' || true
baseline      1 fsdp2 2 1 false false true  -                  -
no_ac         1 fsdp2 2 1 false true  true  -                  -
mbs2          1 fsdp2 2 2 false false true  -                  -
compile       1 fsdp2 2 1 true  false true  -                  -
mbs4          2 fsdp2 2 4 false false true  -                  -
no_reshard    2 fsdp2 2 1 false false false -                  -
wrap_large    2 fsdp2 2 1 false false true  SIZE_BASED_WRAP    100000000
sp1_mbs2      2 fsdp2 1 2 false false true  -                  -
combo_easy    3 fsdp2 2 2 false true  false -                  -
combo_compile 3 fsdp2 2 2 true  true  false -                  -
EOF

in_set() {  # usage: in_set key "a b c"
    local key="$1" set="$2"
    [ -z "${set}" ] && return 0  # empty set = always match
    for k in ${set}; do
        [ "${k}" = "${key}" ] && return 0
    done
    return 1
}

PORT=${MASTER_PORT_BASE}

run_one() {
    local name="$1" tier="$2" backend="$3" sp="$4" mbs="$5" \
          compile="$6" no_ac="$7" reshard="$8" wrap="$9" minparams="${10}"

    if ! in_set "${tier}" "${TIERS}"; then
        log "[sweep] skip ${name} (tier=${tier})"
        return 0
    fi
    if [ -n "${ONLY}" ] && ! in_set "${name}" "${ONLY}"; then
        log "[sweep] skip ${name} (not in ONLY='${ONLY}')"
        return 0
    fi

    local wrap_env=""
    [ "${wrap}" != "-" ] && wrap_env="FSDP_WRAP_POLICY=${wrap}"
    local minp_env=""
    [ "${minparams}" != "-" ] && minp_env="FSDP_MIN_NUM_PARAMS=${minparams}"

    local cfg_dir="${BENCH_ROOT}/${name}"
    mkdir -p "${cfg_dir}"
    echo "{\"name\":\"${name}\",\"tier\":${tier},\"backend\":\"${backend}\",\"sp\":${sp},\"mbs\":${mbs},\"torch_compile\":${compile},\"no_ac\":${no_ac},\"fsdp_reshard\":${reshard},\"fsdp_wrap_policy\":\"${wrap}\",\"fsdp_min_num_params\":\"${minparams}\"}" > "${cfg_dir}/cfg.json"

    local common_env="BACKEND=${backend} SP=${sp} MBS=${mbs} TORCH_COMPILE=${compile} NO_AC=${no_ac} FSDP_RESHARD=${reshard} ${wrap_env} ${minp_env}"

    log "=============================================="
    log "[sweep] ${name}  tier=${tier}  ${common_env}"
    log "=============================================="

    if [ "${DRY_RUN}" = "true" ]; then
        echo "DRY: (a) train: ${common_env} TOTAL_STEPS=${TRAIN_STEPS} bench_swift_sp_v2.sh"
        echo "DRY: (b) profile: ${common_env} TOTAL_STEPS=${PROFILE_STEPS} PROFILE_STEP=${PROFILE_STEP} bench_swift_sp_profile.sh"
        return 0
    fi

    # ---- (a) training run ----
    local train_port=$((PORT)); PORT=$((PORT + 1))
    local train_rc=0
    set +e
    docker exec fsdp_sft bash -lc "
        cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
        MASTER_PORT=${train_port} ${common_env} \
        TOTAL_STEPS=${TRAIN_STEPS} WARMUP_BENCH=3 \
        RUN_NAME=train \
        BENCH_DIR='${cfg_dir}' \
        bash scripts/benchmark/bench_swift_sp_v2.sh
    " > "${cfg_dir}/train_runner.log" 2>&1
    train_rc=$?
    set -e
    echo "train_rc=${train_rc}" > "${cfg_dir}/train.rc"
    log "[sweep] ${name}: train rc=${train_rc}"
    tail -5 "${cfg_dir}/train_runner.log" || true
    if [ "${train_rc}" != "0" ]; then
        log "[sweep] ${name}: TRAIN FAILED (rc=${train_rc}), skipping profile"
        return 0
    fi

    # ---- (b) profile run ----
    local profile_port=$((PORT)); PORT=$((PORT + 1))
    local prof_rc=0
    set +e
    docker exec fsdp_sft bash -lc "
        cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
        MASTER_PORT=${profile_port} ${common_env} \
        TOTAL_STEPS=${PROFILE_STEPS} PROFILE_STEP=${PROFILE_STEP} \
        RUN_NAME=profile \
        BENCH_DIR='${cfg_dir}' \
        bash scripts/benchmark/bench_swift_sp_profile.sh
    " > "${cfg_dir}/profile_runner.log" 2>&1
    prof_rc=$?
    set -e
    echo "profile_rc=${prof_rc}" > "${cfg_dir}/profile.rc"
    log "[sweep] ${name}: profile rc=${prof_rc}"
    tail -5 "${cfg_dir}/profile_runner.log" || true
}

while IFS= read -r line; do
    [ -z "${line}" ] && continue
    # shellcheck disable=SC2086
    set -- ${line}
    run_one "$@"
done <<< "${CONFIG_MATRIX}"

log "[sweep] all configs done. Building summary..."
docker exec fsdp_sft bash -lc "
    cd /home/ubuntu/perf_opt/megatron-sft-recipes && \
    python scripts/benchmark/build_fsdp_opt_summary.py \
        --bench_root '${BENCH_ROOT}' \
        --out_md '${BENCH_ROOT}/_opt_summary.md' \
        --out_json '${BENCH_ROOT}/_opt_summary.json'
"
log "[sweep] summary: ${BENCH_ROOT}/_opt_summary.md"
