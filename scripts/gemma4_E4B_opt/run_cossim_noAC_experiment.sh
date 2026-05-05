#!/usr/bin/env bash
# Run the AC=off + Liger=off cossim experiment.
#
# Hypothesis: layer 0 cos=0.36 in DS3-AC-on vs FSDP2-AC-on comparison may be
# caused by different AC implementations between the two engines (FSDP2 wraps
# layer in CheckpointWrapper, DS3 uses native gradient_checkpointing). Turning
# off AC in both should isolate the AC effect from pure bf16 numerical drift.
#
# Outputs:
#   - DUMP_BASE/ds3_noAC/         (raw .pt grads from DS3 AC=off)
#   - DUMP_BASE/fsdp2_detach_noAC/ (raw .pt grads from FSDP2 detach AC=off)
#   - DUMP_BASE/cossim_noAC.txt   (analysis output)
#
# Usage:
#   bash scripts/gemma4_E4B_opt/run_cossim_noAC_experiment.sh

set -euo pipefail

REPO=/home/ubuntu/fyh/megatron-sft-recipes
DUMP_BASE="${REPO}/experiments/gemma4_E4B_alt_offload/cossim_dump_noAC_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${DUMP_BASE}/ds3_noAC" "${DUMP_BASE}/fsdp2_detach_noAC"

echo "============================================================"
echo "AC=off + Liger=off cossim experiment"
echo "DUMP_BASE: ${DUMP_BASE}"
echo "============================================================"

# Common params to dump
RAW_PREFIXES="model.language_model.layers.22.self_attn.k_proj,model.language_model.layers.22.self_attn.v_proj,model.language_model.layers.0.self_attn.q_proj,model.language_model.layers.41.self_attn.q_proj"

# ─── Stage 1: DS3 baseline AC=off Liger=off ──────────────────────────
echo ""
echo "[1/3] DS3 (AC=off, Liger=off) — 1 step"
echo "============================================================"
cd "${REPO}"
LABEL="cossim_ds3_noAC" \
TEMPLATE="gemma4" \
WEIGHT_DECAY="0.1" \
MAX_STEPS=1 \
USE_LIGER="false" \
GRAD_CKPT="false" \
EXTRA_ENV="GEMMA4_GRAD_DUMP=1 GEMMA4_GRAD_DUMP_DIR=${DUMP_BASE}/ds3_noAC GEMMA4_GRAD_DUMP_MAX_STEPS=1 GEMMA4_GRAD_DUMP_RAW_PREFIXES=${RAW_PREFIXES}" \
EXTRA_ARGS="--max_grad_norm 1.0" \
bash scripts/gemma4_E4B_opt/bench_ds3.sh 2>&1 | tee "${DUMP_BASE}/ds3_noAC.log"

# ─── Stage 2: FSDP2 detach AC=off Liger=off ──────────────────────────
echo ""
echo "[2/3] FSDP2 detach (AC=off, Liger=off) — 1 step"
echo "============================================================"
cd "${REPO}"
LABEL="cossim_fsdp2_detach_noAC" \
TEMPLATE="gemma4" \
WEIGHT_DECAY="0.1" \
MAX_STEPS=1 \
FULL_SCHED_STOP=1 \
FSDP_AC_ON="false" \
USE_LIGER="false" \
EXTRA_ENV="GEMMA4_KV_SHARE_DETACH=1 GEMMA4_GRAD_DUMP=1 GEMMA4_GRAD_DUMP_DIR=${DUMP_BASE}/fsdp2_detach_noAC GEMMA4_GRAD_DUMP_MAX_STEPS=1 GEMMA4_GRAD_DUMP_FORCE_SYNC=1 GEMMA4_FSDP_WRAP_PLE=1 GEMMA4_GRAD_DUMP_RAW_PREFIXES=${RAW_PREFIXES}" \
EXTRA_ARGS="--max_grad_norm 1.0" \
bash scripts/gemma4_E4B_opt/bench_variant.sh 2>&1 | tee "${DUMP_BASE}/fsdp2_detach_noAC.log"

# ─── Stage 3: Compute cossim ─────────────────────────────────────────
echo ""
echo "[3/3] Compute cossim DS3-noAC vs FSDP2-detach-noAC"
echo "============================================================"
docker exec fsdp_sft python3 ${REPO}/scripts/gemma4_E4B_opt/compute_cossim.py \
    --ds3 "${DUMP_BASE}/ds3_noAC" \
    --fsdp2 "${DUMP_BASE}/fsdp2_detach_noAC" \
    --label "FSDP2_detach_noAC" 2>&1 | tee "${DUMP_BASE}/cossim_noAC.txt"

echo ""
echo "============================================================"
echo "DONE. Results saved to ${DUMP_BASE}/cossim_noAC.txt"
echo "============================================================"
