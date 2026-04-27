#!/usr/bin/env bash
# diy_restore.sh — Undo diy_reset.sh: re-apply all patches to fsdp_sft container.
#
# Run this AFTER you finish the DIY journey to get back to a working state.
# Re-running is safe.
set -euo pipefail

REPO=/home/ubuntu/fyh/megatron-sft-recipes
PREAMBLE_DIR="${REPO}/scripts/gemma4_opt/_sdp_preamble"
LIGER_FILE="${REPO}/scripts/benchmark/liger_gemma4_patch.py"

echo "[1/3] Re-applying modeling_gemma4 compat patch..."
NEW_MD5=$(docker exec fsdp_sft md5sum /usr/local/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py | awk '{print $1}')
if [ "${NEW_MD5}" = "39ebf386a992fea9eac0883f459ac658" ]; then
    echo "  modeling_gemma4 already at patched md5; skipping."
else
    docker exec fsdp_sft bash -lc "
cd /usr/local/lib/python3.12/site-packages/transformers/models/gemma4
patch < ${REPO}/scripts/gemma4_opt/gemma4_modeling_compat.patch
md5sum modeling_gemma4.py
"
fi

echo ""
echo "[2/3] Restoring sitecustomize.py..."
if [ -f "${PREAMBLE_DIR}/sitecustomize.py.applied" ]; then
    mv "${PREAMBLE_DIR}/sitecustomize.py.applied" "${PREAMBLE_DIR}/sitecustomize.py"
    echo "  sitecustomize.py.applied -> sitecustomize.py"
elif [ -f "${PREAMBLE_DIR}/sitecustomize.py" ]; then
    echo "  already in place"
else
    echo "  WARN: neither found"
fi

echo ""
echo "[3/3] Restoring liger_gemma4_patch.py..."
if [ -f "${LIGER_FILE}.applied" ]; then
    mv "${LIGER_FILE}.applied" "${LIGER_FILE}"
    echo "  liger_gemma4_patch.py.applied -> liger_gemma4_patch.py"
elif [ -f "${LIGER_FILE}" ]; then
    echo "  already in place"
else
    echo "  WARN: neither found"
fi

echo ""
echo "Restore complete. You can now run the production scripts:"
echo "  bash scripts/gemma4_opt/p5_liger.sh"
