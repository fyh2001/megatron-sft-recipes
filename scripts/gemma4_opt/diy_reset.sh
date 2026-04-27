#!/usr/bin/env bash
# diy_reset.sh — Reset the fsdp_sft container to "no patches applied" state.
#
# Run this BEFORE starting the DIY journey (docs/gemma4_diy_journey.md).
# Re-running is safe.
#
# What it does:
#   1. Restore upstream modeling_gemma4.py (reinstall transformers, no patches).
#   2. Hide sitecustomize.py (rename to .applied so PYTHONPATH won't pick it up).
#   3. Hide liger_gemma4_patch.py (so even if Liger dispatch table got cached
#      from a previous run, fresh imports won't find our patch module).
#   4. Leave fyh2001/ms-swift fork in place — pypi 4.1.2 has an SP bug with
#      transformers 5.5.4, so we keep the fork but DIY-quest-10 will ask you
#      to re-introduce its 3 patches from a stripped-down state.
#
# After running this you should:
#   - md5sum modeling_gemma4.py → NOT 39ebf386a992fea9eac0883f459ac658 (= upstream)
#   - sitecustomize.py renamed to sitecustomize.py.applied
#   - liger_gemma4_patch.py renamed to liger_gemma4_patch.py.applied
set -euo pipefail

REPO=/home/ubuntu/fyh/megatron-sft-recipes
PREAMBLE_DIR="${REPO}/scripts/gemma4_opt/_sdp_preamble"
LIGER_FILE="${REPO}/scripts/benchmark/liger_gemma4_patch.py"

echo "[1/4] Reinstalling upstream transformers (reverts modeling_gemma4 patch)..."
docker exec fsdp_sft bash -lc "
pip install --force-reinstall --no-deps transformers==5.5.4 2>&1 | tail -3
"

echo ""
echo "[2/4] Verifying modeling_gemma4 is upstream..."
NEW_MD5=$(docker exec fsdp_sft md5sum /usr/local/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py | awk '{print $1}')
if [ "${NEW_MD5}" = "39ebf386a992fea9eac0883f459ac658" ]; then
    echo "  ERROR: md5 still matches the patched version. Reinstall failed?"
    exit 1
fi
echo "  modeling_gemma4 md5 = ${NEW_MD5} (upstream)"

echo ""
echo "[3/4] Hiding sitecustomize.py..."
if [ -f "${PREAMBLE_DIR}/sitecustomize.py" ]; then
    mv "${PREAMBLE_DIR}/sitecustomize.py" "${PREAMBLE_DIR}/sitecustomize.py.applied"
    echo "  ${PREAMBLE_DIR}/sitecustomize.py → .applied"
elif [ -f "${PREAMBLE_DIR}/sitecustomize.py.applied" ]; then
    echo "  ${PREAMBLE_DIR}/sitecustomize.py.applied (already hidden)"
else
    echo "  WARN: neither sitecustomize.py nor .applied found"
fi

echo ""
echo "[4/4] Hiding liger_gemma4_patch.py..."
if [ -f "${LIGER_FILE}" ]; then
    mv "${LIGER_FILE}" "${LIGER_FILE}.applied"
    echo "  ${LIGER_FILE} → .applied"
elif [ -f "${LIGER_FILE}.applied" ]; then
    echo "  ${LIGER_FILE}.applied (already hidden)"
else
    echo "  WARN: neither liger_gemma4_patch.py nor .applied found"
fi

echo ""
echo "Reset complete. You're now at \"no patches\" state."
echo "Open docs/gemma4_diy_journey.md and start with Quest 1."
echo ""
echo "When you finish a quest and want to compare vs the reference answer:"
echo "  diff <your_file> <reference>.applied"
echo ""
echo "When you're done with the whole journey and want to restore everything:"
echo "  bash scripts/gemma4_opt/diy_restore.sh"
