#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/instruction_guidance/r1.0.1/reconstruction}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/reconstruction/recon_complex_test_output}"
GPU_ID="${GPU_ID:-0}"

PYTHON_NC="${PYTHON_NC:-python}"
PIPELINE="${ROOT_DIR}/run_pipeline.py"

SAMPLES=(
  "25199_39e3c0d3_0001"    # 2034f 30ext NewBody+Join+Cut
  "30980_d0535092_0000"    # 3312f 25ext NewBody+Join+Cut
  "134072_a64e5fc0_0000"  # 3436f 19ext NewBody+Join+Cut
  "41759_e29b38cb_0000"   # 2912f 10ext NewBody+Join+Cut
  "24548_a611c624_0000"   # 1596f 14ext NewBody+Join+Cut
  "69057_8b7468bb_0000"   # 1214f 14ext NewBody+Join+Cut
  "52682_cc775ecf_0000"   # 1692f 5ext  Intersect+NewBody
  "117786_5f9a7dda_0000"  # 3078f 2ext  Intersect+NewBody
)

mkdir -p "${OUTPUT_ROOT}"
LOGFILE="${OUTPUT_ROOT}/batch_run.log"
> "${LOGFILE}"

TOTAL=${#SAMPLES[@]}
IDX=0
DONE=0
FAIL=0

for SAMPLE_NAME in "${SAMPLES[@]}"; do
    IDX=$((IDX + 1))
    MESH_PATH="${DATASET_ROOT}/${SAMPLE_NAME}.obj"

    if [ ! -f "${MESH_PATH}" ]; then
        echo "[${IDX}/${TOTAL}] SKIP ${SAMPLE_NAME}: mesh not found" | tee -a "${LOGFILE}"
        FAIL=$((FAIL + 1))
        continue
    fi

    SAMPLE_OUTPUT="${OUTPUT_ROOT}/samples/${SAMPLE_NAME}"
    echo "" | tee -a "${LOGFILE}"
    echo ">>> [${IDX}/${TOTAL}] ${SAMPLE_NAME}  $(date '+%H:%M:%S')" | tee -a "${LOGFILE}"

    if "${PYTHON_NC}" "${PIPELINE}" \
        --input_mesh "${MESH_PATH}" \
        --output_dir "${SAMPLE_OUTPUT}" \
        --no_timestamp \
        --guidance_mode instruction \
        --instruction_dataset_root "${DATASET_ROOT}" \
        --n_samples 5000 \
        --n_points 5000 \
        --num_epochs 1 \
        --lr 5e-5 \
        --gradient_size 30 \
        --max_faces 5000 \
        --gpu_id "${GPU_ID}" \
        >> "${LOGFILE}" 2>&1; then
        echo "    OK  $(date '+%H:%M:%S')" | tee -a "${LOGFILE}"
        DONE=$((DONE + 1))
    else
        echo "    FAILED  $(date '+%H:%M:%S')" | tee -a "${LOGFILE}"
        FAIL=$((FAIL + 1))
    fi
done

echo "" | tee -a "${LOGFILE}"
echo "============================================================" | tee -a "${LOGFILE}"
echo "  BATCH COMPLETE: ${TOTAL} total, ${DONE} done, ${FAIL} failed" | tee -a "${LOGFILE}"
echo "============================================================" | tee -a "${LOGFILE}"
