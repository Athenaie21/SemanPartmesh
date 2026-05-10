#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

METHOD="${1:-pipeline}" # pipeline | baseline
INPUT_DIR="${2:-/root/shared-nvme/assemly}"
OUTPUT_DIR="${3:-/root/shared-nvme/output_${METHOD}_assemly}"
GPU_ID="${4:-0}"

PYTHON_BIN="${PYTHON_BIN:-/root/.conda/envs/neurcross/bin/python}"
PIPELINE_SCRIPT="${ROOT_DIR}/run_pipeline.py"
BASELINE_SCRIPT="${ROOT_DIR}/run_baseline_simplified.sh"
ITER_EXTRACT_SCRIPT="${ROOT_DIR}/run_extract_existing_iters.sh"

N_SAMPLES="${N_SAMPLES:-10000}"
N_POINTS="${N_POINTS:-15000}"
BASELINE_N_POINTS="${BASELINE_N_POINTS:-10}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LR="${LR:-5e-5}"
MAX_FACES="${MAX_FACES:-60000}"
ITER_LIST="${ITER_LIST:-3000 5000 8000 9999}"

EXTRACT_TIMEOUT="${EXTRACT_TIMEOUT:-600}"
GRADIENT_SIZE="${GRADIENT_SIZE:-30.0}"
TARGET_QUAD_RATIO="${TARGET_QUAD_RATIO:-0.5}"
BUDGET_TOLERANCE_RATIO="${BUDGET_TOLERANCE_RATIO:-0.2}"
MAX_CATMULL_CLARK_ITERS="${MAX_CATMULL_CLARK_ITERS:-2}"
MAX_FALLBACK_ATTEMPTS="${MAX_FALLBACK_ATTEMPTS:-1}"
MAX_REPAIR_TIERS="${MAX_REPAIR_TIERS:-2}"
DISABLE_AUTO_SWEEP="${DISABLE_AUTO_SWEEP:-1}"
DISABLE_EXTRACT_RETRY="${DISABLE_EXTRACT_RETRY:-1}"
SNAP_TO_BOUNDARY="${SNAP_TO_BOUNDARY:-0}"

if [[ "${METHOD}" != "pipeline" && "${METHOD}" != "baseline" ]]; then
  echo "METHOD must be one of: pipeline, baseline" >&2
  exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory not found: ${INPUT_DIR}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

for required_script in "${ITER_EXTRACT_SCRIPT}"; do
  if [[ ! -f "${required_script}" ]]; then
    echo "Required script not found: ${required_script}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"

echo "Root            : ${ROOT_DIR}"
echo "Method          : ${METHOD}"
echo "Input dir       : ${INPUT_DIR}"
echo "Output dir      : ${OUTPUT_DIR}"
echo "GPU             : ${GPU_ID}"
echo "n_samples       : ${N_SAMPLES}"
echo "Iter extraction : ${ITER_LIST}"
echo

if [[ "${METHOD}" == "pipeline" ]]; then
  "${PYTHON_BIN}" "${PIPELINE_SCRIPT}" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --no_timestamp \
    --guidance_mode feature \
    --n_samples "${N_SAMPLES}" \
    --n_points "${N_POINTS}" \
    --num_epochs "${NUM_EPOCHS}" \
    --lr "${LR}" \
    --max_faces "${MAX_FACES}" \
    --gpu_id "${GPU_ID}" \
    --skip_extract
else
  CUDA_DEVICE="${GPU_ID}" \
  N_SAMPLES="${N_SAMPLES}" \
  N_POINTS="${BASELINE_N_POINTS}" \
  NUM_EPOCHS="${NUM_EPOCHS}" \
  LR="${LR}" \
  MAX_FACES="${MAX_FACES}" \
  DO_EXTRACT=0 \
  RUN_BATCH_EXTRACT_PASS=0 \
  EXTRACT_AFTER_TRAIN_EACH=0 \
  bash "${BASELINE_SCRIPT}" "${INPUT_DIR}" "${OUTPUT_DIR}"
fi

echo
echo "Extracting saved crossfields with pipeline-style extraction"

ITER_LIST="${ITER_LIST}" \
PYTHON_BIN="${PYTHON_BIN}" \
GRADIENT_SIZE="${GRADIENT_SIZE}" \
EXTRACT_TIMEOUT="${EXTRACT_TIMEOUT}" \
TARGET_QUAD_RATIO="${TARGET_QUAD_RATIO}" \
BUDGET_TOLERANCE_RATIO="${BUDGET_TOLERANCE_RATIO}" \
MAX_CATMULL_CLARK_ITERS="${MAX_CATMULL_CLARK_ITERS}" \
MAX_FALLBACK_ATTEMPTS="${MAX_FALLBACK_ATTEMPTS}" \
MAX_REPAIR_TIERS="${MAX_REPAIR_TIERS}" \
DISABLE_AUTO_SWEEP="${DISABLE_AUTO_SWEEP}" \
DISABLE_EXTRACT_RETRY="${DISABLE_EXTRACT_RETRY}" \
SNAP_TO_BOUNDARY="${SNAP_TO_BOUNDARY}" \
bash "${ITER_EXTRACT_SCRIPT}" "${INPUT_DIR}" "${OUTPUT_DIR}" "${METHOD}"
