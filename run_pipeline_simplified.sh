#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_PY="${ROOT_DIR}/run_pipeline.py"

INPUT_PATH="${1:-${ROOT_DIR}/input}"
OUTPUT_DIR="${2:-${ROOT_DIR}/pipeline_output_simplified}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))

PYTHON_BIN="${PYTHON_BIN:-/root/.conda/envs/neurcross/bin/python}"

GUIDANCE_MODE="${GUIDANCE_MODE:-feature}"
MAX_FACES="${MAX_FACES:-60000}"
GRADIENT_SIZE="${GRADIENT_SIZE:-30.0}"
EXTRACT_TIMEOUT="${EXTRACT_TIMEOUT:-1800}"
TARGET_QUAD_RATIO="${TARGET_QUAD_RATIO:-0.5}"
MAX_CATMULL_CLARK_ITERS="${MAX_CATMULL_CLARK_ITERS:-2}"
DISABLE_AUTO_SWEEP="${DISABLE_AUTO_SWEEP:-0}"
DISABLE_EXTRACT_RETRY="${DISABLE_EXTRACT_RETRY:-0}"
KEEP_SWEEP_OUTPUTS="${KEEP_SWEEP_OUTPUTS:-0}"
NO_TIMESTAMP="${NO_TIMESTAMP:-0}"

if [[ ! -f "${PIPELINE_PY}" ]]; then
  echo "Pipeline script not found: ${PIPELINE_PY}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

input_args=()
if [[ -f "${INPUT_PATH}" ]]; then
  input_args=(--input_mesh "${INPUT_PATH}")
elif [[ -d "${INPUT_PATH}" ]]; then
  input_args=(--input_dir "${INPUT_PATH}")
else
  echo "Input path not found: ${INPUT_PATH}" >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}" "${PIPELINE_PY}"
  "${input_args[@]}"
  --output_dir "${OUTPUT_DIR}"
  --guidance_mode "${GUIDANCE_MODE}"
  --max_faces "${MAX_FACES}"
  --gradient_size "${GRADIENT_SIZE}"
  --extract_timeout "${EXTRACT_TIMEOUT}"
  --target_quad_ratio "${TARGET_QUAD_RATIO}"
  --max_catmull_clark_iters "${MAX_CATMULL_CLARK_ITERS}"
)

if [[ "${NO_TIMESTAMP}" == "1" ]]; then
  cmd+=(--no_timestamp)
fi

if [[ "${DISABLE_AUTO_SWEEP}" == "1" ]]; then
  cmd+=(--disable_auto_sweep)
else
  cmd+=(--sweep_values 8 12 16 24 30 40 60)
fi

if [[ "${DISABLE_EXTRACT_RETRY}" == "1" ]]; then
  cmd+=(--disable_extract_retry)
fi

if [[ "${KEEP_SWEEP_OUTPUTS}" == "1" ]]; then
  cmd+=(--keep_sweep_outputs)
fi

extra_args=("$@")
if [[ "${#extra_args[@]}" -gt 0 ]]; then
  cmd+=("${extra_args[@]}")
fi

echo "Root          : ${ROOT_DIR}"
echo "Input         : ${INPUT_PATH}"
echo "Output        : ${OUTPUT_DIR}"
echo "Python        : ${PYTHON_BIN}"
echo "Guidance mode : ${GUIDANCE_MODE}"
echo "Max faces     : ${MAX_FACES}"
echo "Gradient size : ${GRADIENT_SIZE}"
echo "Extract timeout: ${EXTRACT_TIMEOUT} (set via passthrough args if needed)"
echo "Target ratio  : ${TARGET_QUAD_RATIO}"
echo "CC max iters  : ${MAX_CATMULL_CLARK_ITERS}"
echo
echo "Running:"
printf '  %q' "${cmd[@]}"
echo

exec "${cmd[@]}"
