#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${1:-${ROOT_DIR}/10wan}"
OUTPUT_DIR="${2:-${ROOT_DIR}/pipeline_output/baseline_logs}"
PYTHON_BIN="${PYTHON_BIN:-/root/.conda/envs/neurcross/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
EXTRACT_PY="${ROOT_DIR}/extract_quad.py"

QUAD_MESH_DIR="${ROOT_DIR}/Baseline/NeurCross/quad_mesh"
TRAIN_PY="${QUAD_MESH_DIR}/train_quad_mesh.py"

N_SAMPLES="${N_SAMPLES:-10000}"
N_POINTS="${N_POINTS:-10}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LR="${LR:-5e-5}"
LOSS_WEIGHTS="${LOSS_WEIGHTS:-7000 600 10 50 30 3}"
DO_EXTRACT="${DO_EXTRACT:-1}"
EXTRACT_ONLY="${EXTRACT_ONLY:-0}"
QUAD_OUTPUT_DIR="${QUAD_OUTPUT_DIR:-${OUTPUT_DIR}/quad_meshes}"
GRADIENT_SIZE="${GRADIENT_SIZE:-30.0}"
EXTRACT_TIMEOUT="${EXTRACT_TIMEOUT:-600}"
EXTRACT_RETRY="${EXTRACT_RETRY:-1}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory not found: ${INPUT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "Baseline train script not found: ${TRAIN_PY}" >&2
  exit 1
fi

if [[ ! -f "${EXTRACT_PY}" ]]; then
  echo "Quad extraction script not found: ${EXTRACT_PY}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${QUAD_OUTPUT_DIR}"

shopt -s nullglob nocaseglob
mesh_files=(
  "${INPUT_DIR}"/*.obj
  "${INPUT_DIR}"/*.ply
  "${INPUT_DIR}"/*.off
  "${INPUT_DIR}"/*.stl
)
shopt -u nocaseglob

if [[ ${#mesh_files[@]} -eq 0 ]]; then
  echo "No mesh files found in ${INPUT_DIR} (obj/ply/off/stl)." >&2
  exit 1
fi

echo "Root      : ${ROOT_DIR}"
echo "Input dir : ${INPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Python    : ${PYTHON_BIN}"
echo "GPU       : ${CUDA_DEVICE}"
echo "Meshes    : ${#mesh_files[@]}"
echo "Extract   : ${DO_EXTRACT} (extract-only=${EXTRACT_ONLY})"
echo "Quad out  : ${QUAD_OUTPUT_DIR}"
echo

if [[ "${EXTRACT_ONLY}" != "1" ]]; then
  for mesh_path in "${mesh_files[@]}"; do
    mesh_name="$(basename "${mesh_path}")"
    mesh_stem="${mesh_name%.*}"
    echo ">>> Running baseline NeurCross for: ${mesh_name}"

    (
      cd "${QUAD_MESH_DIR}"
      CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
        "${PYTHON_BIN}" train_quad_mesh.py \
        --data_path "${mesh_path}" \
        --logdir "${OUTPUT_DIR}" \
        --n_samples "${N_SAMPLES}" \
        --n_points "${N_POINTS}" \
        --num_epochs "${NUM_EPOCHS}" \
        --lr "${LR}" \
        --loss_weights ${LOSS_WEIGHTS} \
        --morse_near
    )

    echo "<<< Done: ${mesh_name}"

    if [[ "${DO_EXTRACT}" == "1" ]]; then
      echo ">>> Extracting quad mesh for: ${mesh_name}"
      retry_flag=()
      if [[ "${EXTRACT_RETRY}" == "1" ]]; then
        retry_flag=(--retry)
      fi

      "${PYTHON_BIN}" "${EXTRACT_PY}" \
        --mesh "${mesh_path}" \
        --crossfield_root "${OUTPUT_DIR}" \
        --output "${QUAD_OUTPUT_DIR}/${mesh_stem}_quad.obj" \
        --gradient_size "${GRADIENT_SIZE}" \
        --timeout "${EXTRACT_TIMEOUT}" \
        "${retry_flag[@]}"
      echo "<<< Extract done: ${mesh_name}"
    fi

    echo
  done
fi

if [[ "${DO_EXTRACT}" == "1" ]]; then
  echo ">>> Batch extraction pass on ${OUTPUT_DIR}"
  retry_flag=()
  if [[ "${EXTRACT_RETRY}" == "1" ]]; then
    retry_flag=(--retry)
  fi

  "${PYTHON_BIN}" "${EXTRACT_PY}" \
    --mesh_dir "${INPUT_DIR}" \
    --crossfield_root "${OUTPUT_DIR}" \
    --output_dir "${QUAD_OUTPUT_DIR}" \
    --gradient_size "${GRADIENT_SIZE}" \
    --timeout "${EXTRACT_TIMEOUT}" \
    "${retry_flag[@]}"
  echo "<<< Batch extraction finished"
fi

echo "All baseline runs finished. Logs: ${OUTPUT_DIR}"
if [[ "${DO_EXTRACT}" == "1" ]]; then
  echo "Quad meshes are under: ${QUAD_OUTPUT_DIR}"
fi
