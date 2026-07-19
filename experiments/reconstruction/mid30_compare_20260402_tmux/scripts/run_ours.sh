#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ROOT="${REPO_ROOT}/experiments/reconstruction/mid30_compare_20260402_tmux"
SELECTED="${ROOT}/selected_samples.txt"
LOG="${ROOT}/logs/ours.log"
MESH_ROOT="${MESH_ROOT:-${REPO_ROOT}/instruction_guidance/r1.0.1/reconstruction}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${ROOT}/logs" "${ROOT}/ours"

while read -r sample; do
  [[ -n "${sample}" ]] || continue
  mesh="${MESH_ROOT}/${sample}.obj"
  out="${ROOT}/ours/${sample}"
  echo "[$(date '+%F %T')] START ours ${sample}" >> "${LOG}"
  if PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" "${REPO_ROOT}/run_pipeline.py" \
      --input_mesh "${mesh}" \
      --guidance_mode instruction \
      --instruction_dataset_root "${MESH_ROOT}" \
      --output_dir "${out}" \
      --no_timestamp \
      --gpu_id 0 \
      --n_samples 5000 \
      --n_points 5000 \
      --num_epochs 1 \
      --lr 5e-5 \
      --max_faces 5000 \
      --gradient_size 30 \
      --extract_timeout 1800 \
      --target_quad_ratio 0.5 \
      --max_catmull_clark_iters 2 >> "${LOG}" 2>&1; then
    echo "[$(date '+%F %T')] OK ours ${sample}" >> "${LOG}"
  else
    echo "[$(date '+%F %T')] FAIL ours ${sample}" >> "${LOG}"
  fi
done < "${SELECTED}"
