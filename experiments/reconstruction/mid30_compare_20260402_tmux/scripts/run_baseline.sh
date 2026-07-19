#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ROOT="${REPO_ROOT}/experiments/reconstruction/mid30_compare_20260402_tmux"
SELECTED="${ROOT}/selected_samples.txt"
LOG="${ROOT}/logs/baseline.log"
MESH_ROOT="${MESH_ROOT:-${REPO_ROOT}/instruction_guidance/r1.0.1/reconstruction}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${ROOT}/logs" "${ROOT}/baseline_inputs" "${ROOT}/baseline_runs"

while read -r sample; do
  [[ -n "${sample}" ]] || continue
  in_dir="${ROOT}/baseline_inputs/${sample}"
  out_dir="${ROOT}/baseline_runs/${sample}"
  mesh="${MESH_ROOT}/${sample}.obj"
  rm -rf "${in_dir}"
  mkdir -p "${in_dir}"
  ln -sf "${mesh}" "${in_dir}/${sample}.obj"
  echo "[$(date '+%F %T')] START baseline ${sample}" >> "${LOG}"
  if PYTHONDONTWRITEBYTECODE=1 \
      CUDA_DEVICE=1 \
      PYTHON_BIN="${PYTHON_BIN}" \
      N_SAMPLES=5000 \
      N_POINTS=5000 \
      NUM_EPOCHS=1 \
      LR=5e-5 \
      MAX_FACES=5000 \
      EXTRACT_TIMEOUT=1800 \
      EXTRACT_AUTO_SWEEP=1 \
      bash "${REPO_ROOT}/scripts/run_baseline_simplified.sh" "${in_dir}" "${out_dir}" >> "${LOG}" 2>&1; then
    echo "[$(date '+%F %T')] OK baseline ${sample}" >> "${LOG}"
  else
    echo "[$(date '+%F %T')] FAIL baseline ${sample}" >> "${LOG}"
  fi
done < "${SELECTED}"
