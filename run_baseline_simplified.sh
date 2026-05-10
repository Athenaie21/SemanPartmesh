#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${1:-${ROOT_DIR}/input}"
OUTPUT_DIR="${2:-${ROOT_DIR}/baseline_output_simplified}"
PYTHON_BIN="${PYTHON_BIN:-/root/.conda/envs/neurcross/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
EXTRACT_PY="${ROOT_DIR}/extract_quad.py"

QUAD_MESH_DIR="${ROOT_DIR}/Baseline/NeurCross/quad_mesh"
TRAIN_PY="${QUAD_MESH_DIR}/train_quad_mesh.py"

N_SAMPLES="${N_SAMPLES:-10000}"
N_POINTS="${N_POINTS:-15000}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LR="${LR:-5e-5}"
LOSS_WEIGHTS="${LOSS_WEIGHTS:-7000 600 10 50 30 3}"

# Simplification controls (new)
MAX_FACES="${MAX_FACES:-60000}"
SIMPLIFY_ON_FAIL_KEEP_ORIGINAL="${SIMPLIFY_ON_FAIL_KEEP_ORIGINAL:-1}"

# Extraction controls
DO_EXTRACT="${DO_EXTRACT:-1}"
EXTRACT_ONLY="${EXTRACT_ONLY:-0}"
QUAD_OUTPUT_DIR="${QUAD_OUTPUT_DIR:-${OUTPUT_DIR}/quad_meshes}"
EXTRACT_MESH_DIR="${EXTRACT_MESH_DIR:-${OUTPUT_DIR}/extract_meshes}"
GRADIENT_SIZE="${GRADIENT_SIZE:-30.0}"
EXTRACT_TIMEOUT="${EXTRACT_TIMEOUT:-1800}"
EXTRACT_RETRY="${EXTRACT_RETRY:-1}"
EXTRACT_AUTO_SWEEP="${EXTRACT_AUTO_SWEEP:-1}"
EXTRACT_AFTER_TRAIN_EACH="${EXTRACT_AFTER_TRAIN_EACH:-0}"
RUN_BATCH_EXTRACT_PASS="${RUN_BATCH_EXTRACT_PASS:-1}"

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

mkdir -p "${OUTPUT_DIR}" "${QUAD_OUTPUT_DIR}" "${EXTRACT_MESH_DIR}"
PROCESSED_MESH_DIR="${OUTPUT_DIR}/processed_meshes"
mkdir -p "${PROCESSED_MESH_DIR}"

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

echo "Root        : ${ROOT_DIR}"
echo "Input dir   : ${INPUT_DIR}"
echo "Output dir  : ${OUTPUT_DIR}"
echo "Python      : ${PYTHON_BIN}"
echo "GPU         : ${CUDA_DEVICE}"
echo "Meshes      : ${#mesh_files[@]}"
echo "Max faces   : ${MAX_FACES}"
echo "Extract     : ${DO_EXTRACT} (extract-only=${EXTRACT_ONLY}, per-mesh=${EXTRACT_AFTER_TRAIN_EACH}, batch-pass=${RUN_BATCH_EXTRACT_PASS})"
echo "Quad out    : ${QUAD_OUTPUT_DIR}"
echo "Extract src : ${EXTRACT_MESH_DIR}"
echo

work_meshes=()
skipped_meshes=()

simplify_mesh_if_needed() {
  local src_mesh="$1"
  local dst_mesh="$2"
  local max_faces="$3"

  "${PYTHON_BIN}" - "$src_mesh" "$dst_mesh" "$max_faces" <<'PY'
import os
import sys
import trimesh

src = sys.argv[1]
dst = sys.argv[2]
target = int(sys.argv[3])

try:
    mesh = trimesh.load_mesh(src, process=False)
except Exception as exc:
    print(f"LOAD_ERROR {exc}")
    raise SystemExit(12)
if not isinstance(mesh, trimesh.Trimesh):
    print("LOAD_ERROR expected trimesh.Trimesh")
    raise SystemExit(12)

n_faces = len(mesh.faces)
if target <= 0 or n_faces <= target:
    print(f"NO_SIMPLIFY {n_faces}")
    raise SystemExit(10)

try:
    import fast_simplification
except Exception as exc:
    print(f"SIMPLIFY_IMPORT_ERROR {exc}")
    raise SystemExit(11)

target_reduction = 1.0 - float(target) / float(n_faces)
v_out, f_out = fast_simplification.simplify(
    mesh.vertices, mesh.faces, target_reduction=target_reduction)
simplified = trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)
os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
simplified.export(dst)
print(f"SIMPLIFIED {n_faces} {len(simplified.faces)} {dst}")
PY
}

for mesh_path in "${mesh_files[@]}"; do
  mesh_name="$(basename "${mesh_path}")"
  mesh_stem="${mesh_name%.*}"
  work_mesh="${mesh_path}"

  simplified_obj="${PROCESSED_MESH_DIR}/${mesh_stem}.obj"
  set +e
  simplify_mesh_if_needed "${mesh_path}" "${simplified_obj}" "${MAX_FACES}"
  simplify_rc=$?
  set -e
  if [[ ${simplify_rc} -eq 0 ]]; then
    work_mesh="${simplified_obj}"
  elif [[ ${simplify_rc} -eq 10 ]]; then
    work_mesh="${mesh_path}"
  elif [[ ${simplify_rc} -eq 12 ]]; then
    echo "WARNING: skip unreadable/invalid mesh ${mesh_name} (PLY faces may be malformed)."
    skipped_meshes+=("${mesh_path}")
    continue
  else
    if [[ "${SIMPLIFY_ON_FAIL_KEEP_ORIGINAL}" == "1" ]]; then
      echo "WARNING: simplify failed for ${mesh_name}, fallback to original mesh."
      work_mesh="${mesh_path}"
    else
      echo "ERROR: simplify failed for ${mesh_name} and fallback is disabled." >&2
      exit 1
    fi
  fi

  work_meshes+=("${work_mesh}")

  stage_name="$(basename "${work_mesh}")"
  stage_path="${EXTRACT_MESH_DIR}/${stage_name}"
  ln -sfn "$(readlink -f "${work_mesh}")" "${stage_path}"
done

if [[ ${#skipped_meshes[@]} -gt 0 ]]; then
  echo
  echo "Skipped meshes:"
  for skipped_mesh in "${skipped_meshes[@]}"; do
    echo "  - $(basename "${skipped_mesh}")"
  done
  echo
fi

if [[ ${#work_meshes[@]} -eq 0 ]]; then
  echo "No valid meshes remained after input loading/parsing." >&2
  exit 1
fi

if [[ "${EXTRACT_ONLY}" != "1" ]]; then
  for work_mesh in "${work_meshes[@]}"; do
    mesh_name="$(basename "${work_mesh}")"
    mesh_stem="${mesh_name%.*}"
    echo ">>> Running baseline NeurCross for: ${mesh_name}"

    (
      cd "${QUAD_MESH_DIR}"
      CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
        "${PYTHON_BIN}" train_quad_mesh.py \
        --data_path "${work_mesh}" \
        --logdir "${OUTPUT_DIR}" \
        --n_samples "${N_SAMPLES}" \
        --n_points "${N_POINTS}" \
        --num_epochs "${NUM_EPOCHS}" \
        --lr "${LR}" \
        --loss_weights ${LOSS_WEIGHTS} \
        --morse_near
    )

    echo "<<< Done: ${mesh_name}"

    if [[ "${DO_EXTRACT}" == "1" && "${EXTRACT_AFTER_TRAIN_EACH}" == "1" ]]; then
      echo ">>> Extracting quad mesh for: ${mesh_name}"
      retry_flag=()
      sweep_flag=()
      if [[ "${EXTRACT_RETRY}" == "1" ]]; then
        retry_flag=(--retry)
      fi
      if [[ "${EXTRACT_AUTO_SWEEP}" == "1" ]]; then
        sweep_flag=(--auto_sweep --sweep_values 8 12 16 24 30 40 60)
      fi

      "${PYTHON_BIN}" "${EXTRACT_PY}" \
        --mesh "${work_mesh}" \
        --crossfield_root "${OUTPUT_DIR}" \
        --output "${QUAD_OUTPUT_DIR}/${mesh_stem}_quad.obj" \
        --gradient_size "${GRADIENT_SIZE}" \
        --timeout "${EXTRACT_TIMEOUT}" \
        "${retry_flag[@]}" \
        "${sweep_flag[@]}"
      echo "<<< Extract done: ${mesh_name}"
    fi

    echo
  done
fi

if [[ "${DO_EXTRACT}" == "1" && "${RUN_BATCH_EXTRACT_PASS}" == "1" ]]; then
  echo ">>> Batch extraction pass on staged meshes"
  retry_flag=()
  sweep_flag=()
  if [[ "${EXTRACT_RETRY}" == "1" ]]; then
    retry_flag=(--retry)
  fi
  if [[ "${EXTRACT_AUTO_SWEEP}" == "1" ]]; then
    sweep_flag=(--auto_sweep --sweep_values 8 12 16 24 30 40 60)
  fi

  "${PYTHON_BIN}" "${EXTRACT_PY}" \
    --mesh_dir "${EXTRACT_MESH_DIR}" \
    --crossfield_root "${OUTPUT_DIR}" \
    --output_dir "${QUAD_OUTPUT_DIR}" \
    --gradient_size "${GRADIENT_SIZE}" \
    --timeout "${EXTRACT_TIMEOUT}" \
    "${retry_flag[@]}" \
    "${sweep_flag[@]}"
  echo "<<< Batch extraction finished"
fi

echo "All baseline simplified runs finished. Logs: ${OUTPUT_DIR}"
if [[ "${DO_EXTRACT}" == "1" ]]; then
  echo "Quad meshes are under: ${QUAD_OUTPUT_DIR}"
fi
