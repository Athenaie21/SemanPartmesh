#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INPUT_DIR="${1:-${ROOT_DIR}/input}"
OUTPUT_DIR="${2:-${ROOT_DIR}/experiments/pipeline/pipeline_output_simplified}"

PYTHON_BIN="${PYTHON_BIN:-python}"

GRADIENT_SIZE="${GRADIENT_SIZE:-30.0}"
EXTRACT_TIMEOUT="${EXTRACT_TIMEOUT:-300}"
TARGET_QUAD_RATIO="${TARGET_QUAD_RATIO:-0.5}"
MAX_CATMULL_CLARK_ITERS="${MAX_CATMULL_CLARK_ITERS:-2}"
MAX_AD="${MAX_AD:-15.0}"
MIN_JR="${MIN_JR:-0.15}"
BUDGET_TOLERANCE_RATIO="${BUDGET_TOLERANCE_RATIO:-0.2}"
DISABLE_AUTO_SWEEP="${DISABLE_AUTO_SWEEP:-1}"
DISABLE_EXTRACT_RETRY="${DISABLE_EXTRACT_RETRY:-1}"
KEEP_SWEEP_OUTPUTS="${KEEP_SWEEP_OUTPUTS:-0}"
SNAP_TO_BOUNDARY="${SNAP_TO_BOUNDARY:-1}"
DISABLE_CHUNKED_EXTRACT="${DISABLE_CHUNKED_EXTRACT:-0}"
ALLOW_PER_CHUNK_SIZE="${ALLOW_PER_CHUNK_SIZE:-0}"
EXTRACT_CHUNK_MIN_FACES="${EXTRACT_CHUNK_MIN_FACES:-200}"
EXTRACT_CHUNK_MAX_CHUNKS="${EXTRACT_CHUNK_MAX_CHUNKS:-24}"
SEMANTIC_SIZE_K_MIN="${SEMANTIC_SIZE_K_MIN:-2}"
SEMANTIC_SIZE_K_MAX="${SEMANTIC_SIZE_K_MAX:-15}"
SKIP_EXISTING_QUADS="${SKIP_EXISTING_QUADS:-0}"
MAX_FALLBACK_ATTEMPTS="${MAX_FALLBACK_ATTEMPTS:-1}"
MAX_REPAIR_TIERS="${MAX_REPAIR_TIERS:-2}"
KEEP_EXTRACT_INTERMEDIATES="${KEEP_EXTRACT_INTERMEDIATES:-0}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory not found: ${INPUT_DIR}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/quad_meshes"

echo "Root               : ${ROOT_DIR}"
echo "Input dir          : ${INPUT_DIR}"
echo "Pipeline output    : ${OUTPUT_DIR}"
echo "Python             : ${PYTHON_BIN}"
echo "Skip existing quads: ${SKIP_EXISTING_QUADS}"
echo "Extract timeout    : ${EXTRACT_TIMEOUT}"
echo "Auto sweep disabled: ${DISABLE_AUTO_SWEEP}"
echo "Retry disabled     : ${DISABLE_EXTRACT_RETRY}"
echo "Max fallbacks      : ${MAX_FALLBACK_ATTEMPTS}"
echo "Max repair tiers   : ${MAX_REPAIR_TIERS}"
echo
echo "Extracting from existing pipeline crossfields/features"

cd "${ROOT_DIR}"
export INPUT_DIR OUTPUT_DIR
export GRADIENT_SIZE EXTRACT_TIMEOUT TARGET_QUAD_RATIO MAX_CATMULL_CLARK_ITERS
export MAX_AD MIN_JR BUDGET_TOLERANCE_RATIO DISABLE_AUTO_SWEEP DISABLE_EXTRACT_RETRY
export KEEP_SWEEP_OUTPUTS SNAP_TO_BOUNDARY DISABLE_CHUNKED_EXTRACT ALLOW_PER_CHUNK_SIZE
export EXTRACT_CHUNK_MIN_FACES EXTRACT_CHUNK_MAX_CHUNKS
export SEMANTIC_SIZE_K_MIN SEMANTIC_SIZE_K_MAX SKIP_EXISTING_QUADS
export MAX_FALLBACK_ATTEMPTS MAX_REPAIR_TIERS KEEP_EXTRACT_INTERMEDIATES

"${PYTHON_BIN}" - <<'PY'
import glob
import os
from types import SimpleNamespace

import run_pipeline

input_dir = os.environ["INPUT_DIR"]
output_dir = os.environ["OUTPUT_DIR"]
processed_mesh_dir = os.path.join(output_dir, "processed_meshes")
feature_dir = os.path.join(output_dir, "partfield_features")
log_root = os.path.join(output_dir, "neurcross_logs")
quad_dir = os.path.join(output_dir, "quad_meshes")

args = SimpleNamespace(
    disable_chunked_extract=bool(int(os.environ["DISABLE_CHUNKED_EXTRACT"])),
    allow_per_chunk_size=bool(int(os.environ["ALLOW_PER_CHUNK_SIZE"])),
    guidance_mode="feature",
    semantic_size_k=None,
    semantic_size_k_min=int(os.environ["SEMANTIC_SIZE_K_MIN"]),
    semantic_size_k_max=int(os.environ["SEMANTIC_SIZE_K_MAX"]),
    extract_chunk_min_faces=int(os.environ["EXTRACT_CHUNK_MIN_FACES"]),
    extract_chunk_max_chunks=int(os.environ["EXTRACT_CHUNK_MAX_CHUNKS"]),
    gradient_size=float(os.environ["GRADIENT_SIZE"]),
    extract_timeout=int(os.environ["EXTRACT_TIMEOUT"]),
    max_ad=float(os.environ["MAX_AD"]),
    min_jr=float(os.environ["MIN_JR"]),
    min_quads=None,
    disable_auto_sweep=bool(int(os.environ["DISABLE_AUTO_SWEEP"])),
    sweep_values=[8.0, 12.0, 16.0, 24.0, 30.0, 40.0, 60.0],
    keep_sweep_outputs=bool(int(os.environ["KEEP_SWEEP_OUTPUTS"])),
    disable_extract_retry=bool(int(os.environ["DISABLE_EXTRACT_RETRY"])),
    max_fallback_attempts=int(os.environ["MAX_FALLBACK_ATTEMPTS"]),
    max_repair_tiers=int(os.environ["MAX_REPAIR_TIERS"]),
    catmull_clark_iters=0,
    target_quad_ratio=float(os.environ["TARGET_QUAD_RATIO"]),
    budget_tolerance_ratio=float(os.environ["BUDGET_TOLERANCE_RATIO"]),
    max_catmull_clark_iters=int(os.environ["MAX_CATMULL_CLARK_ITERS"]),
    snap_to_boundary=bool(int(os.environ["SNAP_TO_BOUNDARY"])),
    snap_fraction=0.3,
    snap_threshold_ratio=0.01,
    snap_min_angle=25.0,
    snap_smooth_iters=3,
    snap_smooth_weight=0.3,
    stitch_quad_patches=False,
    stitch_threshold_ratio=0.005,
    stitch_smooth_iters=2,
    stitch_smooth_weight=0.2,
    keep_extract_intermediates=bool(int(os.environ["KEEP_EXTRACT_INTERMEDIATES"])),
    size_field_path=None,
    disable_semantic_size_field=True,
)

requested = set()
for ext in ("*.obj", "*.ply", "*.off", "*.stl"):
    for mesh_path in glob.glob(os.path.join(input_dir, ext)):
        requested.add(os.path.splitext(os.path.basename(mesh_path))[0])

failures = []
for basename in sorted(requested):
    try:
        feature_path = run_pipeline.find_feature_file(feature_dir, basename)
    except FileNotFoundError:
        feature_path = None
    crossfield_dir = os.path.join(log_root, basename, "save_crossField")
    crossfield_path = run_pipeline.find_latest_crossfield(crossfield_dir)
    final_quad = os.path.join(quad_dir, f"{basename}_quad.obj")

    candidates = [
        os.path.join(processed_mesh_dir, f"{basename}.obj"),
        os.path.join(input_dir, f"{basename}.obj"),
        os.path.join(input_dir, f"{basename}.ply"),
        os.path.join(input_dir, f"{basename}.off"),
        os.path.join(input_dir, f"{basename}.stl"),
    ]
    mesh_path = next((p for p in candidates if os.path.isfile(p)), None)

    print()
    print(f"[Pipeline Extract Existing] {basename}")
    if mesh_path is None:
        failures.append((basename, "missing mesh input"))
        print("  missing mesh input")
        continue
    if feature_path is None:
        failures.append((basename, f"missing feature file under: {feature_dir}"))
        print(f"  missing feature file under: {feature_dir}")
        continue
    if crossfield_path is None:
        failures.append((basename, f"missing crossfield in {crossfield_dir}"))
        print(f"  missing crossfield in {crossfield_dir}")
        continue
    if bool(int(os.environ["SKIP_EXISTING_QUADS"])) and os.path.isfile(final_quad) and os.path.getsize(final_quad) > 0:
        print(f"  skip existing final quad: {final_quad}")
        continue

    try:
        meta = run_pipeline.run_chunked_quad_extract(
            mesh_path, feature_path, None, crossfield_path, final_quad, args)
        print(f"  extraction meta: {meta}")
        print(f"  output obj     : {final_quad}")
    except Exception as exc:  # noqa: BLE001
        failures.append((basename, str(exc)))
        print(f"  extraction failed: {exc}")

if failures:
    print()
    print("Pipeline existing-crossfield extraction failed for:")
    for basename, reason in failures:
        print(f"  - {basename}: {reason}")
    raise SystemExit(1)

print()
print("All requested pipeline models extracted from existing crossfields.")
print(f"Quad meshes are under: {quad_dir}")
PY
