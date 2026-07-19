#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INPUT_DIR="${1:-${ROOT_DIR}/input}"
OUTPUT_DIR="${2:-${ROOT_DIR}/experiments/pipeline/pipeline_output}"
METHOD="${3:-pipeline}" # pipeline | baseline

PYTHON_BIN="${PYTHON_BIN:-python}"

GRADIENT_SIZE="${GRADIENT_SIZE:-30.0}"
EXTRACT_TIMEOUT="${EXTRACT_TIMEOUT:-600}"
TARGET_QUAD_RATIO="${TARGET_QUAD_RATIO:-0.5}"
MAX_CATMULL_CLARK_ITERS="${MAX_CATMULL_CLARK_ITERS:-2}"
MAX_AD="${MAX_AD:-15.0}"
MIN_JR="${MIN_JR:-0.15}"
BUDGET_TOLERANCE_RATIO="${BUDGET_TOLERANCE_RATIO:-0.2}"
DISABLE_AUTO_SWEEP="${DISABLE_AUTO_SWEEP:-1}"
DISABLE_EXTRACT_RETRY="${DISABLE_EXTRACT_RETRY:-1}"
KEEP_SWEEP_OUTPUTS="${KEEP_SWEEP_OUTPUTS:-0}"
SNAP_TO_BOUNDARY="${SNAP_TO_BOUNDARY:-0}"
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
ITER_LIST="${ITER_LIST:-}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory not found: ${INPUT_DIR}" >&2
  exit 1
fi

if [[ ! -d "${OUTPUT_DIR}" ]]; then
  echo "Output directory not found: ${OUTPUT_DIR}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ "${METHOD}" != "pipeline" && "${METHOD}" != "baseline" ]]; then
  echo "METHOD must be one of: pipeline, baseline" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/quad_meshes_by_iter"

echo "Root               : ${ROOT_DIR}"
echo "Input dir          : ${INPUT_DIR}"
echo "Output dir         : ${OUTPUT_DIR}"
echo "Method             : ${METHOD}"
echo "Python             : ${PYTHON_BIN}"
echo "Iter list          : ${ITER_LIST:-<all available>}"
echo "Skip existing quads: ${SKIP_EXISTING_QUADS}"
echo "Extract timeout    : ${EXTRACT_TIMEOUT}"
echo
echo "Extracting from existing crossfields (iter-by-iter)"

cd "${ROOT_DIR}"
export INPUT_DIR OUTPUT_DIR METHOD ITER_LIST
export GRADIENT_SIZE EXTRACT_TIMEOUT TARGET_QUAD_RATIO MAX_CATMULL_CLARK_ITERS
export MAX_AD MIN_JR BUDGET_TOLERANCE_RATIO DISABLE_AUTO_SWEEP DISABLE_EXTRACT_RETRY
export KEEP_SWEEP_OUTPUTS SNAP_TO_BOUNDARY DISABLE_CHUNKED_EXTRACT ALLOW_PER_CHUNK_SIZE
export EXTRACT_CHUNK_MIN_FACES EXTRACT_CHUNK_MAX_CHUNKS
export SEMANTIC_SIZE_K_MIN SEMANTIC_SIZE_K_MAX SKIP_EXISTING_QUADS
export MAX_FALLBACK_ATTEMPTS MAX_REPAIR_TIERS KEEP_EXTRACT_INTERMEDIATES

"${PYTHON_BIN}" - <<'PY'
import glob
import os
import re
import sys
import types
from types import SimpleNamespace

if "instruction_guidance" not in sys.modules:
    instruction_stub = types.ModuleType("instruction_guidance")
    instruction_stub.build_instruction_metadata = lambda *args, **kwargs: None
    instruction_stub.derive_mesh_basename = lambda path: os.path.splitext(os.path.basename(path))[0]
    instruction_stub.infer_dataset_paths = lambda *args, **kwargs: (None, None)
    instruction_stub.load_instruction_metadata = lambda *args, **kwargs: {}
    instruction_stub.load_instruction_prototypes = lambda *args, **kwargs: []
    instruction_stub.recommend_extraction_policy = lambda *args, **kwargs: {}
    instruction_stub.save_instruction_metadata = lambda *args, **kwargs: None
    sys.modules["instruction_guidance"] = instruction_stub

import run_pipeline

input_dir = os.environ["INPUT_DIR"]
output_dir = os.environ["OUTPUT_DIR"]
method = os.environ["METHOD"]
iter_env = os.environ.get("ITER_LIST", "").strip()
iter_filter = {int(x) for x in iter_env.split()} if iter_env else None

feature_dir = os.path.join(output_dir, "partfield_features")
if method == "pipeline":
    processed_mesh_dir = os.path.join(output_dir, "processed_meshes")
    crossfield_root = os.path.join(output_dir, "neurcross_logs")
else:
    processed_mesh_dir = os.path.join(output_dir, "extract_meshes")
    crossfield_root = output_dir

quad_root = os.path.join(output_dir, "quad_meshes_by_iter")
os.makedirs(quad_root, exist_ok=True)

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
success = 0
tried = 0

for basename in sorted(requested):
    if method == "pipeline":
        crossfield_dir = os.path.join(crossfield_root, basename, "save_crossField")
        feature_path = None
    else:
        crossfield_dir = os.path.join(crossfield_root, basename, "save_crossField")
        feature_path = None

    candidates = [
        os.path.join(processed_mesh_dir, f"{basename}.obj"),
        os.path.join(input_dir, f"{basename}.obj"),
        os.path.join(input_dir, f"{basename}.ply"),
        os.path.join(input_dir, f"{basename}.off"),
        os.path.join(input_dir, f"{basename}.stl"),
    ]
    mesh_path = next((p for p in candidates if os.path.isfile(p)), None)

    print()
    print(f"[Extract Existing Iters] {basename}")
    if mesh_path is None:
        failures.append((basename, "mesh", "missing mesh input"))
        print("  missing mesh input")
        continue
    if method == "pipeline":
        try:
            feature_path = run_pipeline.find_feature_file(feature_dir, basename)
        except FileNotFoundError as exc:
            failures.append((basename, "feature", str(exc)))
            print(f"  missing feature file: {exc}")
            continue
    if not os.path.isdir(crossfield_dir):
        failures.append((basename, "crossfield_dir", f"missing dir: {crossfield_dir}"))
        print(f"  missing dir: {crossfield_dir}")
        continue

    selected = []
    for cf in glob.glob(os.path.join(crossfield_dir, "*_iter_*.txt")):
        m = re.search(r"_iter_(\d+)\.txt$", os.path.basename(cf))
        if not m:
            continue
        it = int(m.group(1))
        if iter_filter is not None and it not in iter_filter:
            continue
        selected.append((it, cf))
    selected.sort(key=lambda x: x[0])

    if not selected:
        print("  no iter files match filter")
        continue
    print(f"  selected iters: {[it for it, _ in selected]}")

    model_out_dir = os.path.join(quad_root, basename)
    os.makedirs(model_out_dir, exist_ok=True)

    for it, cf in selected:
        tried += 1
        out_obj = os.path.join(model_out_dir, f"{basename}_iter_{it}_quad.obj")
        if bool(int(os.environ["SKIP_EXISTING_QUADS"])) and os.path.isfile(out_obj) and os.path.getsize(out_obj) > 0:
            print(f"    iter={it}: skip existing")
            continue
        try:
            run_pipeline.run_chunked_quad_extract(mesh_path, feature_path, None, cf, out_obj, args)
            if os.path.isfile(out_obj) and os.path.getsize(out_obj) > 0:
                success += 1
                print(f"    iter={it}: OK -> {out_obj}")
            else:
                failures.append((basename, str(it), "empty output"))
                print(f"    iter={it}: empty output")
        except Exception as exc:  # noqa: BLE001
            failures.append((basename, str(it), str(exc)))
            print(f"    iter={it}: failed: {exc}")

print()
print("=== SUMMARY ===")
print(f"tried={tried}, success={success}, failed={len(failures)}")
if failures:
    for item in failures:
        print(f"  - {item[0]} [{item[1]}]: {item[2]}")
PY
