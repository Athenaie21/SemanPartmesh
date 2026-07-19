#!/usr/bin/env bash
set -euo pipefail

# ==========================================================================
#  Batch run: reconstruction dataset  (instruction-guided quad meshing)
#
#  Usage:
#    tmux new -s recon
#    bash scripts/run_reconstruction_batch.sh
# ==========================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/instruction_guidance/r1.0.1/reconstruction}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/reconstruction/reconstruction_batch_output}"
GPU_ID="${GPU_ID:-0}"

# --- Training parameters ---
N_SAMPLES=5000
AUTO_NS=1
NS_CANDIDATES="${NS_CANDIDATES:-3000 5000 8000 10000}"
N_POINTS=5000
NUM_EPOCHS=1
LR="5e-5"
GRADIENT_SIZE=30
MIN_FACES=50          # skip meshes with fewer faces (NaN-prone)
MAX_FACES=5000        # decimate meshes larger than this

PYTHON_NC="${PYTHON_NC:-python}"
PIPELINE="${ROOT_DIR}/run_pipeline.py"
LOGFILE="${OUTPUT_ROOT}/batch_run.log"

mkdir -p "${OUTPUT_ROOT}"

run_pipeline_once() {
    local mesh_path="$1"
    local sample_output="$2"
    local ns="$3"
    local run_log="$4"

    (cd "${ROOT_DIR}" && "${PYTHON_NC}" "${PIPELINE}" \
        --input_mesh "${mesh_path}" \
        --output_dir "${sample_output}" \
        --no_timestamp \
        --guidance_mode instruction \
        --instruction_dataset_root "${DATASET_ROOT}" \
        --n_samples "${ns}" \
        --n_points "${N_POINTS}" \
        --num_epochs "${NUM_EPOCHS}" \
        --lr "${LR}" \
        --gradient_size "${GRADIENT_SIZE}" \
        --max_faces "${MAX_FACES}" \
        --gpu_id "${GPU_ID}" \
        >> "${run_log}" 2>&1)
}

score_quad_quality() {
    local quad_path="$1"
    local mesh_path="$2"
    (cd "${ROOT_DIR}" && "${PYTHON_NC}" - "${quad_path}" "${mesh_path}" <<'PYEOF'
import json
import math
import sys
from eval.evaluate import evaluate_single

quad_path = sys.argv[1]
mesh_path = sys.argv[2]
r = evaluate_single(quad_path, orig_mesh_path=mesh_path)
if r is None:
    raise SystemExit(2)

ad = float(r["angle_distortion_mean_deg"])
jr_min = float(r["jacobian_ratio_min"])
sing = int(r["total_singularities"])

# Lower is better.
score = ad + 0.02 * sing + 20.0 * max(0.0, 0.2 - jr_min)
print(json.dumps({
    "score": score,
    "ad_mean": ad,
    "jr_min": jr_min,
    "singularities": sing
}))
PYEOF
)
}

# ---- Build sample list (only paired OBJ+JSON, sorted) ----
SAMPLE_LIST="${OUTPUT_ROOT}/sample_list.txt"
if [ ! -f "${SAMPLE_LIST}" ]; then
    echo "[init] Building paired sample list ..."
    python3 - "${DATASET_ROOT}" "${SAMPLE_LIST}" "${MIN_FACES}" <<'PYEOF'
import sys, os, glob, trimesh, numpy as np

root, outpath, min_faces = sys.argv[1], sys.argv[2], int(sys.argv[3])
jsons = set(os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(root, "*.json")))
objs  = set(os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(root, "*.obj")))
paired = sorted(jsons & objs)

kept = []
skipped_small = 0
for name in paired:
    try:
        m = trimesh.load_mesh(os.path.join(root, f"{name}.obj"), process=False)
        nf = len(m.faces)
        if nf < min_faces:
            skipped_small += 1
            continue
        kept.append((name, nf))
    except Exception as e:
        print(f"  skip {name}: {e}")

kept.sort(key=lambda x: x[1])
with open(outpath, "w") as f:
    for name, nf in kept:
        f.write(f"{name}\t{nf}\n")
print(f"Paired: {len(paired)}, kept (>={min_faces} faces): {len(kept)}, skipped small: {skipped_small}")
PYEOF
    echo "[init] Sample list saved: ${SAMPLE_LIST}"
fi

TOTAL=$(wc -l < "${SAMPLE_LIST}")
echo "============================================================"
echo "  Reconstruction batch run"
echo "  Samples : ${TOTAL}"
echo "  GPU     : ${GPU_ID}"
echo "  Output  : ${OUTPUT_ROOT}"
echo "  Auto ns : ${AUTO_NS} (candidates: ${NS_CANDIDATES})"
echo "============================================================"

DONE_DIR="${OUTPUT_ROOT}/done_markers"
FAIL_DIR="${OUTPUT_ROOT}/fail_markers"
mkdir -p "${DONE_DIR}" "${FAIL_DIR}"

IDX=0
DONE_CNT=$(ls "${DONE_DIR}" 2>/dev/null | wc -l)
FAIL_CNT=$(ls "${FAIL_DIR}" 2>/dev/null | wc -l)

while IFS=$'\t' read -r SAMPLE_NAME FACE_COUNT; do
    IDX=$((IDX + 1))

    # Skip already completed
    if [ -f "${DONE_DIR}/${SAMPLE_NAME}" ]; then
        continue
    fi

    # Skip previously failed (can be retried by removing fail marker)
    if [ -f "${FAIL_DIR}/${SAMPLE_NAME}" ]; then
        continue
    fi

    DONE_CNT=$(ls "${DONE_DIR}" 2>/dev/null | wc -l)
    FAIL_CNT=$(ls "${FAIL_DIR}" 2>/dev/null | wc -l)
    echo ""
    echo ">>> [${IDX}/${TOTAL}] ${SAMPLE_NAME}  (${FACE_COUNT} faces)  done=${DONE_CNT} fail=${FAIL_CNT}"
    echo "    $(date '+%Y-%m-%d %H:%M:%S')"

    SAMPLE_OUTPUT="${OUTPUT_ROOT}/samples/${SAMPLE_NAME}"
    MESH_PATH="${DATASET_ROOT}/${SAMPLE_NAME}.obj"
    mkdir -p "${SAMPLE_OUTPUT}"

    if [[ "${AUTO_NS}" == "1" ]]; then
        AUTO_DIR="${SAMPLE_OUTPUT}/auto_ns"
        mkdir -p "${AUTO_DIR}"
        CAND_FILE="${AUTO_DIR}/candidates.tsv"
        BEST_FILE="${AUTO_DIR}/best_ns.txt"
        echo -e "ns\tstatus\tscore\tad_mean\tjr_min\tsingularities\tquad_path" > "${CAND_FILE}"

        BEST_NS=""
        BEST_SCORE=""
        BEST_QUAD=""

        for NS in ${NS_CANDIDATES}; do
            RUN_DIR="${SAMPLE_OUTPUT}/ns_${NS}"
            RUN_LOG="${AUTO_DIR}/ns_${NS}.log"
            echo "    [auto-ns] try n_samples=${NS}"
            if run_pipeline_once "${MESH_PATH}" "${RUN_DIR}" "${NS}" "${RUN_LOG}"; then
                QUAD_PATH="${RUN_DIR}/quad_meshes/${SAMPLE_NAME}_quad.obj"
                if [[ -f "${QUAD_PATH}" && -s "${QUAD_PATH}" ]]; then
                    METRICS_JSON="$(score_quad_quality "${QUAD_PATH}" "${MESH_PATH}" 2>/dev/null || true)"
                    if [[ -n "${METRICS_JSON}" ]]; then
                        SCORE="$("${PYTHON_NC}" - "${METRICS_JSON}" <<'PYEOF'
import json,sys
m=json.loads(sys.argv[1]); print(m["score"])
PYEOF
)"
                        AD="$("${PYTHON_NC}" - "${METRICS_JSON}" <<'PYEOF'
import json,sys
m=json.loads(sys.argv[1]); print(m["ad_mean"])
PYEOF
)"
                        JR="$("${PYTHON_NC}" - "${METRICS_JSON}" <<'PYEOF'
import json,sys
m=json.loads(sys.argv[1]); print(m["jr_min"])
PYEOF
)"
                        SING="$("${PYTHON_NC}" - "${METRICS_JSON}" <<'PYEOF'
import json,sys
m=json.loads(sys.argv[1]); print(m["singularities"])
PYEOF
)"
                        echo -e "${NS}\tok\t${SCORE}\t${AD}\t${JR}\t${SING}\t${QUAD_PATH}" >> "${CAND_FILE}"
                        if [[ -z "${BEST_SCORE}" ]]; then
                            BEST_SCORE="${SCORE}"
                            BEST_NS="${NS}"
                            BEST_QUAD="${QUAD_PATH}"
                        else
                            BETTER="$("${PYTHON_NC}" - "${SCORE}" "${BEST_SCORE}" <<'PYEOF'
import sys
cur=float(sys.argv[1]); best=float(sys.argv[2])
print("1" if cur < best else "0")
PYEOF
)"
                            if [[ "${BETTER}" == "1" ]]; then
                                BEST_SCORE="${SCORE}"
                                BEST_NS="${NS}"
                                BEST_QUAD="${QUAD_PATH}"
                            fi
                        fi
                    else
                        echo -e "${NS}\tbad_quad\t-\t-\t-\t-\t${QUAD_PATH}" >> "${CAND_FILE}"
                    fi
                else
                    echo -e "${NS}\tmissing_quad\t-\t-\t-\t-\t-" >> "${CAND_FILE}"
                fi
            else
                echo -e "${NS}\tfailed\t-\t-\t-\t-\t-" >> "${CAND_FILE}"
            fi
        done

        if [[ -n "${BEST_NS}" ]]; then
            echo "${BEST_NS}" > "${BEST_FILE}"
            ln -sfn "${SAMPLE_OUTPUT}/ns_${BEST_NS}" "${SAMPLE_OUTPUT}/best"
            touch "${DONE_DIR}/${SAMPLE_NAME}"
            echo "    OK (best n_samples=${BEST_NS}, score=${BEST_SCORE})"
            echo "    Best quad: ${BEST_QUAD}"
        else
            touch "${FAIL_DIR}/${SAMPLE_NAME}"
            echo "    FAILED (no successful n_samples, see ${AUTO_DIR} and ${LOGFILE})"
        fi
    else
        if run_pipeline_once "${MESH_PATH}" "${SAMPLE_OUTPUT}" "${N_SAMPLES}" "${LOGFILE}"; then
            touch "${DONE_DIR}/${SAMPLE_NAME}"
            echo "    OK"
        else
            touch "${FAIL_DIR}/${SAMPLE_NAME}"
            echo "    FAILED (see ${LOGFILE})"
        fi
    fi

done < "${SAMPLE_LIST}"

DONE_CNT=$(ls "${DONE_DIR}" 2>/dev/null | wc -l)
FAIL_CNT=$(ls "${FAIL_DIR}" 2>/dev/null | wc -l)
echo ""
echo "============================================================"
echo "  BATCH COMPLETE"
echo "  Total: ${TOTAL}  Done: ${DONE_CNT}  Failed: ${FAIL_CNT}"
echo "============================================================"
