#!/usr/bin/env python
"""
Compare baseline vs. our method on quad mesh evaluation metrics.

End-to-end workflow:
    1. Locate cross-field results for each method.
    2. Extract quad meshes (if not already done) via the C++ tool.
    3. Run all available metrics (geometric; semantic if GT labels provided).
    4. Print a side-by-side comparison table.

Usage
-----
    # Minimal — geometry-only comparison (no GT labels needed):
    python -m eval.compare \
        --orig_dir input/ \
        --baseline_log_root pipeline_output/baseline_logs \
        --ours_log_root     pipeline_output/neurcross_logs \
        --output_dir        pipeline_output/eval_compare

    # Full — with GT labels and PartField features:
    python -m eval.compare \
        --orig_dir input/ \
        --baseline_log_root pipeline_output/baseline_logs \
        --ours_log_root     pipeline_output/neurcross_logs \
        --label_dir         input/labels/ \
        --feat_dir          pipeline_output/partfield_features/ \
        --output_dir        pipeline_output/eval_compare

    # Evaluate only specific meshes:
    python -m eval.compare \
        --orig_dir input/ \
        --baseline_log_root pipeline_output/baseline_logs \
        --ours_log_root     pipeline_output/neurcross_logs \
        --meshes armadillo \
        --output_dir        pipeline_output/eval_compare
"""

import os
import sys
import json
import glob
import argparse
import subprocess

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUAD_EXTRACT_BIN = os.path.join(
    PROJECT_ROOT, "quad_extract", "build", "extract_quad_mesh")

ALL_MESH_EXTS = {".obj", ".ply", ".off", ".glb", ".stl"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare baseline vs. ours on quad mesh metrics")

    p.add_argument("--orig_dir", required=True,
                   help="Directory of original triangle meshes")
    p.add_argument("--baseline_log_root", required=True,
                   help="Root of baseline NeurCross logs "
                        "(expects <root>/<name>/save_crossField/)")
    p.add_argument("--ours_log_root", required=True,
                   help="Root of our method NeurCross logs")

    p.add_argument("--meshes", nargs="*", default=None,
                   help="Specific mesh names to evaluate (without extension). "
                        "If omitted, evaluates all meshes present in BOTH logs.")

    p.add_argument("--label_dir", default=None,
                   help="Directory of per-face GT label .npy files "
                        "(enables mIoU and BAE)")
    p.add_argument("--feat_dir", default=None,
                   help="Directory of PartField feature .npy files "
                        "(enables clustering-based mIoU)")

    p.add_argument("--output_dir", default="pipeline_output/eval_compare",
                   help="Where to write extracted quad meshes and results")
    p.add_argument("--gradient_size", type=float, default=30.0,
                   help="MIQ gradient size for quad extraction")
    p.add_argument("--timeout", type=int, default=600,
                   help="Timeout per extraction attempt (seconds)")
    p.add_argument("--retry", action="store_true",
                   help="Auto-retry extraction with multiple gradient sizes")
    p.add_argument("--skip_extract", action="store_true",
                   help="Skip extraction; assume quad meshes already exist")
    p.add_argument("--symmetric_bae", action="store_true",
                   help="Compute symmetric (Chamfer) BAE")
    p.add_argument("--k_range", nargs=2, type=int, default=[2, 20],
                   help="K-Means cluster range for mIoU (default: 2 20)")
    p.add_argument("--output_json", default=None,
                   help="Write full results to JSON")

    return p.parse_args()


# ---------------------------------------------------------------------------
#  Cross-field and quad mesh helpers
# ---------------------------------------------------------------------------

def find_latest_crossfield(cf_dir):
    files = glob.glob(os.path.join(cf_dir, "*_iter_*.txt"))
    if not files:
        return None
    def _iter_num(f):
        base = os.path.splitext(os.path.basename(f))[0]
        return int(base.rsplit("_iter_", 1)[1])
    return max(files, key=_iter_num)


def extract_quad_mesh(input_mesh, crossfield_txt, output_obj,
                      gradient_size=30.0, timeout=600, retry=False):
    """Run the C++ extraction tool. Returns output path on success, None on failure."""
    if not os.path.isfile(QUAD_EXTRACT_BIN):
        print(f"  ERROR: extraction binary not found: {QUAD_EXTRACT_BIN}")
        return None

    os.makedirs(os.path.dirname(os.path.abspath(output_obj)), exist_ok=True)

    gradient_sizes = [gradient_size]
    if retry:
        for gs in [50.0, 15.0, 80.0]:
            if gs != gradient_size:
                gradient_sizes.append(gs)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        "/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", ""))

    for gs in gradient_sizes:
        cmd = [
            QUAD_EXTRACT_BIN,
            os.path.abspath(input_mesh),
            os.path.abspath(crossfield_txt),
            os.path.abspath(output_obj),
            str(gs),
        ]
        try:
            print(f"    Extracting (gs={gs}) ...")
            subprocess.run(cmd, env=env, check=True, timeout=timeout,
                           capture_output=True)
            if os.path.isfile(output_obj) and os.path.getsize(output_obj) > 0:
                return output_obj
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT (gs={gs})")
        except subprocess.CalledProcessError as e:
            print(f"    FAILED (gs={gs}): return code {e.returncode}")

    return None


def find_mesh_file(orig_dir, name):
    for ext in ALL_MESH_EXTS:
        p = os.path.join(orig_dir, name + ext)
        if os.path.isfile(p):
            return p
    return None


def find_label_file(label_dir, name):
    if label_dir is None:
        return None
    for suffix in ["", "_labels", "_label", "_seg"]:
        p = os.path.join(label_dir, name + suffix + ".npy")
        if os.path.isfile(p):
            return p
    return None


def find_feat_file(feat_dir, name):
    if feat_dir is None:
        return None
    patterns = [
        f"part_feat_{name}_0_batch.npy",
        f"part_feat_{name}_0.npy",
        f"{name}.npy",
    ]
    for pat in patterns:
        p = os.path.join(feat_dir, pat)
        if os.path.isfile(p):
            return p
    return None


# ---------------------------------------------------------------------------
#  Discover meshes present in both log roots
# ---------------------------------------------------------------------------

def discover_meshes(baseline_root, ours_root, explicit=None):
    baseline_names = set()
    if os.path.isdir(baseline_root):
        for d in os.listdir(baseline_root):
            cf_dir = os.path.join(baseline_root, d, "save_crossField")
            if os.path.isdir(cf_dir):
                baseline_names.add(d)

    ours_names = set()
    if os.path.isdir(ours_root):
        for d in os.listdir(ours_root):
            cf_dir = os.path.join(ours_root, d, "save_crossField")
            if os.path.isdir(cf_dir):
                ours_names.add(d)

    if explicit:
        names = [n for n in explicit if n in baseline_names and n in ours_names]
        skipped = [n for n in explicit
                   if n not in baseline_names or n not in ours_names]
        if skipped:
            print(f"  WARNING: skipping {skipped} (not found in both logs)")
    else:
        names = sorted(baseline_names & ours_names)

    return names


# ---------------------------------------------------------------------------
#  Run evaluation on a single quad mesh
# ---------------------------------------------------------------------------

def evaluate_quad_mesh(quad_path, orig_path=None, label_path=None,
                       feat_path=None, gt_labels=None,
                       k_range=(2, 20), symmetric_bae=False):
    """Return a dict of metric results.

    Semantic metrics (mIoU, BAE) are computed when *any* of the following
    label sources is available (checked in order of priority):

        1. ``gt_labels`` array passed directly (pre-computed pseudo-GT).
        2. ``label_path`` .npy file with human-annotated per-face labels.
        3. ``feat_path`` .npy PartField features -> auto-cluster into labels.
    """
    import trimesh
    from .angle_distortion import compute_angle_distortion, _load_quad_faces_from_obj
    from .jacobian_ratio import compute_jacobian_ratio
    from .part_miou import compute_class_agnostic_miou, _transfer_labels
    from .boundary_alignment import compute_boundary_alignment_error

    quad_faces = _load_quad_faces_from_obj(quad_path)
    if quad_faces is None or len(quad_faces) == 0:
        print(f"    WARNING: no quad faces in {quad_path}")
        return None

    quad_mesh = trimesh.load(quad_path, process=False)
    verts = np.asarray(quad_mesh.vertices, dtype=np.float64)

    result = {}

    ad = compute_angle_distortion(verts, quad_faces)
    result["AD_mean_deg"] = ad["mean_deviation_deg"]
    result["AD_max_deg"] = ad["max_deviation_deg"]

    jr = compute_jacobian_ratio(verts, quad_faces)
    result["JR_mean"] = jr["mean_jr"]
    result["JR_min"] = jr["min_jr"]
    result["SJ_mean"] = jr["mean_scaled_jacobian"]

    result["n_quads"] = len(quad_faces)
    result["n_verts"] = len(verts)

    has_labels = gt_labels is not None or label_path is not None or feat_path is not None
    if orig_path and has_labels:
        orig_mesh = trimesh.load(orig_path, process=False)
        orig_centers = np.asarray(orig_mesh.triangles_center)
        quad_centers = verts[quad_faces].mean(axis=1)

        if gt_labels is None and label_path is not None:
            gt_labels = np.load(label_path).astype(np.int64)
        if gt_labels is None and feat_path is not None:
            from .label_utils import generate_labels_from_features
            gt_labels, auto_k = generate_labels_from_features(
                feat_path, k_range=k_range)
            result["gt_k"] = auto_k
            print(f"    Auto-generated pseudo-GT labels: K={auto_k}")

        if gt_labels is not None:
            gt_on_quad = _transfer_labels(orig_centers, gt_labels, quad_centers)

            pred_feats = None
            if feat_path is not None:
                from .label_utils import transfer_features_to_quad
                orig_feats = np.load(feat_path)
                pred_feats = transfer_features_to_quad(
                    orig_centers, orig_feats, quad_centers)

            miou_res = compute_class_agnostic_miou(
                gt_labels=gt_on_quad,
                pred_features=pred_feats,
                k_range=tuple(k_range),
            )
            result["mIoU"] = miou_res["miou"]
            result["mIoU_k"] = miou_res["best_k"]

            bae = compute_boundary_alignment_error(
                orig_vertices=orig_mesh.vertices,
                orig_faces=orig_mesh.faces,
                orig_face_labels=gt_labels,
                quad_vertices=verts,
                quad_faces=quad_faces,
                symmetric=symmetric_bae,
            )
            result["BAE"] = bae["bae"]
            result["n_boundary_edges"] = bae["n_boundary_edges"]
            if symmetric_bae and "bae_chamfer" in bae:
                result["BAE_chamfer"] = bae["bae_chamfer"]

    return result


# ---------------------------------------------------------------------------
#  Pretty-print comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(all_results):
    """Print a formatted comparison table."""
    metric_keys = ["AD_mean_deg", "AD_max_deg", "JR_mean", "JR_min",
                   "SJ_mean", "n_quads", "mIoU", "BAE"]
    metric_labels = {
        "AD_mean_deg": "Angle Dist. mean (deg)",
        "AD_max_deg":  "Angle Dist. max  (deg)",
        "JR_mean":     "Jacobian Ratio mean",
        "JR_min":      "Jacobian Ratio min",
        "SJ_mean":     "Scaled Jacobian mean",
        "n_quads":     "Num quads",
        "mIoU":        "mIoU",
        "BAE":         "Boundary Align. Err.",
    }
    better_lower = {"AD_mean_deg", "AD_max_deg", "BAE"}
    better_higher = {"JR_mean", "JR_min", "SJ_mean", "mIoU"}

    for mesh_name, methods in all_results.items():
        print(f"\n{'='*70}")
        print(f"  Mesh: {mesh_name}")
        print(f"{'='*70}")

        bl = methods.get("baseline")
        ours = methods.get("ours")

        header = f"  {'Metric':<28s} {'Baseline':>12s} {'Ours':>12s} {'Better':>8s}"
        print(header)
        print("  " + "-" * 66)

        for key in metric_keys:
            label = metric_labels.get(key, key)
            v_bl = bl.get(key) if bl else None
            v_ours = ours.get(key) if ours else None

            s_bl = _fmt(v_bl)
            s_ours = _fmt(v_ours)

            marker = ""
            if v_bl is not None and v_ours is not None:
                if key in better_lower:
                    marker = "<- Ours" if v_ours < v_bl else ("<- BL" if v_bl < v_ours else "  tie")
                elif key in better_higher:
                    marker = "<- Ours" if v_ours > v_bl else ("<- BL" if v_bl > v_ours else "  tie")

            print(f"  {label:<28s} {s_bl:>12s} {s_ours:>12s} {marker:>8s}")

    # Aggregate across all meshes
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  AGGREGATE (mean over {len(all_results)} meshes)")
        print(f"{'='*70}")
        header = f"  {'Metric':<28s} {'Baseline':>12s} {'Ours':>12s} {'Better':>8s}"
        print(header)
        print("  " + "-" * 66)

        for key in metric_keys:
            if key == "n_quads":
                continue
            label = metric_labels.get(key, key)
            bl_vals = [r["baseline"][key] for r in all_results.values()
                       if r.get("baseline") and r["baseline"].get(key) is not None]
            ours_vals = [r["ours"][key] for r in all_results.values()
                         if r.get("ours") and r["ours"].get(key) is not None]

            s_bl = _fmt(np.mean(bl_vals)) if bl_vals else "N/A"
            s_ours = _fmt(np.mean(ours_vals)) if ours_vals else "N/A"

            marker = ""
            if bl_vals and ours_vals:
                m_bl, m_ours = np.mean(bl_vals), np.mean(ours_vals)
                if key in better_lower:
                    marker = "<- Ours" if m_ours < m_bl else ("<- BL" if m_bl < m_ours else "  tie")
                elif key in better_higher:
                    marker = "<- Ours" if m_ours > m_bl else ("<- BL" if m_bl > m_ours else "  tie")

            print(f"  {label:<28s} {s_bl:>12s} {s_ours:>12s} {marker:>8s}")
        print("=" * 70)


def _fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    orig_dir = os.path.abspath(args.orig_dir)
    bl_root = os.path.abspath(args.baseline_log_root)
    ours_root = os.path.abspath(args.ours_log_root)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    mesh_names = discover_meshes(bl_root, ours_root, args.meshes)
    if not mesh_names:
        sys.exit("No meshes found in both baseline and ours log roots.")

    print(f"Meshes to evaluate: {mesh_names}\n")

    bl_quad_dir = os.path.join(out_dir, "baseline_quads")
    ours_quad_dir = os.path.join(out_dir, "ours_quads")
    os.makedirs(bl_quad_dir, exist_ok=True)
    os.makedirs(ours_quad_dir, exist_ok=True)

    all_results = {}

    for name in mesh_names:
        print(f"\n{'#'*60}")
        print(f"  Processing: {name}")
        print(f"{'#'*60}")

        # Prefer the decimated (processed) mesh whose face count matches
        # the cross-field, then fall back to the raw input mesh.
        processed_candidates = [
            os.path.join(os.path.dirname(out_dir), "processed_meshes"),
            os.path.join(orig_dir, "..", "pipeline_output", "processed_meshes"),
            os.path.join(orig_dir, "..", "processed_meshes"),
        ]
        orig_path = None
        for cand_dir in processed_candidates:
            orig_path = find_mesh_file(cand_dir, name)
            if orig_path is not None:
                break
        if orig_path is None:
            orig_path = find_mesh_file(orig_dir, name)
        if orig_path is None:
            print(f"  WARNING: original mesh not found for {name}, skipping")
            continue
        print(f"  Input mesh: {orig_path}")

        label_path = find_label_file(args.label_dir, name)
        feat_path = find_feat_file(args.feat_dir, name)

        gt_labels_shared = None
        if label_path is not None:
            gt_labels_shared = np.load(label_path).astype(np.int64)
            print(f"  GT labels (annotated): {label_path}")
        elif feat_path is not None:
            from .label_utils import generate_labels_from_features
            gt_labels_shared, auto_k = generate_labels_from_features(
                feat_path, k_range=tuple(args.k_range))
            label_save = os.path.join(out_dir, f"{name}_pseudo_gt_k{auto_k}.npy")
            np.save(label_save, gt_labels_shared)
            print(f"  Pseudo-GT labels: K={auto_k} (from PartField features)")

        methods = {}

        for method_name, log_root, quad_dir in [
            ("baseline", bl_root, bl_quad_dir),
            ("ours", ours_root, ours_quad_dir),
        ]:
            print(f"\n  [{method_name.upper()}]")
            cf_dir = os.path.join(log_root, name, "save_crossField")
            quad_path = os.path.join(quad_dir, f"{name}_quad.obj")

            if args.skip_extract and os.path.isfile(quad_path):
                print(f"    Using existing quad mesh: {quad_path}")
            else:
                cf_txt = find_latest_crossfield(cf_dir)
                if cf_txt is None:
                    print(f"    No cross-field found in {cf_dir}")
                    continue
                print(f"    Cross-field: {os.path.basename(cf_txt)}")

                result_path = extract_quad_mesh(
                    orig_path, cf_txt, quad_path,
                    gradient_size=args.gradient_size,
                    timeout=args.timeout,
                    retry=args.retry,
                )
                if result_path is None:
                    print(f"    Quad extraction FAILED for {method_name}/{name}")
                    continue
                print(f"    Quad mesh -> {quad_path}")

            r = evaluate_quad_mesh(
                quad_path=quad_path,
                orig_path=orig_path,
                label_path=label_path,
                feat_path=feat_path,
                gt_labels=gt_labels_shared,
                k_range=args.k_range,
                symmetric_bae=args.symmetric_bae,
            )
            if r is not None:
                methods[method_name] = r

        if methods:
            all_results[name] = methods

    if not all_results:
        print("\nNo results to compare.")
        sys.exit(1)

    print_comparison_table(all_results)

    if args.output_json:
        serializable = {}
        for mesh_name, methods in all_results.items():
            serializable[mesh_name] = {}
            for meth, vals in methods.items():
                serializable[mesh_name][meth] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in vals.items()
                }
        json_path = args.output_json
        if not os.path.isabs(json_path):
            json_path = os.path.join(out_dir, json_path)
        with open(json_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nFull results written to {json_path}")

    summary_path = os.path.join(out_dir, "comparison_results.json")
    serializable = {}
    for mesh_name, methods in all_results.items():
        serializable[mesh_name] = {}
        for meth, vals in methods.items():
            serializable[mesh_name][meth] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in vals.items()
            }
    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Comparison saved to {summary_path}")


if __name__ == "__main__":
    main()
