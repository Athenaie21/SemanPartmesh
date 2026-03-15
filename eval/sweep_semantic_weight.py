#!/usr/bin/env python
"""
Hyperparameter sweep for the semantic alignment loss weight.

For each candidate weight, this script:
    1. Trains NeurCross with the given semantic weight
    2. Extracts a quad mesh from the cross-field output
    3. Evaluates Angle Distortion and Jacobian Ratio
    4. Collects all results into a comparison table and CSV/JSON

Usage
-----
    # Sweep on cheburashka (fast, ~13k faces):
    python -m eval.sweep_semantic_weight \
        --mesh input/cheburashka.obj \
        --part_feat pipeline_output/partfield_features/part_feat_cheburashka_0_batch.npy \
        --output_dir pipeline_output/sweep_semantic

    # Custom weight list:
    python -m eval.sweep_semantic_weight \
        --mesh input/cheburashka.obj \
        --part_feat pipeline_output/partfield_features/part_feat_cheburashka_0_batch.npy \
        --weights 0 1 5 10 20 50 100 \
        --output_dir pipeline_output/sweep_semantic

    /root/.conda/envs/neurcross/bin/python -m eval.sweep_semantic_weight \
        --mesh input/cheburashka.obj \
        --part_feat pipeline_output/partfield_features/part_feat_cheburashka_0_batch.npy \
        --weights 15 30 500 \
        --output_dir pipeline_output/sweep_semantic2 \
        --gpu_id 1
    # Include baseline (weight=0, no semantic features):
    # weight=0 is always included automatically as the baseline reference.
"""

import os
import sys
import json
import glob
import argparse
import subprocess
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEURCROSS_DIR = os.path.join(PROJECT_ROOT, "NeurCross")
QUAD_EXTRACT_BIN = os.path.join(
    PROJECT_ROOT, "quad_extract", "build", "extract_quad_mesh")


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep semantic loss weight for NeurCross")

    p.add_argument("--mesh", required=True,
                   help="Path to input triangle mesh (.obj)")
    p.add_argument("--part_feat", required=True,
                   help="Path to PartField per-face features (.npy)")
    p.add_argument("--weights", nargs="+", type=float,
                   default=[0, 1, 5, 10, 20, 50, 100],
                   help="Semantic loss weights to sweep (default: 0 1 5 10 20 50 100)")
    p.add_argument("--output_dir", default="pipeline_output/sweep_semantic",
                   help="Root output directory")

    p.add_argument("--python", default=None,
                   help="Python interpreter for NeurCross env "
                        "(default: auto-detect neurcross conda env)")
    p.add_argument("--gpu_id", default="0", help="CUDA_VISIBLE_DEVICES")

    p.add_argument("--n_samples", type=int, default=10000)
    p.add_argument("--n_points", type=int, default=15000)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--base_weights", nargs=6, type=float,
                   default=[7e3, 6e2, 10, 5e1, 30, 3],
                   help="First 6 loss weights [sdf inter theta_hess eikonal "
                        "theta_neigh morse] (default: 7e3 6e2 10 5e1 30 3)")

    p.add_argument("--gradient_size", type=float, default=30.0)
    p.add_argument("--extract_timeout", type=int, default=1200)
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training; use existing cross-field results")
    p.add_argument("--skip_extract", action="store_true",
                   help="Skip extraction; use existing quad meshes")
    p.add_argument("--k_range", nargs=2, type=int, default=[2, 15],
                   help="Cluster K range for label generation and mIoU "
                        "(default: 2 15)")

    return p.parse_args()


def find_neurcross_python():
    candidates = [
        os.path.expanduser("~/.conda/envs/neurcross/bin/python"),
        "/root/.conda/envs/neurcross/bin/python",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return "python"


def find_latest_crossfield(cf_dir):
    files = glob.glob(os.path.join(cf_dir, "*_iter_*.txt"))
    if not files:
        return None
    def _iter_num(f):
        base = os.path.splitext(os.path.basename(f))[0]
        return int(base.rsplit("_iter_", 1)[1])
    return max(files, key=_iter_num)


def run_training(python, mesh_path, feat_path, logdir, base_weights,
                 sem_weight, args):
    """Run NeurCross training with a given semantic weight."""
    all_weights = list(base_weights) + [sem_weight]

    cmd = [
        python, "train_quad_mesh.py",
        "--data_path",      os.path.abspath(mesh_path),
        "--part_feat_path", os.path.abspath(feat_path),
        "--logdir",         os.path.abspath(logdir),
        "--n_samples",      str(args.n_samples),
        "--n_points",       str(args.n_points),
        "--num_epochs",     str(args.num_epochs),
        "--lr",             str(args.lr),
        "--loss_weights",   *[str(w) for w in all_weights],
        "--morse_near",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    cwd = os.path.join(NEURCROSS_DIR, "quad_mesh")
    print(f"    CMD: {' '.join(cmd[:4])} ... --loss_weights {' '.join(str(w) for w in all_weights)}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"    TRAIN FAILED (exit {result.returncode})")
        print(f"    stderr: {result.stderr[-500:]}")
        return False, elapsed

    print(f"    Training done in {elapsed:.0f}s")
    return True, elapsed


def run_extraction(mesh_path, cf_txt, output_obj, gradient_size=30.0,
                   timeout=600):
    """Extract quad mesh from cross-field."""
    if not os.path.isfile(QUAD_EXTRACT_BIN):
        print(f"    ERROR: binary not found: {QUAD_EXTRACT_BIN}")
        return None

    os.makedirs(os.path.dirname(os.path.abspath(output_obj)), exist_ok=True)

    cmd = [
        QUAD_EXTRACT_BIN,
        os.path.abspath(mesh_path),
        os.path.abspath(cf_txt),
        os.path.abspath(output_obj),
        str(gradient_size),
    ]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        "/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", ""))

    t0 = time.time()
    try:
        subprocess.run(cmd, env=env, check=True, timeout=timeout,
                       capture_output=True)
        if os.path.isfile(output_obj) and os.path.getsize(output_obj) > 0:
            print(f"    Extraction done in {time.time()-t0:.0f}s")
            return output_obj
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT ({timeout}s)")
    except subprocess.CalledProcessError as e:
        print(f"    FAILED (exit {e.returncode})")

    return None


def _load_verts_from_obj(path):
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]),
                              float(parts[3])])
    return np.array(verts, dtype=np.float64)


def evaluate_quad(quad_path, orig_mesh_path=None, feat_path=None,
                  gt_labels=None, orig_face_centers=None, orig_features=None,
                  k_range=(2, 15)):
    """Evaluate a quad mesh on all four metrics.

    Geometric metrics (AD, JR) always computed.
    Semantic metrics (mIoU, BAE) computed when orig mesh + features are given.
    """
    sys.path.insert(0, PROJECT_ROOT)
    from eval.angle_distortion import compute_angle_distortion, _load_quad_faces_from_obj
    from eval.jacobian_ratio import compute_jacobian_ratio

    verts = _load_verts_from_obj(quad_path)
    qf = _load_quad_faces_from_obj(quad_path)
    if qf is None or len(qf) == 0:
        return None

    ad = compute_angle_distortion(verts, qf)
    jr = compute_jacobian_ratio(verts, qf)

    result = {
        "n_quads": int(len(qf)),
        "n_verts": int(len(verts)),
        "AD_mean_deg": ad["mean_deviation_deg"],
        "AD_max_deg": ad["max_deviation_deg"],
        "JR_mean": jr["mean_jr"],
        "JR_min": jr["min_jr"],
        "JR_std": jr["std_jr"],
        "SJ_mean": jr["mean_scaled_jacobian"],
    }

    if orig_mesh_path is not None and feat_path is not None:
        import trimesh
        from eval.label_utils import (generate_labels_from_features,
                                      transfer_features_to_quad)
        from eval.part_miou import compute_class_agnostic_miou, _transfer_labels
        from eval.boundary_alignment import compute_boundary_alignment_error

        orig_mesh = trimesh.load(orig_mesh_path, process=False)
        orig_centers = np.asarray(orig_mesh.triangles_center)
        orig_feats = np.load(feat_path)

        if gt_labels is None:
            gt_labels, gt_k = generate_labels_from_features(
                feat_path, k_range=k_range)
            result["gt_k"] = gt_k

        quad_centers = verts[qf].mean(axis=1)

        gt_on_quad = _transfer_labels(orig_centers, gt_labels, quad_centers)
        quad_feats = transfer_features_to_quad(
            orig_centers, orig_feats, quad_centers)

        miou_res = compute_class_agnostic_miou(
            gt_labels=gt_on_quad,
            pred_features=quad_feats,
            k_range=k_range,
        )
        result["mIoU"] = miou_res["miou"]
        result["mIoU_k"] = miou_res["best_k"]

        bae = compute_boundary_alignment_error(
            orig_vertices=orig_mesh.vertices,
            orig_faces=orig_mesh.faces,
            orig_face_labels=gt_labels,
            quad_vertices=verts,
            quad_faces=qf,
        )
        result["BAE"] = bae["bae"]
        result["n_boundary_edges"] = bae["n_boundary_edges"]

    return result


def print_sweep_table(all_results):
    """Print a formatted comparison table across all weights."""
    metric_keys = ["n_quads", "AD_mean_deg", "AD_max_deg",
                   "JR_mean", "JR_min", "SJ_mean", "mIoU", "BAE"]
    metric_labels = {
        "n_quads":     "Num Quads",
        "AD_mean_deg": "Angle Dist. mean (deg)",
        "AD_max_deg":  "Angle Dist. max  (deg)",
        "JR_mean":     "Jacobian Ratio mean",
        "JR_min":      "Jacobian Ratio min",
        "SJ_mean":     "Scaled Jacobian mean",
        "mIoU":        "mIoU",
        "BAE":         "Boundary Align. Err.",
    }

    weights_sorted = sorted(all_results.keys())
    col_width = 12

    print("\n" + "=" * (32 + col_width * len(weights_sorted)))
    print("  Semantic Weight Sweep Results")
    print("=" * (32 + col_width * len(weights_sorted)))

    header = f"  {'Metric':<28s}"
    for w in weights_sorted:
        label = "BL(w=0)" if w == 0 else f"w={w:g}"
        header += f" {label:>{col_width}s}"
    print(header)
    print("  " + "-" * (26 + col_width * len(weights_sorted)))

    better_lower = {"AD_mean_deg", "AD_max_deg", "BAE"}
    better_higher = {"JR_mean", "JR_min", "SJ_mean", "mIoU"}

    for key in metric_keys:
        label = metric_labels.get(key, key)
        row = f"  {label:<28s}"

        vals = {}
        for w in weights_sorted:
            r = all_results[w]
            v = r.get(key) if r else None
            vals[w] = v
            if v is None:
                row += f" {'FAIL':>{col_width}s}"
            elif isinstance(v, float):
                row += f" {v:>{col_width}.4f}"
            else:
                row += f" {v:>{col_width}}"

        numeric = {w: v for w, v in vals.items() if isinstance(v, float)}
        if numeric and key != "n_quads":
            if key in better_lower:
                best_w = min(numeric, key=numeric.get)
            elif key in better_higher:
                best_w = max(numeric, key=numeric.get)
            else:
                best_w = None
            if best_w is not None:
                row += f"  <- w={best_w:g}"

        print(row)

    print("=" * (32 + col_width * len(weights_sorted)))


def main():
    args = parse_args()

    python = args.python or find_neurcross_python()
    mesh_path = os.path.abspath(args.mesh)
    feat_path = os.path.abspath(args.part_feat)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]

    weights = sorted(set(args.weights))
    if 0 not in weights:
        weights = [0] + weights

    print(f"Mesh: {mesh_name}")
    print(f"Feature: {feat_path}")
    print(f"Semantic weights to sweep: {weights}")
    print(f"Output: {out_dir}")

    # Pre-compute GT labels from PartField features (shared across all runs)
    from eval.label_utils import generate_labels_from_features
    print(f"\n[PRE] Generating pseudo-GT labels from PartField features ...")
    gt_labels, gt_k = generate_labels_from_features(
        feat_path, k_range=tuple(args.k_range))
    gt_label_path = os.path.join(out_dir, f"{mesh_name}_gt_labels_k{gt_k}.npy")
    np.save(gt_label_path, gt_labels)
    print(f"  GT labels: K={gt_k}, saved to {gt_label_path}\n")

    all_results = {}

    for sem_w in weights:
        tag = f"w{sem_w:g}"
        print(f"\n{'#'*60}")
        print(f"  Semantic weight = {sem_w:g}  [{tag}]")
        print(f"{'#'*60}")

        trial_logdir = os.path.join(out_dir, "logs", tag)
        os.makedirs(trial_logdir, exist_ok=True)

        mesh_logdir = os.path.join(trial_logdir, mesh_name)
        cf_dir = os.path.join(mesh_logdir, "save_crossField")
        quad_path = os.path.join(out_dir, "quads", f"{mesh_name}_{tag}_quad.obj")

        # --- Training ---
        if not args.skip_train:
            use_feat = feat_path if sem_w > 0 else None
            if use_feat is None:
                train_weights = list(args.base_weights)
            else:
                train_weights = list(args.base_weights)

            print(f"  [TRAIN] logdir={trial_logdir}")
            ok, elapsed = run_training(
                python, mesh_path,
                feat_path if sem_w > 0 else feat_path,
                trial_logdir, args.base_weights, sem_w, args)
            if not ok:
                all_results[sem_w] = None
                continue

        # --- Extraction ---
        if not args.skip_extract:
            cf_txt = find_latest_crossfield(cf_dir)
            if cf_txt is None:
                print(f"  No cross-field found in {cf_dir}")
                all_results[sem_w] = None
                continue

            print(f"  [EXTRACT] {os.path.basename(cf_txt)} -> {tag}_quad.obj")
            result = run_extraction(
                mesh_path, cf_txt, quad_path,
                gradient_size=args.gradient_size,
                timeout=args.extract_timeout)
            if result is None:
                all_results[sem_w] = None
                continue

        # --- Evaluation ---
        if not os.path.isfile(quad_path):
            print(f"  Quad mesh not found: {quad_path}")
            all_results[sem_w] = None
            continue

        print(f"  [EVAL]")
        metrics = evaluate_quad(
            quad_path,
            orig_mesh_path=mesh_path,
            feat_path=feat_path,
            gt_labels=gt_labels,
            k_range=tuple(args.k_range),
        )
        if metrics:
            metrics["semantic_weight"] = sem_w
            metrics["train_time_s"] = elapsed if not args.skip_train else None
        all_results[sem_w] = metrics

        if metrics:
            msg = (f"    AD_mean={metrics['AD_mean_deg']:.3f}°  "
                   f"JR_mean={metrics['JR_mean']:.4f}  "
                   f"SJ_mean={metrics['SJ_mean']:.4f}  "
                   f"quads={metrics['n_quads']}")
            if "mIoU" in metrics:
                msg += f"  mIoU={metrics['mIoU']:.4f}"
            if "BAE" in metrics:
                msg += f"  BAE={metrics['BAE']:.6f}"
            print(msg)

    # --- Summary ---
    valid = {w: r for w, r in all_results.items() if r is not None}
    if valid:
        print_sweep_table(valid)

        json_path = os.path.join(out_dir, f"sweep_{mesh_name}.json")
        serializable = {}
        for w, r in all_results.items():
            serializable[str(w)] = r
        with open(json_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {json_path}")

        csv_path = os.path.join(out_dir, f"sweep_{mesh_name}.csv")
        keys = ["semantic_weight", "n_quads", "n_verts",
                "AD_mean_deg", "AD_max_deg",
                "JR_mean", "JR_min", "JR_std", "SJ_mean",
                "mIoU", "mIoU_k", "BAE", "n_boundary_edges",
                "train_time_s"]
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for w in sorted(all_results.keys()):
                r = all_results[w]
                if r:
                    f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        print(f"CSV saved to {csv_path}")
    else:
        print("\nNo successful results.")


if __name__ == "__main__":
    main()
