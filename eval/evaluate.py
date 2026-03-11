#!/usr/bin/env python
"""
Unified evaluation script for semantic quad meshes.

Computes four metrics:
    1. Angle Distortion       – deviation of quad corners from 90 degrees
    2. Jacobian Ratio (JR)    – element quality via scaled Jacobian
    3. Class-agnostic mIoU    – part segmentation accuracy
    4. Boundary Alignment Error (BAE) – quad edge / semantic boundary distance

Usage
-----
    # Evaluate a single quad mesh with GT labels:
    python -m eval.evaluate \
        --quad_mesh  pipeline_output/quad_meshes/armadillo_quad.obj \
        --orig_mesh  input/armadillo.obj \
        --gt_labels  input/armadillo_labels.npy

    # With PartField features (auto-cluster for mIoU):
    python -m eval.evaluate \
        --quad_mesh  pipeline_output/quad_meshes/armadillo_quad.obj \
        --orig_mesh  input/armadillo.obj \
        --gt_labels  input/armadillo_labels.npy \
        --part_features pipeline_output/partfield_features/part_feat_armadillo_0.npy

    # Batch mode — process all *_quad.obj files in a directory:
    python -m eval.evaluate \
        --quad_dir   pipeline_output/quad_meshes/ \
        --orig_dir   input/ \
        --label_dir  input/labels/ \
        --output_csv results.csv
"""

import os
import sys
import json
import argparse

import numpy as np
import trimesh

from .angle_distortion import compute_angle_distortion, _load_quad_faces_from_obj
from .jacobian_ratio import compute_jacobian_ratio
from .part_miou import compute_class_agnostic_miou
from .boundary_alignment import compute_boundary_alignment_error


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate semantic quad meshes on four metrics")

    single = p.add_argument_group("Single mesh mode")
    single.add_argument("--quad_mesh", help="Path to a quad mesh (.obj)")
    single.add_argument("--orig_mesh",
                        help="Path to the original triangle mesh (.obj)")
    single.add_argument("--gt_labels",
                        help="Per-face GT part labels (.npy, int array) "
                             "on the original mesh")

    batch = p.add_argument_group("Batch mode")
    batch.add_argument("--quad_dir",
                       help="Directory of quad meshes (*_quad.obj)")
    batch.add_argument("--orig_dir",
                       help="Directory of original triangle meshes")
    batch.add_argument("--label_dir",
                       help="Directory of per-face GT label .npy files")

    p.add_argument("--part_features", default=None,
                   help="PartField per-face features (.npy) on orig mesh; "
                        "used for clustering-based mIoU (single mode)")
    p.add_argument("--feat_dir", default=None,
                   help="Directory of feature .npy files (batch mode)")
    p.add_argument("--pred_labels", default=None,
                   help="Predicted per-face labels (.npy) on quad mesh "
                        "(single mode). If given, used instead of clustering.")
    p.add_argument("--k_range", nargs=2, type=int, default=[2, 20],
                   help="Min and max cluster count for K-Means sweep "
                        "(default: 2 20)")
    p.add_argument("--output_csv", default=None,
                   help="Write per-mesh results to CSV (batch mode)")
    p.add_argument("--output_json", default=None,
                   help="Write results to JSON")
    p.add_argument("--symmetric_bae", action="store_true",
                   help="Compute symmetric (Chamfer) BAE")

    return p.parse_args()


def _quad_face_centers(vertices, quad_faces):
    """Return (F, 3) centers of quad faces."""
    return vertices[quad_faces].mean(axis=1)


def _tri_face_centers(mesh):
    """Return (F, 3) triangle face centers."""
    return mesh.triangles_center


def evaluate_single(
    quad_mesh_path,
    orig_mesh_path=None,
    gt_label_path=None,
    part_feat_path=None,
    pred_label_path=None,
    k_range=(2, 20),
    symmetric_bae=False,
):
    """Evaluate a single quad mesh.

    Returns a dict with all metric results.
    """
    quad_mesh = trimesh.load(quad_mesh_path, process=False)
    quad_faces = _load_quad_faces_from_obj(quad_mesh_path)

    if quad_faces is None or len(quad_faces) == 0:
        print(f"  WARNING: No quad faces in {quad_mesh_path}, skipping.")
        return None

    verts = np.asarray(quad_mesh.vertices, dtype=np.float64)

    results = {"mesh": os.path.basename(quad_mesh_path)}

    # --- Angle Distortion ---
    ad = compute_angle_distortion(verts, quad_faces)
    results["angle_distortion_mean_deg"] = ad["mean_deviation_deg"]
    results["angle_distortion_max_deg"] = ad["max_deviation_deg"]

    # --- Jacobian Ratio ---
    jr = compute_jacobian_ratio(verts, quad_faces)
    results["jacobian_ratio_mean"] = jr["mean_jr"]
    results["jacobian_ratio_min"] = jr["min_jr"]
    results["jacobian_ratio_std"] = jr["std_jr"]
    results["scaled_jacobian_mean"] = jr["mean_scaled_jacobian"]

    # --- Semantic metrics (need original mesh + labels) ---
    if orig_mesh_path is not None and gt_label_path is not None:
        orig_mesh = trimesh.load(orig_mesh_path, process=False)
        gt_labels = np.load(gt_label_path).astype(np.int64)

        if len(gt_labels) != len(orig_mesh.faces):
            print(f"  WARNING: label count ({len(gt_labels)}) != "
                  f"face count ({len(orig_mesh.faces)})")

        orig_centers = _tri_face_centers(orig_mesh)
        quad_centers = _quad_face_centers(verts, quad_faces)

        # Transfer GT labels to quad faces
        from .part_miou import _transfer_labels
        gt_on_quad = _transfer_labels(orig_centers, gt_labels, quad_centers)

        # Predicted labels
        pred_on_quad = None
        pred_feats = None
        if pred_label_path is not None:
            pred_on_quad = np.load(pred_label_path).astype(np.int64)
        elif part_feat_path is not None:
            orig_feats = np.load(part_feat_path)
            from scipy.spatial import cKDTree
            tree = cKDTree(orig_centers)
            _, idx = tree.query(quad_centers, k=1)
            pred_feats = orig_feats[idx]

        miou_result = compute_class_agnostic_miou(
            gt_labels=gt_on_quad,
            pred_labels=pred_on_quad,
            pred_features=pred_feats,
            k_range=tuple(k_range),
        )
        results["miou"] = miou_result["miou"]
        results["miou_best_k"] = miou_result["best_k"]

        # --- Boundary Alignment Error ---
        bae = compute_boundary_alignment_error(
            orig_vertices=orig_mesh.vertices,
            orig_faces=orig_mesh.faces,
            orig_face_labels=gt_labels,
            quad_vertices=verts,
            quad_faces=quad_faces,
            symmetric=symmetric_bae,
        )
        results["bae"] = bae["bae"]
        results["n_boundary_edges"] = bae["n_boundary_edges"]
        results["n_quad_edges"] = bae["n_quad_edges"]
        if symmetric_bae:
            results["bae_reverse"] = bae.get("bae_reverse", None)
            results["bae_chamfer"] = bae.get("bae_chamfer", None)
    else:
        results["miou"] = None
        results["bae"] = None

    return results


def _find_matching_file(directory, basename, extensions):
    """Find a file in *directory* whose stem matches *basename*."""
    for ext in extensions:
        candidate = os.path.join(directory, basename + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def evaluate_batch(args):
    """Evaluate all quad meshes in a directory."""
    quad_dir = os.path.abspath(args.quad_dir)
    quad_files = sorted(
        f for f in os.listdir(quad_dir) if f.endswith(".obj"))

    if not quad_files:
        sys.exit(f"No .obj files found in {quad_dir}")

    all_results = []
    for qf in quad_files:
        basename = os.path.splitext(qf)[0]
        stem = basename.replace("_quad", "")

        print(f"\n--- Evaluating: {qf} ---")

        quad_path = os.path.join(quad_dir, qf)
        orig_path = None
        label_path = None
        feat_path = None

        if args.orig_dir:
            orig_path = _find_matching_file(
                args.orig_dir, stem, [".obj", ".ply", ".off", ".glb"])
        if args.label_dir:
            label_path = _find_matching_file(
                args.label_dir, stem, [".npy"])
            if label_path is None:
                label_path = _find_matching_file(
                    args.label_dir, stem + "_labels", [".npy"])
        if args.feat_dir:
            feat_path = _find_matching_file(
                args.feat_dir, "part_feat_" + stem + "_0", [".npy"])
            if feat_path is None:
                feat_path = _find_matching_file(
                    args.feat_dir, stem, [".npy"])

        r = evaluate_single(
            quad_mesh_path=quad_path,
            orig_mesh_path=orig_path,
            gt_label_path=label_path,
            part_feat_path=feat_path,
            k_range=args.k_range,
            symmetric_bae=args.symmetric_bae,
        )
        if r is not None:
            all_results.append(r)
            _print_result(r)

    return all_results


def _print_result(r):
    """Pretty-print a single result dict."""
    print(f"  Angle Distortion : mean = {r['angle_distortion_mean_deg']:.3f} deg"
          f"  |  max = {r['angle_distortion_max_deg']:.3f} deg")
    print(f"  Jacobian Ratio   : mean = {r['jacobian_ratio_mean']:.4f}"
          f"  |  min = {r['jacobian_ratio_min']:.4f}"
          f"  |  std = {r['jacobian_ratio_std']:.4f}")
    if r.get("miou") is not None:
        print(f"  mIoU             : {r['miou']:.4f}"
              f"  (k={r.get('miou_best_k', 'N/A')})")
    if r.get("bae") is not None:
        print(f"  BAE              : {r['bae']:.6f}"
              f"  (boundary edges={r['n_boundary_edges']}, "
              f"quad edges={r['n_quad_edges']})")
        if r.get("bae_chamfer") is not None:
            print(f"  BAE (chamfer)    : {r['bae_chamfer']:.6f}")


def _write_csv(results, path):
    """Write results list to CSV."""
    if not results:
        return
    keys = list(results[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in results:
            vals = [str(r.get(k, "")) for k in keys]
            f.write(",".join(vals) + "\n")
    print(f"\nResults written to {path}")


def _aggregate(results):
    """Print aggregate statistics across all meshes."""
    numeric_keys = [
        "angle_distortion_mean_deg", "angle_distortion_max_deg",
        "jacobian_ratio_mean", "jacobian_ratio_min",
        "miou", "bae",
    ]
    print("\n" + "=" * 60)
    print("  Aggregate Results")
    print("=" * 60)
    for key in numeric_keys:
        vals = [r[key] for r in results if r.get(key) is not None]
        if vals:
            arr = np.array(vals)
            print(f"  {key:35s}  mean={arr.mean():.4f}  std={arr.std():.4f}")
    print("=" * 60)


def main():
    args = parse_args()

    if args.quad_mesh:
        r = evaluate_single(
            quad_mesh_path=args.quad_mesh,
            orig_mesh_path=args.orig_mesh,
            gt_label_path=args.gt_labels,
            part_feat_path=args.part_features,
            pred_label_path=args.pred_labels,
            k_range=args.k_range,
            symmetric_bae=args.symmetric_bae,
        )
        if r is not None:
            _print_result(r)
            if args.output_json:
                serializable = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in r.items()
                }
                with open(args.output_json, "w") as f:
                    json.dump(serializable, f, indent=2)
                print(f"\nResults written to {args.output_json}")

    elif args.quad_dir:
        results = evaluate_batch(args)
        if results:
            _aggregate(results)
            if args.output_csv:
                _write_csv(results, args.output_csv)
            if args.output_json:
                serializable = []
                for r in results:
                    serializable.append({
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in r.items()
                    })
                with open(args.output_json, "w") as f:
                    json.dump(serializable, f, indent=2)
                print(f"Results written to {args.output_json}")
    else:
        print("Provide --quad_mesh (single) or --quad_dir (batch).")
        sys.exit(1)


if __name__ == "__main__":
    main()
