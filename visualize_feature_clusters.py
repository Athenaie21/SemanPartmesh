#!/usr/bin/env python
"""
Visualize clustered PartField features by coloring mesh faces.

This script clusters per-face PartField features into discrete labels and
exports a colored mesh where faces in the same cluster share the same color.
It can also transfer the clustered labels or features to a quad mesh and
export a second colored visualization.

Examples
--------
python visualize_feature_clusters.py \
    --mesh input/cheburashka.obj \
    --feat_path pipeline_output/partfield_features/part_feat_cheburashka_0_batch.npy \
    --output cheburashka_clusters.ply

python visualize_feature_clusters.py \
    --mesh input/cheburashka.obj \
    --feat_path pipeline_output/partfield_features/part_feat_cheburashka_0_batch.npy \
    --quad_mesh pipeline_output/neurcross_logs/cheburashka_quad.obj \
    --output cheburashka_clusters.ply \
    --quad_output cheburashka_quad_clusters.ply
"""

import argparse
import os

import numpy as np
import trimesh

from eval.label_utils import cluster_features, transfer_features_to_quad


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster PartField features and export colored meshes.")
    parser.add_argument("--mesh", required=True,
                        help="Original triangle mesh used for PartField features.")
    parser.add_argument("--feat_path", required=True,
                        help="Path to PartField per-face features (.npy).")
    parser.add_argument("--output", required=True,
                        help="Output colored mesh path for the original mesh. Prefer .ply.")
    parser.add_argument("--quad_mesh", default=None,
                        help="Optional quad mesh to visualize with transferred semantics.")
    parser.add_argument("--quad_output", default=None,
                        help="Output colored mesh path for the quad mesh. Prefer .ply.")
    parser.add_argument("--k", type=int, default=None,
                        help="Fixed number of clusters. If omitted, use best silhouette.")
    parser.add_argument("--k_min", type=int, default=2,
                        help="Minimum K for silhouette sweep.")
    parser.add_argument("--k_max", type=int, default=15,
                        help="Maximum K for silhouette sweep.")
    parser.add_argument("--transfer_mode", choices=["labels", "features"], default="labels",
                        help="How to transfer semantics to quad mesh.")
    parser.add_argument("--save_labels", default=None,
                        help="Optional path to save original face labels (.npy).")
    parser.add_argument("--save_quad_labels", default=None,
                        help="Optional path to save quad face labels (.npy).")
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def make_label_colors(labels):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    unique_labels = np.unique(labels)

    rng = np.random.RandomState(42)
    palette = {}
    for label in unique_labels:
        color = rng.randint(32, 256, size=3, dtype=np.uint8)
        palette[int(label)] = np.concatenate([color, np.array([255], dtype=np.uint8)])

    face_colors = np.stack([palette[int(label)] for label in labels], axis=0)
    return face_colors


def export_colored_mesh(mesh, labels, output_path):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(labels) != len(mesh.faces):
        raise ValueError(
            f"Label count ({len(labels)}) does not match face count ({len(mesh.faces)}).")

    ensure_parent_dir(output_path)
    colored_mesh = mesh.copy()
    colored_mesh.visual.face_colors = make_label_colors(labels)
    colored_mesh.export(output_path)


def cluster_from_features(features, k=None, k_range=(2, 15)):
    features = np.asarray(features, dtype=np.float64)
    feat_norm = np.linalg.norm(features, axis=-1, keepdims=True)
    features = features / np.clip(feat_norm, 1e-12, None)

    if k is not None:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        labels = km.fit_predict(features).astype(np.int64)
        return labels, k, None

    result = cluster_features(features, k_range=k_range, method="best_silhouette")
    return result["labels"], result["k"], result["silhouette"]


def transfer_labels_to_mesh(src_centers, src_labels, tgt_centers):
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(src_centers))
    _, idx = tree.query(np.asarray(tgt_centers), k=1)
    return np.asarray(src_labels, dtype=np.int64)[idx]


def main():
    args = parse_args()

    if not os.path.isfile(args.mesh):
        raise FileNotFoundError(f"Mesh not found: {args.mesh}")
    if not os.path.isfile(args.feat_path):
        raise FileNotFoundError(f"Feature file not found: {args.feat_path}")
    if args.quad_mesh is not None and not os.path.isfile(args.quad_mesh):
        raise FileNotFoundError(f"Quad mesh not found: {args.quad_mesh}")
    if args.quad_mesh is not None and args.quad_output is None:
        raise ValueError("--quad_output is required when --quad_mesh is provided.")

    mesh = trimesh.load_mesh(args.mesh, process=False)
    features = np.load(args.feat_path)

    if len(features) != len(mesh.faces):
        raise ValueError(
            f"Feature count ({len(features)}) does not match face count ({len(mesh.faces)}).")

    labels, k_used, silhouette = cluster_from_features(
        features, k=args.k, k_range=(args.k_min, args.k_max))

    export_colored_mesh(mesh, labels, args.output)

    print(f"Original mesh exported to: {args.output}")
    print(f"Clusters: K={k_used}")
    if silhouette is not None:
        print(f"Silhouette: {silhouette:.4f}")

    if args.save_labels is not None:
        ensure_parent_dir(args.save_labels)
        np.save(args.save_labels, labels)
        print(f"Original labels saved to: {args.save_labels}")

    if args.quad_mesh is None:
        return

    quad_mesh = trimesh.load_mesh(args.quad_mesh, process=False)
    orig_centers = np.asarray(mesh.triangles_center)
    quad_centers = np.asarray(quad_mesh.triangles_center)

    if args.transfer_mode == "labels":
        quad_labels = transfer_labels_to_mesh(orig_centers, labels, quad_centers)
    else:
        quad_features = transfer_features_to_quad(orig_centers, features, quad_centers)
        quad_labels, quad_k, quad_silhouette = cluster_from_features(
            quad_features, k=args.k, k_range=(args.k_min, args.k_max))
        print(f"Quad re-clustering: K={quad_k}")
        if quad_silhouette is not None:
            print(f"Quad silhouette: {quad_silhouette:.4f}")

    export_colored_mesh(quad_mesh, quad_labels, args.quad_output)
    print(f"Quad mesh exported to: {args.quad_output}")

    if args.save_quad_labels is not None:
        ensure_parent_dir(args.save_quad_labels)
        np.save(args.save_quad_labels, quad_labels)
        print(f"Quad labels saved to: {args.save_quad_labels}")


if __name__ == "__main__":
    main()
