#!/usr/bin/env python
"""
Build a per-face complexity map and a relative size hint.

The script combines three interpretable signals:
    1. geometry: local normal variation / curvature proxy
    2. semantic: PartField feature boundary strength + pseudo-label boundaries
    3. field: local cross-field variation after transport to neighboring faces

Outputs:
    * colored mesh visualizations (.ply recommended)
    * a .npz archive with per-face arrays
    * a summary .json with statistics and effective weights

Examples
--------
python build_complexity_map.py \
    --mesh input/cheburashka.obj \
    --feat_path pipeline_output/partfield_features/part_feat_cheburashka_0_batch.npy \
    --crossfield pipeline_output/neurcross_logs/cheburashka/save_crossField/cheburashka_iter_9999.txt \
    --output output/cheburashka_complexity.ply \
    --component_dir output/cheburashka_components \
    --save_npz output/cheburashka_complexity.npz
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
NEURCROSS_ROOT = os.path.join(PROJECT_ROOT, "NeurCross")
if NEURCROSS_ROOT not in sys.path:
    sys.path.append(NEURCROSS_ROOT)

from utils.semantic_utils import compute_semantic_gradient
from eval.boundary_alignment import _find_boundary_edges_from_labels
from eval.label_utils import cluster_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a per-face complexity map and relative size hint.")
    parser.add_argument("--mesh", required=True,
                        help="Input triangle mesh.")
    parser.add_argument("--feat_path", default=None,
                        help="Optional PartField per-face features (.npy).")
    parser.add_argument("--crossfield", default=None,
                        help="Optional cross-field text file with 6 columns per face.")
    parser.add_argument("--output", required=True,
                        help="Output colored mesh for total complexity. Prefer .ply.")
    parser.add_argument("--component_dir", default=None,
                        help="Optional directory to export component visualizations.")
    parser.add_argument("--save_npz", default=None,
                        help="Optional .npz path to save per-face arrays.")
    parser.add_argument("--save_size_hint_txt", default=None,
                        help="Optional text path to save per-face size hints, one value per line.")
    parser.add_argument("--save_density_txt", default=None,
                        help="Optional text path to save per-face density hints, one value per line.")
    parser.add_argument("--summary_json", default=None,
                        help="Optional .json path to save summary statistics.")
    parser.add_argument("--save_labels", default=None,
                        help="Optional .npy path to save semantic pseudo labels.")
    parser.add_argument("--k", type=int, default=None,
                        help="Fixed semantic cluster count. If omitted, use best silhouette.")
    parser.add_argument("--k_min", type=int, default=2,
                        help="Minimum K for silhouette sweep.")
    parser.add_argument("--k_max", type=int, default=15,
                        help="Maximum K for silhouette sweep.")
    parser.add_argument("--w_geometry", type=float, default=0.45,
                        help="Weight for geometry complexity.")
    parser.add_argument("--w_semantic", type=float, default=0.35,
                        help="Weight for semantic complexity.")
    parser.add_argument("--w_field", type=float, default=0.20,
                        help="Weight for cross-field complexity.")
    parser.add_argument("--semantic_grad_mix", type=float, default=0.7,
                        help="Blend factor for semantic gradient vs label-boundary score.")
    parser.add_argument("--robust_percentile", type=float, default=95.0,
                        help="Percentile used to robustly normalize raw scores.")
    parser.add_argument("--size_min", type=float, default=0.60,
                        help="Minimum relative size hint for highly complex regions.")
    parser.add_argument("--size_max", type=float, default=1.0,
                        help="Maximum relative size hint for simple regions.")
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def ensure_dir(path):
    os.makedirs(os.path.abspath(path), exist_ok=True)


def normalize_face_centers(face_centers):
    center = face_centers.mean(axis=0)
    centered = face_centers - center[None, :]
    scale = np.abs(centered).max()
    if scale < 1e-12:
        scale = 1.0
    return centered / scale


def get_face_neighbors(mesh):
    face_adj = mesh.face_adjacency.astype(np.int32)
    neighbors = [[] for _ in range(len(mesh.faces))]
    for face_a, face_b in face_adj:
        neighbors[int(face_a)].append(int(face_b))
        neighbors[int(face_b)].append(int(face_a))
    return neighbors


def calculate_same_neighbors_verts(vertex_neighbors):
    lens = np.asarray(list(map(len, vertex_neighbors)))
    lens_unique, lens_inverse = np.unique(lens, return_inverse=True)
    groups = []
    for i in range(lens_unique.shape[0]):
        groups.append(np.argwhere(lens_inverse == i).squeeze(-1).tolist())
    return groups


def map_edge_information_to_neighbors(face_adjacency, edge_info, neighbors_each_face,
                                      return_missing_mask=False):
    info_map = {
        (min(f1, f2), max(f1, f2)): angle
        for (f1, f2), angle in zip(face_adjacency, edge_info)
    }

    mapped = []
    missing_mask = []
    for face_id, neighbors in enumerate(neighbors_each_face):
        row = []
        row_missing = []
        for neighbor in neighbors:
            key = (min(face_id, neighbor), max(face_id, neighbor))
            if key in info_map:
                row.append(info_map[key])
                row_missing.append(False)
            else:
                if edge_info.ndim == 2:
                    row.append(np.zeros(edge_info.shape[1], dtype=edge_info.dtype))
                else:
                    row.append(0.0)
                row_missing.append(True)
        mapped.append(row)
        missing_mask.append(row_missing)

    mapped = np.array(mapped, dtype=edge_info.dtype if hasattr(edge_info, "dtype") else None)
    missing_mask = np.array(missing_mask, dtype=bool)
    if return_missing_mask:
        return mapped, missing_mask
    return mapped


def batch_axis_angle_to_rotation_matrix_only_rotation(axes, thetas):
    axes = axes / (np.linalg.norm(axes, axis=-1, keepdims=True) + 1e-8)
    vx = axes[..., 0]
    vy = axes[..., 1]
    vz = axes[..., 2]
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    one_minus_cos = 1.0 - cos_theta

    k = np.zeros((axes.shape[0], axes.shape[1], 3, 3), dtype=np.float32)
    k[..., 0, 1] = -vz
    k[..., 0, 2] = vy
    k[..., 1, 0] = vz
    k[..., 1, 2] = -vx
    k[..., 2, 0] = -vy
    k[..., 2, 1] = vx

    identity = np.eye(3, dtype=np.float32)[np.newaxis, np.newaxis, :, :]
    return identity + sin_theta[..., np.newaxis, np.newaxis] * k + one_minus_cos[..., np.newaxis, np.newaxis] * np.einsum(
        "...ij,...jk->...ik", k, k
    )


def get_rotation_matrix(vertex_neighbors_list, vertex_neighbors, mesh):
    face_normals = mesh.face_normals.astype(np.float32)
    face_adjacency = mesh.face_adjacency.astype(np.int32)
    face_adjacency_angles = mesh.face_adjacency_angles.astype(np.float32)
    face_adjacency_edges = mesh.face_adjacency_edges.astype(np.int32)
    verts = mesh.vertices.astype(np.float32)

    rota_axis = verts[face_adjacency_edges[:, 0]] - verts[face_adjacency_edges[:, 1]]
    rotation_mats = []

    for group in vertex_neighbors_list:
        idx = np.array(group)
        face_normals_i = np.expand_dims(face_normals[idx], axis=1)
        vertex_neighbors_i = [vertex_neighbors[z] for z in idx]
        vertex_neighbors_i = np.array(vertex_neighbors_i)
        face_normals_i_neighbor = face_normals[vertex_neighbors_i]
        desired_rota_axis_direction = np.cross(face_normals_i_neighbor, face_normals_i)

        rota_axis_map_to_face, missing_axis_mask = map_edge_information_to_neighbors(
            face_adjacency, rota_axis, vertex_neighbors_i, return_missing_mask=True
        )
        if np.any(missing_axis_mask):
            fallback_axis = desired_rota_axis_direction.copy()
            fallback_axis /= (np.linalg.norm(fallback_axis, axis=-1, keepdims=True) + 1e-8)
            rota_axis_map_to_face[missing_axis_mask] = fallback_axis[missing_axis_mask]

        desired_axis_dot_axis = np.einsum("ijk,ijk->ij", desired_rota_axis_direction, rota_axis_map_to_face)
        flag = desired_axis_dot_axis < 0
        rota_axis_map_to_face[flag] = -rota_axis_map_to_face[flag]

        dihedral_angle_map_to_face, missing_angle_mask = map_edge_information_to_neighbors(
            face_adjacency, face_adjacency_angles, vertex_neighbors_i, return_missing_mask=True
        )
        if np.any(missing_angle_mask):
            dihedral_angle_map_to_face[missing_angle_mask] = 0.0

        rotation_mats.append(
            batch_axis_angle_to_rotation_matrix_only_rotation(
                rota_axis_map_to_face, dihedral_angle_map_to_face
            )
        )

    return rotation_mats


def robust_normalize(values, percentile=95.0):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(np.percentile(values, percentile))
    if hi <= lo + 1e-12:
        hi = float(values.max())
    if hi <= lo + 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    normalized = (values - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0)


def scalar_to_rgba(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = np.clip(values, 0.0, 1.0)

    # Dark blue -> cyan -> yellow -> red
    anchors = np.array([0.0, 0.33, 0.66, 1.0], dtype=np.float64)
    colors = np.array([
        [24, 34, 84],
        [53, 196, 233],
        [244, 208, 63],
        [192, 57, 43],
    ], dtype=np.float64)

    rgba = np.zeros((len(values), 4), dtype=np.uint8)
    for c in range(3):
        rgba[:, c] = np.interp(values, anchors, colors[:, c]).astype(np.uint8)
    rgba[:, 3] = 255
    return rgba


def export_scalar_mesh(mesh, values, output_path):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) != len(mesh.faces):
        raise ValueError(
            f"Value count ({len(values)}) does not match face count ({len(mesh.faces)}).")
    ensure_parent_dir(output_path)
    out_mesh = mesh.copy()
    out_mesh.visual.face_colors = scalar_to_rgba(values)
    out_mesh.export(output_path)


def load_crossfield(path):
    rows = np.loadtxt(path, dtype=np.float64)
    rows = np.atleast_2d(rows)
    if rows.shape[1] != 6:
        raise ValueError(f"Expected cross-field with 6 columns, got shape {rows.shape}")
    alpha = rows[:, :3]
    beta = rows[:, 3:]
    alpha /= np.linalg.norm(alpha, axis=1, keepdims=True) + 1e-12
    beta /= np.linalg.norm(beta, axis=1, keepdims=True) + 1e-12
    return alpha, beta


def compute_geometry_component(face_normals, vertex_neighbors):
    scores = np.zeros(len(face_normals), dtype=np.float64)
    for i, neighbors in enumerate(vertex_neighbors):
        if len(neighbors) == 0:
            continue
        dots = np.sum(face_normals[i][None, :] * face_normals[neighbors], axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        scores[i] = float(np.mean(angles))
    return scores


def boundary_fraction_from_labels(labels, vertex_neighbors):
    labels = np.asarray(labels, dtype=np.int64)
    boundary_fraction = np.zeros(len(labels), dtype=np.float64)
    for i, neighbors in enumerate(vertex_neighbors):
        if len(neighbors) == 0:
            continue
        neighbor_labels = labels[np.asarray(neighbors, dtype=np.int64)]
        boundary_fraction[i] = float(np.mean(neighbor_labels != labels[i]))
    return boundary_fraction


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


def compute_semantic_component(mesh, features, vertex_neighbors, args):
    face_centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    face_centers_norm = normalize_face_centers(face_centers)

    grad_dir, grad_weight = compute_semantic_gradient(
        face_centers_norm, face_normals, vertex_neighbors, features
    )
    grad_weight = np.asarray(grad_weight, dtype=np.float64)

    labels, k_used, silhouette = cluster_from_features(
        features, k=args.k, k_range=(args.k_min, args.k_max)
    )
    label_boundary = boundary_fraction_from_labels(labels, vertex_neighbors)
    mix = float(np.clip(args.semantic_grad_mix, 0.0, 1.0))
    semantic_raw = mix * grad_weight + (1.0 - mix) * label_boundary

    boundary_edges = _find_boundary_edges_from_labels(
        np.asarray(mesh.faces, dtype=np.int64), labels
    )
    return {
        "raw": semantic_raw,
        "grad_weight": grad_weight,
        "label_boundary": label_boundary,
        "labels": labels,
        "k_used": int(k_used),
        "silhouette": None if silhouette is None else float(silhouette),
        "n_boundary_edges": int(len(boundary_edges)),
        "grad_dir": grad_dir,
    }


def rotate_neighbor_vectors(rotation_mats, vectors):
    return np.einsum("...ij,...j->...i", rotation_mats, vectors)


def compute_field_component(mesh_path, vertex_neighbors, vector_alpha, vector_beta):
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertex_neighbors_list = calculate_same_neighbors_verts(vertex_neighbors)
    axis_angle_R_mat_list = get_rotation_matrix(
        vertex_neighbors_list, vertex_neighbors, mesh
    )

    scores = np.zeros(len(vector_alpha), dtype=np.float64)

    for group_idx, face_group in enumerate(vertex_neighbors_list):
        if len(face_group) == 0:
            continue

        face_group = np.asarray(face_group, dtype=np.int64)
        neighbor_ids = np.asarray([vertex_neighbors[idx] for idx in face_group], dtype=np.int64)

        alpha_i = vector_alpha[face_group][:, None, :]
        beta_i = vector_beta[face_group][:, None, :]

        alpha_j = vector_alpha[neighbor_ids]
        beta_j = vector_beta[neighbor_ids]

        rotation_mats = np.asarray(axis_angle_R_mat_list[group_idx], dtype=np.float64)
        alpha_j = rotate_neighbor_vectors(rotation_mats, alpha_j)
        beta_j = rotate_neighbor_vectors(rotation_mats, beta_j)

        aa = np.abs(np.sum(alpha_i * alpha_j, axis=-1))
        ab = np.abs(np.sum(alpha_i * beta_j, axis=-1))
        ba = np.abs(np.sum(beta_i * alpha_j, axis=-1))
        bb = np.abs(np.sum(beta_i * beta_j, axis=-1))
        raw_neighbors_term = aa + ab + ba + bb - 2.0
        scores[face_group] = raw_neighbors_term.mean(axis=1)

    return scores


def normalize_weights(weight_items):
    active = [(name, float(weight)) for name, weight, enabled in weight_items if enabled and weight > 0.0]
    if not active:
        raise ValueError("At least one positive active component weight is required.")
    total = sum(weight for _, weight in active)
    return {name: weight / total for name, weight in active}


def summarize_array(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def main():
    args = parse_args()

    if not os.path.isfile(args.mesh):
        raise FileNotFoundError(f"Mesh not found: {args.mesh}")
    if args.feat_path is not None and not os.path.isfile(args.feat_path):
        raise FileNotFoundError(f"Feature file not found: {args.feat_path}")
    if args.crossfield is not None and not os.path.isfile(args.crossfield):
        raise FileNotFoundError(f"Cross-field file not found: {args.crossfield}")
    if args.size_min <= 0.0 or args.size_max <= 0.0 or args.size_min > args.size_max:
        raise ValueError("Require 0 < size_min <= size_max.")

    mesh = trimesh.load_mesh(args.mesh, process=False)
    face_count = len(mesh.faces)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    vertex_neighbors = get_face_neighbors(mesh)

    if len(vertex_neighbors) != face_count:
        raise ValueError(
            f"Neighbor list length ({len(vertex_neighbors)}) does not match face count ({face_count}).")

    geometry_raw = compute_geometry_component(face_normals, vertex_neighbors)
    geometry = robust_normalize(geometry_raw, percentile=args.robust_percentile)

    semantic = np.zeros(face_count, dtype=np.float64)
    semantic_grad = np.zeros(face_count, dtype=np.float64)
    semantic_label_boundary = np.zeros(face_count, dtype=np.float64)
    semantic_labels = None
    semantic_info = None
    if args.feat_path is not None:
        features = np.load(args.feat_path)
        if len(features) != face_count:
            raise ValueError(
                f"Feature count ({len(features)}) does not match face count ({face_count}).")
        semantic_info = compute_semantic_component(mesh, features, vertex_neighbors, args)
        semantic = robust_normalize(semantic_info["raw"], percentile=args.robust_percentile)
        semantic_grad = semantic_info["grad_weight"]
        semantic_label_boundary = semantic_info["label_boundary"]
        semantic_labels = semantic_info["labels"]

    field = np.zeros(face_count, dtype=np.float64)
    field_raw = np.zeros(face_count, dtype=np.float64)
    if args.crossfield is not None:
        vector_alpha, vector_beta = load_crossfield(args.crossfield)
        if len(vector_alpha) != face_count:
            raise ValueError(
                f"Cross-field row count ({len(vector_alpha)}) does not match face count ({face_count}).")
        field_raw = compute_field_component(args.mesh, vertex_neighbors, vector_alpha, vector_beta)
        field = robust_normalize(field_raw, percentile=args.robust_percentile)

    effective_weights = normalize_weights([
        ("geometry", args.w_geometry, True),
        ("semantic", args.w_semantic, args.feat_path is not None),
        ("field", args.w_field, args.crossfield is not None),
    ])

    complexity = np.zeros(face_count, dtype=np.float64)
    complexity += effective_weights.get("geometry", 0.0) * geometry
    complexity += effective_weights.get("semantic", 0.0) * semantic
    complexity += effective_weights.get("field", 0.0) * field
    complexity = np.clip(complexity, 0.0, 1.0)

    size_hint = args.size_max - complexity * (args.size_max - args.size_min)
    density_hint = 1.0 / np.clip(size_hint, 1e-12, None)

    export_scalar_mesh(mesh, complexity, args.output)

    if args.component_dir is not None:
        ensure_dir(args.component_dir)
        export_scalar_mesh(mesh, geometry, os.path.join(args.component_dir, "geometry.ply"))
        export_scalar_mesh(mesh, semantic, os.path.join(args.component_dir, "semantic.ply"))
        export_scalar_mesh(mesh, field, os.path.join(args.component_dir, "field.ply"))
        export_scalar_mesh(mesh, complexity, os.path.join(args.component_dir, "complexity.ply"))
        export_scalar_mesh(mesh, 1.0 - size_hint / args.size_max, os.path.join(args.component_dir, "size_hint_inverse.ply"))

    if args.save_npz is not None:
        ensure_parent_dir(args.save_npz)
        arrays = {
            "geometry_raw": geometry_raw,
            "geometry": geometry,
            "semantic": semantic,
            "semantic_grad_weight": semantic_grad,
            "semantic_label_boundary": semantic_label_boundary,
            "field_raw": field_raw,
            "field": field,
            "complexity": complexity,
            "size_hint": size_hint,
            "density_hint": density_hint,
        }
        if semantic_labels is not None:
            arrays["semantic_labels"] = semantic_labels.astype(np.int64)
        np.savez(args.save_npz, **arrays)

    if args.save_size_hint_txt is not None:
        ensure_parent_dir(args.save_size_hint_txt)
        np.savetxt(args.save_size_hint_txt, size_hint, fmt="%.8f")

    if args.save_density_txt is not None:
        ensure_parent_dir(args.save_density_txt)
        np.savetxt(args.save_density_txt, density_hint, fmt="%.8f")

    if args.save_labels is not None:
        if semantic_labels is None:
            raise ValueError("--save_labels requires --feat_path.")
        ensure_parent_dir(args.save_labels)
        np.save(args.save_labels, semantic_labels.astype(np.int64))

    summary = {
        "mesh": os.path.abspath(args.mesh),
        "feat_path": None if args.feat_path is None else os.path.abspath(args.feat_path),
        "crossfield": None if args.crossfield is None else os.path.abspath(args.crossfield),
        "face_count": int(face_count),
        "component_availability": {
            "geometry": True,
            "semantic": args.feat_path is not None,
            "field": args.crossfield is not None,
        },
        "requested_weights": {
            "geometry": float(args.w_geometry),
            "semantic": float(args.w_semantic),
            "field": float(args.w_field),
        },
        "effective_weights": effective_weights,
        "stats": {
            "geometry": summarize_array(geometry),
            "semantic": summarize_array(semantic),
            "field": summarize_array(field),
            "complexity": summarize_array(complexity),
            "size_hint": summarize_array(size_hint),
            "density_hint": summarize_array(density_hint),
        },
    }

    if semantic_info is not None:
        summary["semantic"] = {
            "k_used": semantic_info["k_used"],
            "silhouette": semantic_info["silhouette"],
            "n_boundary_edges": semantic_info["n_boundary_edges"],
        }

    summary_path = args.summary_json
    if summary_path is None:
        base, _ = os.path.splitext(args.output)
        summary_path = f"{base}_summary.json"
    ensure_parent_dir(summary_path)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Complexity mesh exported to: {args.output}")
    print(f"Summary saved to: {summary_path}")
    if args.component_dir is not None:
        print(f"Component meshes exported to: {args.component_dir}")
    if args.save_npz is not None:
        print(f"Per-face arrays saved to: {args.save_npz}")
    if semantic_info is not None:
        print(
            "Semantic pseudo labels: K={} silhouette={}".format(
                semantic_info["k_used"],
                "n/a" if semantic_info["silhouette"] is None else f"{semantic_info['silhouette']:.4f}"
            )
        )
    print(f"Effective weights: {effective_weights}")


if __name__ == "__main__":
    main()
