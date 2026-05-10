#!/usr/bin/env python
"""
Standalone quad mesh extraction from NeurCross cross-field output.

Reads a triangle mesh (.obj) and a NeurCross cross-field .txt file,
then runs MIQ parameterization + libQEx to produce a quad mesh.

Usage
-----
    # Single cross-field file:
    python extract_quad.py --mesh input/cheburashka.obj \
        --crossfield pipeline_output/neurcross_logs/cheburashka/save_crossField/cheburashka_iter_9999.txt \
        --output output_quad.obj

    # Auto-detect latest cross-field from a NeurCross log directory:
    python extract_quad.py --mesh input/cheburashka.obj \
        --crossfield_dir pipeline_output/neurcross_logs/cheburashka/save_crossField/

    # Batch mode — process all meshes under a directory:
    python extract_quad.py --mesh_dir input/ \
        --crossfield_root pipeline_output/neurcross_logs/ \
        --output_dir pipeline_output/quad_meshes/
"""

import os
import sys
import glob
import argparse
import subprocess
import json
import shutil

import numpy as np
import trimesh

from eval.angle_distortion import compute_angle_distortion, _load_quad_faces_from_obj
from eval.jacobian_ratio import compute_jacobian_ratio

from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
QUAD_EXTRACT_BIN = os.path.join(
    PROJECT_ROOT, "quad_extract", "build", "extract_quad_mesh")

ALL_MESH_EXTS = {".obj", ".glb", ".off", ".ply", ".stl"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone quad mesh extraction (MIQ + libQEx)")

    mesh_grp = p.add_mutually_exclusive_group(required=True)
    mesh_grp.add_argument("--mesh", help="Path to a single input triangle mesh (.obj)")
    mesh_grp.add_argument("--mesh_dir", help="Directory of triangle meshes for batch mode")

    cf_grp = p.add_mutually_exclusive_group()
    cf_grp.add_argument("--crossfield", help="Path to a single cross-field .txt file")
    cf_grp.add_argument(
        "--crossfield_dir",
        help="Directory containing *_iter_*.txt files; the latest iteration is used")
    cf_grp.add_argument(
        "--crossfield_root",
        help="Root directory of NeurCross logs (batch mode). "
             "Expects <root>/<mesh_name>/save_crossField/ structure")

    p.add_argument("--output", default=None,
                   help="Output quad mesh path (single-mesh mode)")
    p.add_argument("--output_dir", default=None,
                   help="Output directory for quad meshes (batch mode)")
    p.add_argument("--gradient_size", type=float, default=30.0,
                   help="MIQ gradient size — controls quad density (default: 30.0)")
    p.add_argument("--size_field", default=None,
                   help="Deprecated and ignored. Extraction now follows the pure MIQ -> libQEx pipeline.")
    p.add_argument("--size_field_strength", type=float, default=0.25,
                   help="Deprecated and ignored.")
    p.add_argument("--size_field_smooth_iters", type=int, default=6,
                   help="Deprecated and ignored.")
    p.add_argument("--size_field_relax", action="store_true",
                   help="Deprecated and ignored.")
    p.add_argument("--timeout", type=int, default=6000,
                   help="Timeout per extraction attempt in seconds (default: 600)")
    p.add_argument("--retry", action="store_true",
                   help="Auto-retry with multiple gradient sizes and iterations on failure")
    p.add_argument("--max_fallback_attempts", type=int, default=None,
                   help="Limit MIQ fallback configs per extraction call. "
                        "Useful for batch retries where bad chunks should fail fast.")
    p.add_argument("--max_repair_tiers", type=int, default=4,
                   help="Maximum repair tiers to attempt, from 1 to 4 "
                        "(default: 4)")
    p.add_argument("--auto_sweep", action="store_true",
                   help="Sweep multiple gradient sizes, evaluate each result, and keep the best one")
    p.add_argument("--sweep_values", nargs="+", type=float, default=None,
                   help="Explicit gradient_size values for auto sweep "
                        "(example: --sweep_values 8 12 16 24 30 40 60)")
    p.add_argument("--keep_sweep_outputs", action="store_true",
                   help="Keep all intermediate sweep OBJ files instead of only the best one")
    p.add_argument("--intermediate_dir", default=None,
                   help="Directory for repair and sweep intermediates. "
                        "Defaults to <output>_extract_work")
    p.add_argument("--keep_intermediates", action="store_true",
                   help="Keep extraction intermediates after a successful run")
    p.add_argument("--summary_json", default=None,
                   help="Optional JSON path to save auto-sweep metrics and ranking summary")
    p.add_argument("--min_quads", type=int, default=None,
                   help="Hard lower bound for acceptable quad count in auto-sweep mode. "
                        "Default: max(100, 5%% of input triangle faces)")
    p.add_argument("--max_ad", type=float, default=15.0,
                   help="Hard upper bound for acceptable mean angle distortion in degrees "
                        "during auto sweep (default: 15.0)")
    p.add_argument("--min_jr", type=float, default=0.15,
                   help="Hard lower bound for acceptable minimum Jacobian Ratio "
                        "during auto sweep (default: 0.15)")
    p.add_argument("--catmull_clark_iters", type=int, default=0,
                   help="Fixed number of Catmull-Clark subdivision iterations after extraction.")
    p.add_argument("--target_quad_ratio", type=float, default=0.5,
                   help="Target final quad count as a ratio of input triangle faces. "
                        "Set <= 0 to disable automatic Catmull-Clark targeting.")
    p.add_argument("--budget_tolerance_ratio", type=float, default=0.2,
                   help="Relative tolerance band around the quad budget during auto-sweep ranking.")
    p.add_argument("--max_catmull_clark_iters", type=int, default=2,
                   help="Maximum automatic Catmull-Clark subdivision iterations.")
    p.add_argument("--snap_to_boundary", action="store_true",
                   help="Post-process: snap quad vertices to nearby semantic boundaries")
    p.add_argument("--snap_feat_path", default=None,
                   help="Path to PartField features (.npy) for boundary snapping")
    p.add_argument("--snap_fraction", type=float, default=0.3,
                   help="How far to snap toward boundary (0=none, 1=full)")
    p.add_argument("--snap_threshold_ratio", type=float, default=0.01,
                   help="Max snap distance as fraction of bounding box diagonal")
    p.add_argument("--snap_min_angle", type=float, default=25.0,
                   help="Reject snap if any incident quad angle drops below this (degrees)")
    p.add_argument("--snap_smooth_iters", type=int, default=3,
                   help="Laplacian smoothing iterations after snapping")
    p.add_argument("--snap_smooth_weight", type=float, default=0.3,
                   help="Laplacian smoothing step size (0-1)")

    p.add_argument("--stitch_quad_patches", action="store_true",
                   help="Post-process: merge disconnected quad patches into a single "
                        "connected mesh (for reconstruction/B-rep models)")
    p.add_argument("--stitch_threshold_ratio", type=float, default=0.005,
                   help="Max merge distance for stitching as fraction of bbox diagonal")
    p.add_argument("--stitch_smooth_iters", type=int, default=2,
                   help="Laplacian smoothing iterations after stitching")
    p.add_argument("--stitch_smooth_weight", type=float, default=0.2,
                   help="Laplacian smoothing step size after stitching")

    return p.parse_args()


def find_latest_crossfield(crossfield_dir):
    """Find the cross-field txt with the highest iteration number."""
    files = glob.glob(os.path.join(crossfield_dir, "*_iter_*.txt"))
    if not files:
        return None

    def iter_num(f):
        base = os.path.splitext(os.path.basename(f))[0]
        return int(base.rsplit("_iter_", 1)[1])
    return max(files, key=iter_num)


def weld_mesh_vertices(mesh_path, output_path):
    """Merge duplicate vertices so disconnected B-Rep patches become connected.

    Many CAD-exported meshes triangulate each B-Rep face independently,
    duplicating vertices along shared edges.  The MIQ + libQEx pipeline
    requires a single connected component to produce a stitched quad mesh.

    Returns the path to the welded OBJ (may be the original if no welding
    was needed) and a bool indicating whether welding was performed.
    """
    mesh = trimesh.load_mesh(mesh_path, process=False)
    n_verts_before = len(mesh.vertices)

    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    n_verts_after = len(mesh.vertices)

    if n_verts_after == n_verts_before:
        return mesh_path, False

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    mesh.export(output_path)
    print(f"  [weld] merged duplicate vertices: {n_verts_before} -> {n_verts_after}"
          f" ({n_verts_before - n_verts_after} removed)")
    return output_path, True


def _vertex_connected_components(V, F):
    """Return component label per vertex and the number of components."""
    from collections import defaultdict
    adj = defaultdict(set)
    for face in F:
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            adj[a].add(b)
            adj[b].add(a)
    labels = np.full(len(V), -1, dtype=np.int64)
    comp_id = 0
    for seed in range(len(V)):
        if labels[seed] >= 0:
            continue
        stack = [seed]
        labels[seed] = comp_id
        while stack:
            v = stack.pop()
            for nb in adj[v]:
                if labels[nb] < 0:
                    labels[nb] = comp_id
                    stack.append(nb)
        comp_id += 1
    return labels, comp_id


def _boundary_vertices(F):
    """Return set of vertex indices on mesh boundary edges."""
    edge_count = {}
    for face in F:
        for i in range(3):
            e = tuple(sorted([int(face[i]), int(face[(i + 1) % 3])]))
            edge_count[e] = edge_count.get(e, 0) + 1
    bv = set()
    for (a, b), c in edge_count.items():
        if c == 1:
            bv.add(a)
            bv.add(b)
    return bv


def resolve_nonmanifold(mesh_path, crossfield_path, output_mesh_path,
                        output_crossfield_path):
    """Make mesh edge-manifold by isolating each face-connected component
    into its own vertex set, so no vertex or edge is shared across components.

    Steps:
      1. Build face adjacency via shared edges.
      2. Find face-connected components.
      3. For each vertex used by more than one component, duplicate it so
         each component gets its own copy.

    The cross-field (per-face) is unchanged because face order is preserved.

    Returns (new_mesh_path, new_cf_path, n_dup_verts).
    """
    mesh = trimesh.load_mesh(mesh_path, process=False)
    V = np.array(mesh.vertices, dtype=np.float64)
    F = np.array(mesh.faces, dtype=np.int64)
    nf = len(F)

    edge_to_faces = {}
    for fi in range(nf):
        for i in range(3):
            a, b = int(F[fi, i]), int(F[fi, (i + 1) % 3])
            e = (min(a, b), max(a, b))
            edge_to_faces.setdefault(e, []).append(fi)

    face_adj = [set() for _ in range(nf)]
    for e_faces in edge_to_faces.values():
        if len(e_faces) == 2:
            face_adj[e_faces[0]].add(e_faces[1])
            face_adj[e_faces[1]].add(e_faces[0])

    face_comp = np.full(nf, -1, dtype=np.int64)
    n_comp = 0
    for seed in range(nf):
        if face_comp[seed] >= 0:
            continue
        stack = [seed]
        face_comp[seed] = n_comp
        while stack:
            cur = stack.pop()
            for nb in face_adj[cur]:
                if face_comp[nb] < 0:
                    face_comp[nb] = n_comp
                    stack.append(nb)
        n_comp += 1

    if n_comp <= 1:
        ec_nm = sum(1 for fl in edge_to_faces.values() if len(fl) > 2)
        if ec_nm == 0:
            return mesh_path, crossfield_path, 0

    vert_usage = {}
    for fi in range(nf):
        c = int(face_comp[fi])
        for k in range(3):
            vi = int(F[fi, k])
            vert_usage.setdefault(vi, {})
            vert_usage[vi].setdefault(c, []).append((fi, k))

    new_V = list(V)
    new_F = F.copy()
    n_dup = 0

    for vi, comp_map in vert_usage.items():
        if len(comp_map) <= 1:
            continue
        comps_sorted = sorted(comp_map.keys())
        for extra_c in comps_sorted[1:]:
            new_idx = len(new_V)
            new_V.append(V[vi].copy())
            n_dup += 1
            for fi, k in comp_map[extra_c]:
                new_F[fi, k] = new_idx

    if n_comp <= 1 and n_dup == 0:
        nm_edges = {e: fl for e, fl in edge_to_faces.items() if len(fl) > 2}
        for e, face_list in nm_edges.items():
            for fi in face_list[2:]:
                for k in range(3):
                    vi = int(new_F[fi, k])
                    new_idx = len(new_V)
                    new_V.append(V[vi].copy())
                    new_F[fi, k] = new_idx
                n_dup += 1

    if n_dup == 0:
        return mesh_path, crossfield_path, 0

    new_V = np.array(new_V, dtype=np.float64)

    os.makedirs(os.path.dirname(os.path.abspath(output_mesh_path)), exist_ok=True)
    repaired = trimesh.Trimesh(vertices=new_V, faces=new_F, process=False)
    repaired.export(output_mesh_path)

    import shutil
    shutil.copy2(crossfield_path, output_crossfield_path)

    edge_to_faces2 = {}
    for fi in range(len(new_F)):
        for i in range(3):
            a, b = int(new_F[fi, i]), int(new_F[fi, (i + 1) % 3])
            e2 = (min(a, b), max(a, b))
            edge_to_faces2.setdefault(e2, []).append(fi)
    nm_after = sum(1 for fl in edge_to_faces2.values() if len(fl) > 2)

    print(f"  [nm-fix] duplicated {n_dup} vertex instances, "
          f"verts: {len(V)} -> {len(new_V)}, "
          f"face-comps: {n_comp}, nm_edges: {nm_after}")
    return output_mesh_path, output_crossfield_path, n_dup


def bridge_connected_components(mesh_path, crossfield_path, output_mesh_path,
                                output_crossfield_path):
    """Connect disconnected components via thin bridge triangles.

    For each pair of adjacent components the closest boundary vertex pair
    is found and two bridge triangles are added.  Corresponding cross-field
    rows are interpolated from the nearest original face.

    Returns (bridged_mesh_path, bridged_cf_path, n_bridge_faces, face_map)
    where *face_map* maps original face indices to the bridged mesh.
    If the mesh is already a single component the originals are returned
    unchanged.
    """
    from scipy.spatial import cKDTree

    mesh = trimesh.load_mesh(mesh_path, process=False)
    V = np.array(mesh.vertices, dtype=np.float64)
    F = np.array(mesh.faces, dtype=np.int64)

    labels, n_comp = _vertex_connected_components(V, F)
    if n_comp <= 1:
        return mesh_path, crossfield_path, 0, None

    comp_verts = {}
    for c in range(n_comp):
        cv = sorted(int(v) for v in np.flatnonzero(labels == c))
        comp_verts[c] = np.array(cv, dtype=np.int64) if cv else np.empty(0, dtype=np.int64)

    bridge_faces = []
    bridge_cf_rows = []
    new_V = list(V)

    cf_rows = []
    with open(crossfield_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                cf_rows.append(line)

    face_centers = V[F].mean(axis=1)
    face_tree = cKDTree(face_centers)

    connected = {0}
    remaining = set(range(1, n_comp))

    while remaining:
        best_dist = float("inf")
        best_pair = None
        for c_from in connected:
            cv_from = comp_verts.get(c_from, np.empty(0, dtype=np.int64))
            if len(cv_from) == 0:
                continue
            pts_from = V[cv_from]
            for c_to in remaining:
                cv_to = comp_verts.get(c_to, np.empty(0, dtype=np.int64))
                if len(cv_to) == 0:
                    continue
                pts_to = V[cv_to]
                tree_to = cKDTree(pts_to)
                dists, idxs = tree_to.query(pts_from, k=1)
                best_i = int(np.argmin(dists))
                d = dists[best_i]
                if d < best_dist:
                    best_dist = d
                    vi_from = int(cv_from[best_i])
                    vi_to = int(cv_to[idxs[best_i]])
                    best_pair = (c_from, c_to, vi_from, vi_to)

        if best_pair is None:
            break

        c_from, c_to, vi_from, vi_to = best_pair
        connected.add(c_to)
        remaining.discard(c_to)

        edge_vec = V[vi_to] - V[vi_from]
        edge_len = np.linalg.norm(edge_vec)
        mid_pos = (V[vi_from] + V[vi_to]) * 0.5

        perp = np.cross(edge_vec, np.array([0.0, 0.0, 1.0]))
        pn = np.linalg.norm(perp)
        if pn < 1e-12:
            perp = np.cross(edge_vec, np.array([0.0, 1.0, 0.0]))
            pn = np.linalg.norm(perp)
        offset = max(edge_len * 0.3, best_dist * 0.05)
        if pn > 1e-12:
            perp = perp / pn * offset

        mid_idx_a = len(new_V)
        new_V.append(mid_pos + perp)
        mid_idx_b = len(new_V)
        new_V.append(mid_pos - perp)

        bridge_faces.append([vi_from, vi_to, mid_idx_a])
        bridge_faces.append([vi_to, vi_from, mid_idx_b])

        _, nearest_fi = face_tree.query(mid_pos)
        bridge_cf_rows.append(cf_rows[nearest_fi])
        bridge_cf_rows.append(cf_rows[nearest_fi])

    if not bridge_faces:
        return mesh_path, crossfield_path, 0, None

    new_V = np.array(new_V, dtype=np.float64)
    new_F = np.vstack([F, np.array(bridge_faces, dtype=np.int64)])

    n_bridge = len(bridge_faces)
    face_map = np.arange(len(F), dtype=np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(output_mesh_path)), exist_ok=True)
    bridged_mesh = trimesh.Trimesh(vertices=new_V, faces=new_F, process=False)
    bridged_mesh.export(output_mesh_path)

    with open(output_crossfield_path, "w") as f:
        for row in cf_rows:
            f.write(row + "\n")
        for row in bridge_cf_rows:
            f.write(row + "\n")

    labels2, nc2 = _vertex_connected_components(new_V, new_F)
    print(f"  [bridge] added {n_bridge} bridge faces, components: {n_comp} -> {nc2}")
    return output_mesh_path, output_crossfield_path, n_bridge, face_map


def _compute_min_angles(V, F):
    """Return per-face minimum interior angle (radians) and per-face shortest
    edge local index, all vectorised."""
    p0 = V[F[:, 0]]
    p1 = V[F[:, 1]]
    p2 = V[F[:, 2]]

    e01 = p1 - p0;  e02 = p2 - p0
    e10 = p0 - p1;  e12 = p2 - p1
    e20 = p0 - p2;  e21 = p1 - p2

    def _angle(a, b):
        na = np.linalg.norm(a, axis=1, keepdims=True).clip(1e-30)
        nb = np.linalg.norm(b, axis=1, keepdims=True).clip(1e-30)
        cos = np.clip((a * b).sum(axis=1) / (na[:, 0] * nb[:, 0]), -1, 1)
        return np.arccos(cos)

    a0 = _angle(e01, e02)
    a1 = _angle(e10, e12)
    a2 = _angle(e20, e21)
    angles = np.column_stack([a0, a1, a2])
    min_angle = angles.min(axis=1)

    edge_lens = np.column_stack([
        np.linalg.norm(e01, axis=1),
        np.linalg.norm(e12, axis=1),
        np.linalg.norm(e20, axis=1),
    ])
    shortest_edge = edge_lens.argmin(axis=1)
    return min_angle, shortest_edge


def repair_degenerate_triangles(mesh_path, crossfield_path, output_mesh_path,
                                output_crossfield_path, min_angle_deg=1.0):
    """Collapse edges of extremely skinny triangles (min-angle < threshold).

    Returns (mesh_path, cf_path, n_collapsed).
    """
    mesh = trimesh.load_mesh(mesh_path, process=False)
    V = np.array(mesh.vertices, dtype=np.float64)
    F = np.array(mesh.faces, dtype=np.int64)

    cf_data = np.loadtxt(crossfield_path, dtype=np.float64)
    if len(cf_data.shape) == 1:
        cf_data = cf_data.reshape(1, -1)
    if cf_data.shape[0] != len(F):
        print(f"  [repair] cross-field rows ({cf_data.shape[0]}) != faces ({len(F)}), skipping")
        return mesh_path, crossfield_path, 0

    min_angle_rad = np.radians(min_angle_deg)
    n_collapsed = 0
    max_iters = 500

    for _ in range(max_iters):
        if len(F) == 0:
            break

        min_angles, shortest_edges = _compute_min_angles(V, F)
        worst_idx = int(np.argmin(min_angles))
        if min_angles[worst_idx] >= min_angle_rad:
            break

        se = int(shortest_edges[worst_idx])
        va = int(F[worst_idx, se])
        vb = int(F[worst_idx, (se + 1) % 3])

        V[va] = (V[va] + V[vb]) * 0.5
        F[F == vb] = va

        has_dup = (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2])
        keep = ~has_dup
        F = F[keep]
        cf_data = cf_data[keep]
        n_collapsed += 1

    if n_collapsed == 0:
        return mesh_path, crossfield_path, 0

    unique_v, inv = np.unique(F.reshape(-1), return_inverse=True)
    new_V = V[unique_v]
    new_F = inv.reshape(-1, 3)

    os.makedirs(os.path.dirname(os.path.abspath(output_mesh_path)), exist_ok=True)
    repaired = trimesh.Trimesh(vertices=new_V, faces=new_F, process=False)
    repaired.export(output_mesh_path)
    np.savetxt(output_crossfield_path, cf_data, fmt="%.8f")

    print(f"  [repair] collapsed {n_collapsed} degenerate edge(s), "
          f"V: {len(mesh.vertices)}->{len(new_V)}, F: {len(mesh.faces)}->{len(new_F)}")
    return output_mesh_path, output_crossfield_path, n_collapsed


def transfer_crossfield(src_mesh_path, src_cf_path, dst_mesh_path, dst_cf_path):
    """Transfer a per-face cross-field from one triangle mesh to another.

    For each face in *dst_mesh*, the nearest face (by centroid) in *src_mesh*
    is found and its cross-field row (6 floats: PD1 xyz + PD2 xyz) is copied.

    Returns the maximum centroid distance encountered.
    """
    src_mesh = trimesh.load_mesh(src_mesh_path, process=False)
    dst_mesh = trimesh.load_mesh(dst_mesh_path, process=False)
    cf_data = np.loadtxt(src_cf_path, dtype=np.float64)
    if len(cf_data.shape) == 1:
        cf_data = cf_data.reshape(1, -1)
    if cf_data.shape[0] != len(src_mesh.faces):
        raise ValueError(
            f"CF rows ({cf_data.shape[0]}) != src faces ({len(src_mesh.faces)})")

    src_centroids = src_mesh.triangles_center
    dst_centroids = dst_mesh.triangles_center
    tree = cKDTree(src_centroids)
    dists, idxs = tree.query(dst_centroids)
    new_cf = cf_data[idxs]
    np.savetxt(dst_cf_path, new_cf, fmt="%.12f")
    max_dist = float(dists.max()) if len(dists) > 0 else 0.0
    print(f"  [cf-transfer] {len(src_mesh.faces)} -> {len(dst_mesh.faces)} faces, "
          f"max_dist={max_dist:.6f}, mean_dist={dists.mean():.6f}")
    return max_dist


def pymeshlab_manifold_repair(mesh_path, output_path):
    """Tier 1 repair: use PyMeshLab to fix non-manifold topology and close holes.

    Steps: merge close vertices -> repair NM edges -> repair NM vertices ->
    close holes (large) -> second pass NM repair.

    Returns (output_path, face_count_changed: bool) on success,
    or (mesh_path, False) if pymeshlab is unavailable.
    """
    try:
        import pymeshlab
    except ImportError:
        print("  [pml-repair] pymeshlab not available, skipping")
        return mesh_path, False

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.path.abspath(mesh_path))
    nf_before = ms.current_mesh().face_number()

    ms.meshing_merge_close_vertices(threshold=pymeshlab.Percentage(0.5))
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_close_holes(maxholesize=1000)
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()

    nf_after = ms.current_mesh().face_number()
    nv_after = ms.current_mesh().vertex_number()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ms.save_current_mesh(os.path.abspath(output_path))

    face_count_changed = (nf_after != nf_before)
    print(f"  [pml-repair] faces: {nf_before} -> {nf_after}, "
          f"verts: {nv_after}, changed={face_count_changed}")
    return output_path, face_count_changed


def pymeshlab_isotropic_remesh(mesh_path, output_path, iterations=5,
                               targetlen_pct=100):
    """Tier 2 repair: PyMeshLab manifold repair + isotropic remeshing.

    Produces uniform, well-shaped triangles. Face count will change, so
    cross-field transfer is always required afterward.

    Returns output_path on success, or mesh_path if pymeshlab unavailable.
    """
    try:
        import pymeshlab
    except ImportError:
        print("  [pml-remesh] pymeshlab not available, skipping")
        return mesh_path

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.path.abspath(mesh_path))
    nf_before = ms.current_mesh().face_number()

    ms.meshing_merge_close_vertices(threshold=pymeshlab.Percentage(0.5))
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_close_holes(maxholesize=1000)
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_isotropic_explicit_remeshing(
        iterations=iterations, targetlen=pymeshlab.Percentage(targetlen_pct))

    nf_after = ms.current_mesh().face_number()
    nv_after = ms.current_mesh().vertex_number()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ms.save_current_mesh(os.path.abspath(output_path))

    print(f"  [pml-remesh] faces: {nf_before} -> {nf_after}, verts: {nv_after}")
    return output_path


def sdf_reconstruct(mesh_path, output_path, resolution=128):
    """Tier 3 repair: SDF reconstruction via mesh2sdf + marching cubes.

    Converts the mesh to a signed distance field on a regular grid, then
    extracts the zero-level-set via marching cubes. The result is guaranteed
    to be manifold, single-component, and watertight.

    Returns output_path on success, or mesh_path on failure.
    """
    try:
        import mesh2sdf
        from skimage.measure import marching_cubes
    except ImportError as e:
        print(f"  [sdf-recon] required library not available ({e}), skipping")
        return mesh_path

    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)

    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    extent = (vertices.max(axis=0) - vertices.min(axis=0)).max()
    if extent < 1e-12:
        print("  [sdf-recon] degenerate mesh (zero extent), skipping")
        return mesh_path

    norm_factor = extent / 1.8

    try:
        sdf_grid = mesh2sdf.core.compute(
            vertices.astype(np.float32), faces, resolution)
    except Exception as e:
        print(f"  [sdf-recon] mesh2sdf failed: {e}, skipping")
        return mesh_path

    try:
        mc_verts, mc_faces, _, _ = marching_cubes(sdf_grid, level=0.0)
    except Exception as e:
        print(f"  [sdf-recon] marching_cubes failed: {e}, skipping")
        return mesh_path

    if len(mc_verts) == 0 or len(mc_faces) == 0:
        print("  [sdf-recon] empty reconstruction, skipping")
        return mesh_path

    mc_verts_world = (mc_verts / resolution * 2.0 - 1.0) * norm_factor + center

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    recon = trimesh.Trimesh(vertices=mc_verts_world, faces=mc_faces, process=False)
    recon.export(output_path)

    print(f"  [sdf-recon] reconstructed: {len(mc_faces)} faces, "
          f"{len(mc_verts_world)} verts (resolution={resolution})")
    return output_path


def ensure_obj_mesh(mesh_path, output_path):
    """Ensure the extractor input is OBJ, converting if needed."""
    mesh_path = os.path.abspath(mesh_path)
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".obj":
        return mesh_path

    output_dir = os.path.dirname(os.path.abspath(output_path))
    output_base = os.path.splitext(os.path.basename(output_path))[0]
    converted_obj = os.path.join(output_dir, f"{output_base}_input.obj")

    mesh = trimesh.load_mesh(mesh_path, process=False)
    mesh.export(converted_obj)
    print(f"  converted mesh: {mesh_path} -> {converted_obj}")
    return converted_obj


def _cleanup_intermediate_dir(work_dir, keep_intermediates):
    if keep_intermediates:
        print(f"  intermediates kept: {work_dir}")
        return
    if work_dir and os.path.isdir(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)


def _run_extract_once(mesh_path, crossfield_path, output_path,
                      gradient_size=30.0, timeout=600, stiffness=5.0,
                      direct_round=False, iters=5, local_iters=5,
                      size_field_path=None, size_field_strength=0.25,
                      size_field_smooth_iters=6):
    """Run the C++ extract_quad_mesh binary once.

    Returns True on success, False on failure.
    """
    cmd = [
        QUAD_EXTRACT_BIN,
        os.path.abspath(mesh_path),
        os.path.abspath(crossfield_path),
        os.path.abspath(output_path),
        str(gradient_size),
        str(stiffness),
        str(int(direct_round)),
        str(iters),
        str(local_iters),
    ]

    tag = ""
    if stiffness != 5.0 or direct_round or iters != 5:
        tag = f"  [stiff={stiffness} dr={int(direct_round)} it={iters}]"
    print(f"  binary       : {QUAD_EXTRACT_BIN}")
    print(f"  input mesh   : {mesh_path}")
    print(f"  cross field  : {crossfield_path}")
    print(f"  output       : {output_path}")
    print(f"  gradient_size: {gradient_size}  timeout: {timeout}s{tag}\n")
    if size_field_path is not None:
        print("  warning      : --size_field is deprecated and ignored")
        print(f"  ignored_path : {size_field_path}\n")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        "/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", ""))

    try:
        subprocess.run(cmd, env=env, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {e}")
        return False

    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        return True
    return False


_FALLBACK_CONFIGS = [
    {"stiffness": 5.0,   "direct_round": False, "iters": 5},
    {"stiffness": 50.0,  "direct_round": False, "iters": 5},
    {"stiffness": 5.0,   "direct_round": True,  "iters": 5},
    {"stiffness": 50.0,  "direct_round": True,  "iters": 5},
    {"stiffness": 200.0, "direct_round": True,  "iters": 3},
]


def run_extract(mesh_path, crossfield_path, output_path,
                gradient_size=30.0, timeout=600, size_field_path=None,
                size_field_strength=0.25, size_field_smooth_iters=6,
                size_field_relax=False, max_fallback_attempts=None):
    """Run extraction with automatic fallback through increasingly relaxed
    MIQ configurations when the default parameters crash or timeout."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if os.path.isfile(output_path):
        os.remove(output_path)

    fallback_configs = _FALLBACK_CONFIGS
    if max_fallback_attempts is not None:
        max_fallback_attempts = max(1, int(max_fallback_attempts))
        fallback_configs = fallback_configs[:max_fallback_attempts]

    for ci, cfg in enumerate(fallback_configs):
        ok = _run_extract_once(
            mesh_path, crossfield_path, output_path,
            gradient_size=gradient_size,
            timeout=timeout,
            stiffness=cfg["stiffness"],
            direct_round=cfg["direct_round"],
            iters=cfg["iters"],
        )
        if ok:
            if ci > 0:
                print(f"  OK (fallback level {ci})")
            return True
    return False


def sanitize_float_tag(value):
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def default_sweep_values(base_value):
    """Return a compact set of sweep candidates around *base_value*."""
    multipliers = [0.25, 0.4, 0.55, 0.75, 1.0, 1.35, 1.8, 2.4]
    values = []
    seen = set()
    for m in multipliers:
        v = round(base_value * m, 4)
        if v <= 0:
            continue
        key = round(v, 6)
        if key in seen:
            continue
        seen.add(key)
        values.append(v)
    return values


def evaluate_quad_mesh(quad_path):
    """Return basic geometric metrics for a quad OBJ."""
    quad_faces = _load_quad_faces_from_obj(quad_path)
    if quad_faces is None or len(quad_faces) == 0:
        raise ValueError(f"No valid quad faces found in {quad_path}")

    mesh = trimesh.load(quad_path, process=False)
    vertices = mesh.vertices
    ad = compute_angle_distortion(vertices, quad_faces)
    jr = compute_jacobian_ratio(vertices, quad_faces)

    return {
        "n_quads": int(len(quad_faces)),
        "n_verts": int(len(vertices)),
        "AD_mean_deg": float(ad["mean_deviation_deg"]),
        "AD_max_deg": float(ad["max_deviation_deg"]),
        "JR_mean": float(jr["mean_jr"]),
        "JR_min": float(jr["min_jr"]),
        "JR_std": float(jr["std_jr"]),
        "SJ_mean": float(jr["mean_scaled_jacobian"]),
    }


def load_quad_obj(path):
    vertices = []
    quads = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face = [int(p.split("/")[0]) - 1 for p in parts]
                if len(face) == 4:
                    quads.append(face)

    if not vertices or not quads:
        raise ValueError(f"No valid quad OBJ data found in {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(quads, dtype=np.int64)


def write_quad_obj(path, vertices, quads):
    with open(path, "w") as f:
        for v in np.asarray(vertices, dtype=np.float64):
            f.write(f"v {v[0]:.12g} {v[1]:.12g} {v[2]:.12g}\n")
        for q in np.asarray(quads, dtype=np.int64):
            f.write(f"f {q[0] + 1} {q[1] + 1} {q[2] + 1} {q[3] + 1}\n")


def catmull_clark_subdivide(vertices, quads):
    vertices = np.asarray(vertices, dtype=np.float64)
    quads = np.asarray(quads, dtype=np.int64)

    face_points = vertices[quads].mean(axis=1)

    edge_to_faces = {}
    vertex_neighbors = [set() for _ in range(len(vertices))]
    vertex_faces = [[] for _ in range(len(vertices))]
    boundary_neighbors = [set() for _ in range(len(vertices))]

    for face_idx, face in enumerate(quads):
        for local_idx in range(4):
            v0 = int(face[local_idx])
            v1 = int(face[(local_idx + 1) % 4])
            edge = tuple(sorted((v0, v1)))
            edge_to_faces.setdefault(edge, []).append(face_idx)
            vertex_neighbors[v0].add(v1)
            vertex_neighbors[v1].add(v0)
        for vid in face:
            vertex_faces[int(vid)].append(face_idx)

    edge_points = {}
    for edge, adj_faces in edge_to_faces.items():
        v0, v1 = edge
        if len(adj_faces) == 2:
            point = (
                vertices[v0] + vertices[v1] +
                face_points[adj_faces[0]] + face_points[adj_faces[1]]
            ) / 4.0
        else:
            point = (vertices[v0] + vertices[v1]) / 2.0
            boundary_neighbors[v0].add(v1)
            boundary_neighbors[v1].add(v0)
        edge_points[edge] = point

    new_vertex_positions = np.zeros_like(vertices)
    for vid in range(len(vertices)):
        boundary_nbrs = sorted(boundary_neighbors[vid])
        if len(boundary_nbrs) >= 2:
            new_vertex_positions[vid] = (
                0.75 * vertices[vid] +
                0.125 * vertices[boundary_nbrs[0]] +
                0.125 * vertices[boundary_nbrs[1]]
            )
            continue

        faces = np.unique(vertex_faces[vid])
        n = len(faces)
        if n == 0:
            new_vertex_positions[vid] = vertices[vid]
            continue
        f_avg = face_points[faces].mean(axis=0)
        edge_midpoints = []
        for nbr in vertex_neighbors[vid]:
            edge_midpoints.append(0.5 * (vertices[vid] + vertices[nbr]))
        r_avg = np.mean(np.asarray(edge_midpoints, dtype=np.float64), axis=0)
        new_vertex_positions[vid] = (f_avg + 2.0 * r_avg + (n - 3.0) * vertices[vid]) / n

    new_vertices = new_vertex_positions.tolist()
    edge_index = {}
    for edge, point in edge_points.items():
        edge_index[edge] = len(new_vertices)
        new_vertices.append(point.tolist())

    face_index = []
    for point in face_points:
        face_index.append(len(new_vertices))
        new_vertices.append(point.tolist())

    new_quads = []
    for face_idx, face in enumerate(quads):
        fp = face_index[face_idx]
        for local_idx in range(4):
            v_curr = int(face[local_idx])
            v_next = int(face[(local_idx + 1) % 4])
            v_prev = int(face[(local_idx - 1) % 4])
            e_next = edge_index[tuple(sorted((v_curr, v_next)))]
            e_prev = edge_index[tuple(sorted((v_prev, v_curr)))]
            new_quads.append([v_curr, e_next, fp, e_prev])

    return np.asarray(new_vertices, dtype=np.float64), np.asarray(new_quads, dtype=np.int64)


def choose_catmull_clark_iters(current_quads, target_quads, max_iters):
    if target_quads is None or target_quads <= 0 or current_quads <= 0 or max_iters <= 0:
        return 0

    best_iter = 0
    best_diff = abs(current_quads - target_quads)
    for iters in range(1, max_iters + 1):
        candidate = current_quads * (4 ** iters)
        diff = abs(candidate - target_quads)
        if diff < best_diff:
            best_diff = diff
            best_iter = iters
    return best_iter


def maybe_apply_catmull_clark(quad_path, mesh_path, args):
    if not os.path.isfile(quad_path):
        return None

    quad_faces = _load_quad_faces_from_obj(quad_path)
    if quad_faces is None or len(quad_faces) == 0:
        return None

    target_quads = None
    if args.target_quad_ratio > 0:
        target_quads = int(round(count_triangle_faces(mesh_path) * args.target_quad_ratio))

    auto_iters = choose_catmull_clark_iters(
        len(quad_faces), target_quads, args.max_catmull_clark_iters)
    total_iters = max(args.catmull_clark_iters, auto_iters)
    if total_iters <= 0:
        return None

    print(f"[Catmull-Clark] Applying {total_iters} iteration(s) to {os.path.basename(quad_path)}")
    if target_quads is not None:
        print(f"  target_quads : {target_quads}")

    vertices, quads = load_quad_obj(quad_path)
    for _ in range(total_iters):
        vertices, quads = catmull_clark_subdivide(vertices, quads)
    write_quad_obj(quad_path, vertices, quads)

    metrics = evaluate_quad_mesh(quad_path)
    print(
        f"  subdivided   : n_quads={metrics['n_quads']} "
        f"AD_mean={metrics['AD_mean_deg']:.3f}deg JR_min={metrics['JR_min']:.4f}"
    )
    return {
        "catmull_clark_iters": int(total_iters),
        "target_quads": None if target_quads is None else int(target_quads),
        "n_quads_after_subdivision": int(metrics["n_quads"]),
    }


def count_triangle_faces(mesh_path):
    mesh = trimesh.load(mesh_path, process=False)
    return int(len(mesh.faces))


def build_candidate_output_path(output_path, gradient_size, output_dir=None):
    base, ext = os.path.splitext(output_path)
    candidate_name = f"{os.path.basename(base)}_gs{sanitize_float_tag(gradient_size)}{ext}"
    if output_dir is None:
        return os.path.join(os.path.dirname(output_path), candidate_name)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, candidate_name)


def get_min_quads_threshold(mesh_path, min_quads):
    if min_quads is not None:
        return min_quads
    tri_faces = count_triangle_faces(mesh_path)
    return max(100, int(round(0.05 * tri_faces)))


def candidate_violation(metrics, min_quads, max_ad, min_jr):
    quad_deficit = max(0, min_quads - metrics["n_quads"])
    ad_excess = max(0.0, metrics["AD_mean_deg"] - max_ad)
    jr_deficit = max(0.0, min_jr - metrics["JR_min"])
    return {
        "quad_deficit": quad_deficit,
        "ad_excess": ad_excess,
        "jr_deficit": jr_deficit,
        "num_failed_constraints": int(quad_deficit > 0) + int(ad_excess > 0.0) + int(jr_deficit > 0.0),
    }


def candidate_budget_terms(metrics, target_quads, slack_ratio=0.2):
    if target_quads is None or target_quads <= 0:
        return {
            "target_quads": None,
            "budget_tolerance": None,
            "quad_budget_excess": 0,
            "quad_budget_abs_diff": 0,
            "quad_budget_rel_diff": 0.0,
        }

    tolerance = max(16, int(round(float(target_quads) * float(slack_ratio))))
    diff = int(metrics["n_quads"]) - int(target_quads)
    excess = max(0, diff - tolerance)
    abs_diff = max(0, abs(diff) - tolerance)
    rel_diff = float(abs(diff)) / max(float(target_quads), 1.0)
    return {
        "target_quads": int(target_quads),
        "budget_tolerance": int(tolerance),
        "quad_budget_excess": int(excess),
        "quad_budget_abs_diff": int(abs_diff),
        "quad_budget_rel_diff": rel_diff,
    }


def rank_candidate(metrics, violation, budget_terms=None):
    """Lower tuple is better."""
    if budget_terms is None:
        budget_terms = candidate_budget_terms(metrics, target_quads=None)
    if budget_terms["target_quads"] is None:
        return (
            violation["num_failed_constraints"],
            violation["quad_deficit"],
            violation["ad_excess"],
            violation["jr_deficit"],
            -metrics["n_quads"],
            metrics["AD_mean_deg"],
            -metrics["JR_mean"],
            -metrics["SJ_mean"],
        )
    return (
        violation["num_failed_constraints"],
        violation["quad_deficit"],
        violation["ad_excess"],
        violation["jr_deficit"],
        budget_terms["quad_budget_excess"],
        budget_terms["quad_budget_abs_diff"],
        metrics["AD_mean_deg"],
        -metrics["JR_mean"],
        -metrics["SJ_mean"],
        abs(metrics["n_quads"] - (budget_terms["target_quads"] or metrics["n_quads"])),
        metrics["n_quads"],
    )


def auto_sweep_single(mesh_path, crossfield_path, output_path, args):
    """Sweep gradient_size values, evaluate each output, and keep the best."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    sweep_values = args.sweep_values or default_sweep_values(args.gradient_size)
    min_quads = get_min_quads_threshold(mesh_path, args.min_quads)
    target_quads = None
    if args.target_quad_ratio > 0:
        target_quads = int(round(count_triangle_faces(mesh_path) * args.target_quad_ratio))

    print("\n[Auto Sweep]")
    print(f"  mesh        : {mesh_path}")
    print(f"  cross field : {crossfield_path}")
    print(f"  output      : {output_path}")
    print(f"  sweep_values: {sweep_values}")
    print(f"  thresholds  : min_quads={min_quads}, max_ad={args.max_ad}, min_jr={args.min_jr}\n")
    if target_quads is not None:
        print(f"  target_quads: {target_quads} (budget-aware ranking)\n")
        print(f"  budget_tol  : {args.budget_tolerance_ratio:.3f}\n")

    candidates = []
    candidate_dir = getattr(args, "intermediate_dir", None)
    for gs in sweep_values:
        candidate_output = build_candidate_output_path(output_path, gs, candidate_dir)
        print(f"[Auto Sweep] Trying gradient_size={gs}")
        ok = run_extract(
            mesh_path,
            crossfield_path,
            candidate_output,
            gs,
            args.timeout,
            size_field_path=args.size_field,
            size_field_strength=args.size_field_strength,
            size_field_smooth_iters=args.size_field_smooth_iters,
            size_field_relax=args.size_field_relax,
            max_fallback_attempts=args.max_fallback_attempts,
        )
        if not ok:
            print(f"  FAILED: extraction failed for gs={gs}\n")
            continue

        try:
            metrics = evaluate_quad_mesh(candidate_output)
        except Exception as exc:
            print(f"  FAILED: evaluation failed for gs={gs}: {exc}\n")
            continue

        subdiv_info = maybe_apply_catmull_clark(candidate_output, mesh_path, args)
        if subdiv_info is not None:
            metrics = evaluate_quad_mesh(candidate_output)
            metrics.update(subdiv_info)

        violation = candidate_violation(metrics, min_quads, args.max_ad, args.min_jr)
        budget_terms = candidate_budget_terms(
            metrics,
            target_quads=target_quads,
            slack_ratio=args.budget_tolerance_ratio,
        )
        metrics["gradient_size"] = float(gs)
        metrics["output_path"] = os.path.abspath(candidate_output)
        metrics["constraint_violation"] = violation
        metrics["budget_terms"] = budget_terms
        metrics["is_valid"] = violation["num_failed_constraints"] == 0
        metrics["rank_key"] = list(rank_candidate(metrics, violation, budget_terms))
        candidates.append(metrics)

        status = "valid" if metrics["is_valid"] else "constraint-violating"
        print(
            f"  RESULT [{status}] "
            f"n_quads={metrics['n_quads']} "
            f"AD_mean={metrics['AD_mean_deg']:.3f}deg "
            f"JR_min={metrics['JR_min']:.4f} "
            f"SJ_mean={metrics['SJ_mean']:.4f}\n"
        )

    if not candidates:
        print("FAILED: no valid sweep candidate produced an evaluable quad mesh")
        return False

    candidates.sort(key=lambda m: tuple(m["rank_key"]))
    best = candidates[0]
    shutil.copyfile(best["output_path"], output_path)

    if not args.keep_sweep_outputs:
        best_abs = os.path.abspath(output_path)
        for candidate in candidates:
            path = candidate["output_path"]
            if os.path.abspath(path) != best_abs and os.path.isfile(path):
                os.remove(path)

    summary = {
        "mesh_path": os.path.abspath(mesh_path),
        "crossfield_path": os.path.abspath(crossfield_path),
        "output_path": os.path.abspath(output_path),
        "selection_rule": {
            "priority": [
                "fewest failed constraints",
                "smallest quad deficit",
                "smallest angle-distortion excess",
                "smallest Jacobian deficit",
                "smallest budget overshoot",
                "closest quad count to budget",
                "lowest angle distortion",
                "highest mean Jacobian ratio",
                "highest mean scaled Jacobian",
                "smaller absolute quad-count mismatch",
                "smaller quad count",
            ],
            "min_quads": min_quads,
            "max_ad": args.max_ad,
            "min_jr": args.min_jr,
            "target_quads": target_quads,
            "budget_tolerance_ratio": args.budget_tolerance_ratio,
        },
        "best": best,
        "candidates": candidates,
    }

    summary_path = args.summary_json
    if summary_path is None:
        summary_base = os.path.join(candidate_dir, os.path.splitext(os.path.basename(output_path))[0]) \
            if candidate_dir is not None else os.path.splitext(output_path)[0]
        base = summary_base
        summary_path = f"{base}_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("[Auto Sweep] Best candidate selected")
    print(f"  gradient_size: {best['gradient_size']}")
    print(f"  n_quads      : {best['n_quads']}")
    print(f"  AD_mean_deg  : {best['AD_mean_deg']:.3f}")
    print(f"  JR_min       : {best['JR_min']:.4f}")
    print(f"  saved output : {output_path}")
    print(f"  saved summary: {summary_path}")
    return True


def _apply_post_extraction(output_path, mesh_path, original_mesh_path, args):
    """Apply optional post-extraction steps (Catmull-Clark, stitch, snap)."""
    if args is None:
        return
    maybe_apply_catmull_clark(output_path, mesh_path, args)
    if getattr(args, 'stitch_quad_patches', False):
        stitch_quad_patches(
            output_path, original_mesh_path,
            merge_threshold_ratio=getattr(args, 'stitch_threshold_ratio', 0.005),
            smooth_iterations=getattr(args, 'stitch_smooth_iters', 2),
            smooth_weight=getattr(args, 'stitch_smooth_weight', 0.2),
        )
    if getattr(args, 'snap_to_boundary', False):
        snap_quad_to_semantic_boundary(
            output_path, mesh_path,
            getattr(args, 'snap_feat_path', None),
            snap_fraction=getattr(args, 'snap_fraction', 0.3),
            snap_threshold_ratio=getattr(args, 'snap_threshold_ratio', 0.01),
            min_angle_deg=getattr(args, 'snap_min_angle', 25.0),
            smooth_iterations=getattr(args, 'snap_smooth_iters', 3),
            smooth_weight=getattr(args, 'snap_smooth_weight', 0.3),
        )


def _attempt_extraction(mesh_path, crossfield_path, output_path,
                        original_mesh_path, gradient_size, timeout,
                        retry, args):
    """Core extraction attempt using a pre-repaired mesh and cross-field.

    Tries auto_sweep, single-shot, or multi-retry depending on args.
    Returns True on success (output_path exists with valid quad mesh).
    """
    if args is not None and args.auto_sweep:
        ok = auto_sweep_single(mesh_path, crossfield_path, output_path, args)
        if ok:
            _apply_post_extraction(output_path, mesh_path,
                                   original_mesh_path, args)
        return ok

    if not retry:
        print(f"\n[Quad Extraction]")
        ok = run_extract(mesh_path, crossfield_path, output_path,
                         gradient_size, timeout,
                         size_field_path=args.size_field if args is not None else None,
                         size_field_strength=args.size_field_strength if args is not None else 0.25,
                         size_field_smooth_iters=args.size_field_smooth_iters if args is not None else 6,
                         size_field_relax=args.size_field_relax if args is not None else False,
                         max_fallback_attempts=(
                             args.max_fallback_attempts if args is not None else None))
        if ok:
            _apply_post_extraction(output_path, mesh_path,
                                   original_mesh_path, args)
        return ok

    cf_dir = os.path.dirname(crossfield_path)
    cf_files = sorted(
        glob.glob(os.path.join(cf_dir, "*_iter_*.txt")),
        key=lambda f: int(
            os.path.splitext(os.path.basename(f))[0].rsplit("_iter_", 1)[1]),
        reverse=True,
    )
    if not cf_files:
        cf_files = [crossfield_path]

    gradient_sizes = [gradient_size, 50.0, 15.0, 80.0]
    seen = set()
    gradient_sizes = [g for g in gradient_sizes
                      if not (g in seen or seen.add(g))]

    for cf_txt in cf_files[:3]:
        cf_label = os.path.basename(cf_txt)
        for gs in gradient_sizes:
            print(f"\n[Quad Extraction] Trying {cf_label}  gs={gs}")
            ok = run_extract(
                mesh_path, cf_txt, output_path, gs, timeout,
                size_field_path=args.size_field if args is not None else None,
                size_field_strength=args.size_field_strength if args is not None else 0.25,
                size_field_smooth_iters=args.size_field_smooth_iters if args is not None else 6,
                size_field_relax=args.size_field_relax if args is not None else False,
                max_fallback_attempts=(
                    args.max_fallback_attempts if args is not None else None),
            )
            if ok:
                _apply_post_extraction(output_path, mesh_path,
                                       original_mesh_path, args)
                return True
    return False


def extract_single(mesh_path, crossfield_path, output_path,
                   gradient_size, timeout, retry, args=None):
    """Extract a quad mesh with cascading repair tiers on failure.

    Tier 0: existing repair (weld + NM-fix + bridge + degenerate)
    Tier 1: PyMeshLab manifold repair (from original mesh)
    Tier 2: PyMeshLab isotropic remeshing (from original mesh)
    Tier 3: SDF reconstruction via mesh2sdf + marching cubes
    """
    original_mesh_path = mesh_path
    max_repair_tiers = 4 if args is None else max(1, min(4, int(args.max_repair_tiers)))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    base_dir = os.path.dirname(os.path.abspath(output_path))
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    work_dir = getattr(args, "intermediate_dir", None) if args is not None else None
    if work_dir is None:
        work_dir = os.path.join(base_dir, base_name + "_extract_work")
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    if args is not None:
        args.intermediate_dir = work_dir

    mesh_path = ensure_obj_mesh(mesh_path, os.path.join(work_dir, base_name + ".obj"))
    original_obj_path = mesh_path
    original_cf_path = crossfield_path

    # ---- Tier 0: existing repair pipeline ----
    print(f"\n{'='*50}")
    print(f"  [Tier 0] Standard repair pipeline")
    print(f"{'='*50}")

    welded_obj = os.path.join(work_dir, base_name + "_welded_input.obj")
    mesh_path, did_weld = weld_mesh_vertices(mesh_path, welded_obj)

    nm_mesh = os.path.join(work_dir, base_name + "_nm_fixed.obj")
    nm_cf = os.path.join(work_dir, base_name + "_nm_fixed_cf.txt")
    mesh_path, crossfield_path, n_nm = resolve_nonmanifold(
        mesh_path, crossfield_path, nm_mesh, nm_cf)

    bridged_mesh = os.path.join(work_dir, base_name + "_bridged_input.obj")
    bridged_cf = os.path.join(work_dir, base_name + "_bridged_crossfield.txt")
    mesh_path, crossfield_path, n_bridge, _ = bridge_connected_components(
        mesh_path, crossfield_path, bridged_mesh, bridged_cf)

    repaired_mesh = os.path.join(work_dir, base_name + "_repaired.obj")
    repaired_cf = os.path.join(work_dir, base_name + "_repaired_cf.txt")
    mesh_path, crossfield_path, _ = repair_degenerate_triangles(
        mesh_path, crossfield_path, repaired_mesh, repaired_cf, min_angle_deg=1.0)

    ok = _attempt_extraction(mesh_path, crossfield_path, output_path,
                             original_mesh_path, gradient_size, timeout,
                             retry, args)
    if ok:
        print(f"SUCCESS (Tier 0) -> {output_path}")
        _cleanup_intermediate_dir(work_dir, getattr(args, "keep_intermediates", False))
        return True
    if max_repair_tiers <= 1:
        print(f"FAILED: repair tier limit ({max_repair_tiers}) reached for {mesh_path}")
        return False

    # ---- Tier 1: PyMeshLab manifold repair ----
    print(f"\n{'='*50}")
    print(f"  [Tier 1] PyMeshLab manifold repair")
    print(f"{'='*50}")

    pml_mesh = os.path.join(work_dir, base_name + "_pml_repaired.obj")
    pml_cf = os.path.join(work_dir, base_name + "_pml_repaired_cf.txt")

    pml_mesh_path, face_changed = pymeshlab_manifold_repair(
        original_obj_path, pml_mesh)
    if pml_mesh_path != original_obj_path:
        if face_changed:
            transfer_crossfield(original_obj_path, original_cf_path,
                                pml_mesh_path, pml_cf)
        else:
            shutil.copy2(original_cf_path, pml_cf)

        ok = _attempt_extraction(pml_mesh_path, pml_cf, output_path,
                                 original_mesh_path, gradient_size, timeout,
                                 retry, args)
        if ok:
            print(f"SUCCESS (Tier 1 - PyMeshLab repair) -> {output_path}")
            _cleanup_intermediate_dir(work_dir, getattr(args, "keep_intermediates", False))
            return True
    if max_repair_tiers <= 2:
        print(f"FAILED: repair tier limit ({max_repair_tiers}) reached for {mesh_path}")
        return False

    # ---- Tier 2: PyMeshLab isotropic remeshing ----
    print(f"\n{'='*50}")
    print(f"  [Tier 2] PyMeshLab isotropic remeshing")
    print(f"{'='*50}")

    iso_mesh = os.path.join(work_dir, base_name + "_pml_isomesh.obj")
    iso_cf = os.path.join(work_dir, base_name + "_pml_isomesh_cf.txt")

    iso_mesh_path = pymeshlab_isotropic_remesh(original_obj_path, iso_mesh)
    if iso_mesh_path != original_obj_path:
        transfer_crossfield(original_obj_path, original_cf_path,
                            iso_mesh_path, iso_cf)
        ok = _attempt_extraction(iso_mesh_path, iso_cf, output_path,
                                 original_mesh_path, gradient_size, timeout,
                                 retry, args)
        if ok:
            print(f"SUCCESS (Tier 2 - isotropic remesh) -> {output_path}")
            _cleanup_intermediate_dir(work_dir, getattr(args, "keep_intermediates", False))
            return True
    if max_repair_tiers <= 3:
        print(f"FAILED: repair tier limit ({max_repair_tiers}) reached for {mesh_path}")
        return False

    # ---- Tier 3: SDF reconstruction ----
    print(f"\n{'='*50}")
    print(f"  [Tier 3] SDF reconstruction (mesh2sdf + marching cubes)")
    print(f"{'='*50}")

    sdf_mesh = os.path.join(work_dir, base_name + "_sdf_recon.obj")
    sdf_cf = os.path.join(work_dir, base_name + "_sdf_recon_cf.txt")

    sdf_mesh_path = sdf_reconstruct(original_obj_path, sdf_mesh)
    if sdf_mesh_path != original_obj_path:
        transfer_crossfield(original_obj_path, original_cf_path,
                            sdf_mesh_path, sdf_cf)
        ok = _attempt_extraction(sdf_mesh_path, sdf_cf, output_path,
                                 original_mesh_path, gradient_size, timeout,
                                 retry, args)
        if ok:
            print(f"SUCCESS (Tier 3 - SDF reconstruction) -> {output_path}")
            _cleanup_intermediate_dir(work_dir, getattr(args, "keep_intermediates", False))
            return True

    print(f"FAILED: all repair tiers exhausted for {mesh_path}")
    print(f"  intermediates kept for debugging: {work_dir}")
    return False


def compute_semantic_boundary_samples(tri_mesh, labels, samples_per_edge=3):
    """Sample points along semantic boundary edges of the original triangle mesh.

    A boundary edge is one shared by two faces with different semantic labels.
    Returns an (M, 3) array of 3D points lying on semantic boundaries.
    """
    faces = tri_mesh.faces
    face_adj = tri_mesh.face_adjacency
    face_adj_edges = tri_mesh.face_adjacency_edges

    boundary_mask = labels[face_adj[:, 0]] != labels[face_adj[:, 1]]
    boundary_edge_verts = face_adj_edges[boundary_mask]

    if len(boundary_edge_verts) == 0:
        return np.empty((0, 3), dtype=np.float64)

    verts = tri_mesh.vertices
    v0 = verts[boundary_edge_verts[:, 0]]
    v1 = verts[boundary_edge_verts[:, 1]]

    samples = []
    for t in np.linspace(0.0, 1.0, samples_per_edge):
        samples.append(v0 + t * (v1 - v0))
    return np.concatenate(samples, axis=0)


def _quad_min_angle(verts, quad):
    """Return the minimum interior angle (radians) of a quad."""
    angles = []
    for k in range(4):
        p0 = verts[quad[(k - 1) % 4]]
        p1 = verts[quad[k]]
        p2 = verts[quad[(k + 1) % 4]]
        e1 = p0 - p1
        e2 = p2 - p1
        n1 = np.linalg.norm(e1)
        n2 = np.linalg.norm(e2)
        if n1 < 1e-15 or n2 < 1e-15:
            return 0.0
        cos_a = np.clip(np.dot(e1, e2) / (n1 * n2), -1.0, 1.0)
        angles.append(np.arccos(cos_a))
    return min(angles)


def _build_vertex_to_quads(n_verts, quads):
    v2q = [[] for _ in range(n_verts)]
    for qi, q in enumerate(quads):
        for vi in q:
            v2q[int(vi)].append(qi)
    return v2q


def _laplacian_smooth_quad(verts, quads, fixed_mask, iterations=3, weight=0.3):
    """Laplacian smoothing on quad mesh, keeping fixed_mask vertices in place."""
    adj = [set() for _ in range(len(verts))]
    for q in quads:
        for k in range(4):
            a, b = int(q[k]), int(q[(k + 1) % 4])
            adj[a].add(b)
            adj[b].add(a)

    for _ in range(iterations):
        new_verts = verts.copy()
        for vi in range(len(verts)):
            if fixed_mask[vi] or len(adj[vi]) == 0:
                continue
            neighbors = np.array(list(adj[vi]), dtype=np.intp)
            centroid = verts[neighbors].mean(axis=0)
            new_verts[vi] = verts[vi] + weight * (centroid - verts[vi])
        verts = new_verts
    return verts


def snap_quad_to_semantic_boundary(quad_path, tri_mesh_path, feat_path,
                                   snap_fraction=0.3, snap_threshold_ratio=0.01,
                                   samples_per_edge=5, min_angle_deg=25.0,
                                   smooth_iterations=3, smooth_weight=0.3):
    """Move quad vertices toward nearby semantic boundaries, with quality control.

    Only snaps a vertex if the resulting minimum quad angle stays above
    min_angle_deg. After snapping, applies Laplacian smoothing to non-snapped
    interior vertices to distribute distortion evenly.

    Args:
        quad_path: path to quad OBJ (read & overwritten in-place)
        tri_mesh_path: path to the original triangle mesh
        feat_path: path to PartField features (.npy)
        snap_fraction: how far to move toward boundary (0=no move, 1=full snap)
        snap_threshold_ratio: max snap distance as fraction of bounding box diagonal
        samples_per_edge: boundary sampling density per edge
        min_angle_deg: reject a snap if any incident quad angle would drop below this
        smooth_iterations: Laplacian smoothing passes after snapping
        smooth_weight: Laplacian smoothing step size

    Returns:
        dict with snapping statistics, or None if snapping was skipped.
    """
    if not os.path.isfile(quad_path):
        return None
    if feat_path is None or not os.path.isfile(feat_path):
        print("  [snap] skipped: no feature file provided")
        return None

    tri_mesh = trimesh.load_mesh(tri_mesh_path, process=False)
    features = np.load(feat_path)
    if len(features) != len(tri_mesh.faces):
        print(f"  [snap] skipped: feature count ({len(features)}) != face count ({len(tri_mesh.faces)})")
        return None

    try:
        from eval.label_utils import cluster_features
        result = cluster_features(features, method="best_silhouette")
        labels = result["labels"]
    except Exception as exc:
        print(f"  [snap] skipped: clustering failed: {exc}")
        return None

    boundary_pts = compute_semantic_boundary_samples(
        tri_mesh, labels, samples_per_edge=samples_per_edge)
    if len(boundary_pts) == 0:
        print("  [snap] skipped: no semantic boundaries found")
        return None

    tree = cKDTree(boundary_pts)
    bbox_diag = np.linalg.norm(tri_mesh.vertices.max(axis=0) - tri_mesh.vertices.min(axis=0))
    threshold = bbox_diag * snap_threshold_ratio

    quad_verts, quad_faces = load_quad_obj(quad_path)
    distances, indices = tree.query(quad_verts)
    candidate_mask = distances < threshold

    if not np.any(candidate_mask):
        print(f"  [snap] no vertices within threshold ({threshold:.6f})")
        return {"n_snapped": 0, "n_rejected": 0, "threshold": float(threshold)}

    v2q = _build_vertex_to_quads(len(quad_verts), quad_faces)
    min_angle_rad = np.deg2rad(min_angle_deg)

    n_snapped = 0
    n_rejected = 0
    snapped_mask = np.zeros(len(quad_verts), dtype=bool)

    for vi in np.flatnonzero(candidate_mask):
        target = boundary_pts[indices[vi]]
        new_pos = quad_verts[vi] + snap_fraction * (target - quad_verts[vi])

        old_pos = quad_verts[vi].copy()
        quad_verts[vi] = new_pos
        ok = True
        for qi in v2q[vi]:
            if _quad_min_angle(quad_verts, quad_faces[qi]) < min_angle_rad:
                ok = False
                break
        if ok:
            n_snapped += 1
            snapped_mask[vi] = True
        else:
            quad_verts[vi] = old_pos
            n_rejected += 1

    if smooth_iterations > 0 and n_snapped > 0:
        quad_verts = _laplacian_smooth_quad(
            quad_verts, quad_faces, fixed_mask=snapped_mask,
            iterations=smooth_iterations, weight=smooth_weight)

    write_quad_obj(quad_path, quad_verts, quad_faces)

    print(f"  [snap] moved {n_snapped}/{len(quad_verts)} vertices, "
          f"rejected {n_rejected} (min_angle>{min_angle_deg}deg), "
          f"smoothed {smooth_iterations} iters")
    return {
        "n_snapped": n_snapped,
        "n_rejected": n_rejected,
        "n_total_verts": len(quad_verts),
        "threshold": float(threshold),
        "snap_fraction": float(snap_fraction),
        "min_angle_deg": float(min_angle_deg),
        "smooth_iterations": int(smooth_iterations),
        "mean_snap_dist": float(distances[candidate_mask].mean()),
    }


def _quad_vertex_connected_components(n_verts, quads):
    """Return component label per vertex and component count for a quad mesh."""
    adj = [set() for _ in range(n_verts)]
    for q in quads:
        for k in range(4):
            a, b = int(q[k]), int(q[(k + 1) % 4])
            adj[a].add(b)
            adj[b].add(a)
    labels = np.full(n_verts, -1, dtype=np.int64)
    comp_id = 0
    for seed in range(n_verts):
        if labels[seed] >= 0:
            continue
        stack = [seed]
        labels[seed] = comp_id
        while stack:
            v = stack.pop()
            for nb in adj[v]:
                if labels[nb] < 0:
                    labels[nb] = comp_id
                    stack.append(nb)
        comp_id += 1
    return labels, comp_id


def stitch_quad_patches(quad_path, tri_mesh_path, merge_threshold_ratio=0.005,
                        smooth_iterations=2, smooth_weight=0.2):
    """Merge disconnected quad patches into a single connected mesh.

    When the input triangle mesh is a single connected component but the
    quad extraction produced multiple disconnected patches, this function
    merges nearby vertices across patch boundaries to stitch them together.

    Args:
        quad_path: path to quad OBJ (read & overwritten in-place)
        tri_mesh_path: path to the original triangle mesh (for bbox reference)
        merge_threshold_ratio: max merge distance as fraction of bbox diagonal
        smooth_iterations: Laplacian smoothing passes on stitched vertices
        smooth_weight: Laplacian smoothing step size

    Returns:
        dict with stitching statistics, or None if no stitching needed.
    """
    if not os.path.isfile(quad_path):
        return None

    vertices, quads = load_quad_obj(quad_path)
    labels, n_comp = _quad_vertex_connected_components(len(vertices), quads)
    if n_comp <= 1:
        return None

    # Check that the input triangle mesh is a single connected component
    tri_mesh = trimesh.load_mesh(tri_mesh_path, process=False)
    tri_labels, tri_n_comp = _vertex_connected_components(
        tri_mesh.vertices, tri_mesh.faces)
    if tri_n_comp > 1:
        print(f"  [stitch] skipped: input triangle mesh has {tri_n_comp} components")
        return None

    bbox_diag = np.linalg.norm(
        vertices.max(axis=0) - vertices.min(axis=0))
    threshold = bbox_diag * merge_threshold_ratio

    print(f"  [stitch] quad mesh has {n_comp} components, "
          f"attempting to merge (threshold={threshold:.6f})")

    # Build per-component boundary vertex sets (vertices on edges used by
    # only one quad, i.e., patch boundary)
    edge_count = {}
    for qi, q in enumerate(quads):
        for k in range(4):
            e = tuple(sorted((int(q[k]), int(q[(k + 1) % 4]))))
            edge_count[e] = edge_count.get(e, 0) + 1
    boundary_verts = set()
    for (a, b), c in edge_count.items():
        if c == 1:
            boundary_verts.add(a)
            boundary_verts.add(b)

    # For each component, collect boundary vertices
    comp_boundary = {}
    for vi in boundary_verts:
        c = int(labels[vi])
        comp_boundary.setdefault(c, []).append(vi)

    # Build a KD-tree of all boundary vertices and find merge pairs
    all_boundary = sorted(boundary_verts)
    if len(all_boundary) < 2:
        return None
    bv_positions = vertices[all_boundary]
    tree = cKDTree(bv_positions)

    # Find pairs of nearby vertices from DIFFERENT components
    merge_map = {}  # vertex -> canonical vertex it maps to
    merged_count = 0
    pairs = tree.query_pairs(r=threshold)
    for i, j in pairs:
        vi = all_boundary[i]
        vj = all_boundary[j]
        if labels[vi] == labels[vj]:
            continue
        # Find canonical representatives
        while vi in merge_map:
            vi = merge_map[vi]
        while vj in merge_map:
            vj = merge_map[vj]
        if vi == vj:
            continue
        # Merge vj into vi (average positions)
        avg_pos = (vertices[vi] + vertices[vj]) * 0.5
        vertices[vi] = avg_pos
        merge_map[vj] = vi
        merged_count += 1

    if merged_count == 0:
        print(f"  [stitch] no mergeable vertex pairs found within threshold")
        return None

    # Remap quad faces
    def canonical(v):
        while v in merge_map:
            v = merge_map[v]
        return v

    new_quads = []
    for q in quads:
        nq = [canonical(int(q[k])) for k in range(4)]
        # Skip degenerate quads where merging collapsed vertices
        if len(set(nq)) < 4:
            continue
        new_quads.append(nq)

    if not new_quads:
        print(f"  [stitch] all quads became degenerate after merging")
        return None

    new_quads = np.asarray(new_quads, dtype=np.int64)

    # Compact vertex indices (remove unused vertices)
    used_verts = np.unique(new_quads.reshape(-1))
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new[used_verts] = np.arange(len(used_verts), dtype=np.int64)
    new_vertices = vertices[used_verts]
    new_quads = old_to_new[new_quads]

    # Verify component count after stitching
    labels2, n_comp2 = _quad_vertex_connected_components(
        len(new_vertices), new_quads)

    # Optional smoothing near stitched vertices
    if smooth_iterations > 0:
        stitched_new = set()
        for vj, vi in merge_map.items():
            ci = canonical(vi)
            if ci in used_verts.tolist():
                new_idx = int(old_to_new[ci])
                stitched_new.add(new_idx)
        fixed_mask = np.ones(len(new_vertices), dtype=bool)
        for vi in stitched_new:
            fixed_mask[vi] = False
            # Also unfix immediate neighbors for smoother result
        new_vertices = _laplacian_smooth_quad(
            new_vertices, new_quads, fixed_mask=fixed_mask,
            iterations=smooth_iterations, weight=smooth_weight)

    write_quad_obj(quad_path, new_vertices, new_quads)

    print(f"  [stitch] merged {merged_count} vertex pairs, "
          f"components: {n_comp} -> {n_comp2}, "
          f"quads: {len(quads)} -> {len(new_quads)}, "
          f"verts: {len(vertices)} -> {len(new_vertices)}")

    return {
        "n_merged": merged_count,
        "components_before": n_comp,
        "components_after": n_comp2,
        "quads_before": len(quads),
        "quads_after": len(new_quads),
        "verts_before": len(vertices),
        "verts_after": len(new_vertices),
    }


def main():
    args = parse_args()

    if not os.path.isfile(QUAD_EXTRACT_BIN):
        sys.exit(
            f"Quad extraction binary not found: {QUAD_EXTRACT_BIN}\n"
            f"  Build it with:\n"
            f"    cd libQEx/build && cmake .. -DCMAKE_CXX_FLAGS='-DNDEBUG' && make\n"
            f"    cd quad_extract/build && cmake .. -DCMAKE_CXX_FLAGS='-DNDEBUG -O2' && make")

    # --- Single mesh mode ---
    if args.mesh:
        if not os.path.isfile(args.mesh):
            sys.exit(f"Mesh not found: {args.mesh}")

        if args.crossfield:
            cf_path = args.crossfield
        elif args.crossfield_dir:
            cf_path = find_latest_crossfield(args.crossfield_dir)
            if cf_path is None:
                sys.exit(f"No *_iter_*.txt files found in {args.crossfield_dir}")
            print(f"Auto-selected cross field: {cf_path}")
        elif args.crossfield_root:
            basename = os.path.splitext(os.path.basename(args.mesh))[0]
            cf_dir = os.path.join(args.crossfield_root, basename, "save_crossField")
            cf_path = find_latest_crossfield(cf_dir)
            if cf_path is None:
                sys.exit(f"No cross-field files in {cf_dir}")
            print(f"Auto-selected cross field: {cf_path}")
        else:
            sys.exit("Provide --crossfield, --crossfield_dir, or --crossfield_root")

        if not os.path.isfile(cf_path):
            sys.exit(f"Cross-field file not found: {cf_path}")

        basename = os.path.splitext(os.path.basename(args.mesh))[0]
        output = args.output or f"{basename}_quad.obj"

        ok = extract_single(args.mesh, cf_path, output,
                            args.gradient_size, args.timeout, args.retry, args=args)
        sys.exit(0 if ok else 1)

    # --- Batch mode ---
    mesh_dir = os.path.abspath(args.mesh_dir)
    if not os.path.isdir(mesh_dir):
        sys.exit(f"Mesh directory not found: {mesh_dir}")

    if not args.crossfield_root:
        sys.exit("Batch mode (--mesh_dir) requires --crossfield_root")

    output_dir = args.output_dir or "quad_meshes_output"
    os.makedirs(output_dir, exist_ok=True)

    mesh_files = sorted(
        f for f in os.listdir(mesh_dir)
        if os.path.splitext(f)[1].lower() in ALL_MESH_EXTS
    )
    if not mesh_files:
        sys.exit(f"No mesh files found in {mesh_dir}")

    n = len(mesh_files)
    success, failed = 0, 0

    for idx, mf in enumerate(mesh_files, 1):
        basename = os.path.splitext(mf)[0]
        mesh_path = os.path.join(mesh_dir, mf)

        cf_dir = os.path.join(
            os.path.abspath(args.crossfield_root), basename, "save_crossField")
        cf_path = find_latest_crossfield(cf_dir)

        print(f"\n{'='*60}")
        print(f"  [{idx}/{n}] {basename}")
        print(f"{'='*60}")

        if cf_path is None:
            print(f"  SKIP: no cross-field found in {cf_dir}")
            failed += 1
            continue

        output_path = os.path.join(output_dir, f"{basename}_quad.obj")
        ok = extract_single(mesh_path, cf_path, output_path,
                            args.gradient_size, args.timeout, args.retry, args=args)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Done: {success} succeeded, {failed} failed out of {n}")
    print(f"  Output directory: {os.path.abspath(output_dir)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
