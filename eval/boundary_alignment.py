"""
Boundary Alignment Error (BAE).

Measures how well quad mesh edges align with semantic part boundaries on
the original triangle mesh.

Algorithm
---------
1. Identify semantic boundaries on the original triangle mesh: edges
   whose two adjacent faces carry different part labels.
2. Sample points along those boundary edges.
3. Identify quad mesh edges.
4. For every boundary sample point, compute the shortest distance to
   any quad edge (point-to-segment distance).
5. BAE = mean of these distances (lower is better).

An optional reverse direction (quad edges to boundary) can also be
computed for the Chamfer-style symmetric variant.
"""

import numpy as np
from scipy.spatial import cKDTree


def _edges_from_faces(faces):
    """Return unique undirected edges (N_edges, 2) from faces of any polygon size."""
    n_sides = faces.shape[1]
    edge_set = set()
    edges = []
    for k in range(n_sides):
        k_next = (k + 1) % n_sides
        e0 = faces[:, k]
        e1 = faces[:, k_next]
        for a, b in zip(e0, e1):
            key = (min(a, b), max(a, b))
            if key not in edge_set:
                edge_set.add(key)
                edges.append(key)
    return np.array(edges, dtype=np.int64)


def _find_boundary_edges_from_labels(tri_faces, face_labels):
    """Return edges between faces with different semantic labels.

    Parameters
    ----------
    tri_faces : (F, 3) int
    face_labels : (F,) int

    Returns
    -------
    boundary_edges : (E, 2) int – vertex indices of boundary edges
    """
    from collections import defaultdict

    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(tri_faces):
        n_sides = len(face)
        for k in range(n_sides):
            a, b = int(face[k]), int(face[(k + 1) % n_sides])
            key = (min(a, b), max(a, b))
            edge_to_faces[key].append(fi)

    boundary = []
    for edge, fids in edge_to_faces.items():
        if len(fids) == 2:
            if face_labels[fids[0]] != face_labels[fids[1]]:
                boundary.append(edge)
        elif len(fids) == 1:
            boundary.append(edge)

    if len(boundary) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(boundary, dtype=np.int64)


def _sample_points_on_edges(vertices, edges, n_samples_per_edge=5):
    """Uniformly sample points along each edge segment.

    Returns (N, 3) sampled points.
    """
    if len(edges) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    t = np.linspace(0.0, 1.0, n_samples_per_edge + 2)[1:-1]  # exclude endpoints
    v0 = vertices[edges[:, 0]]  # (E, 3)
    v1 = vertices[edges[:, 1]]  # (E, 3)

    pts = []
    for ti in t:
        pts.append(v0 * (1.0 - ti) + v1 * ti)
    pts.append(v0)
    pts.append(v1)
    return np.concatenate(pts, axis=0)


def _point_to_segment_distance_batch(points, seg_a, seg_b):
    """Minimum distance from each point to a set of segments.

    Uses a KDTree on dense segment samples for efficiency when the number
    of segments is large.

    Parameters
    ----------
    points : (P, 3)
    seg_a : (S, 3) – segment start points
    seg_b : (S, 3) – segment end points

    Returns
    -------
    distances : (P,) – distance from each point to nearest segment
    """
    n_seg = seg_a.shape[0]
    n_sub = max(2, min(10, int(np.sqrt(n_seg))))
    t = np.linspace(0.0, 1.0, n_sub).reshape(-1, 1, 1)
    seg_pts = seg_a[np.newaxis] * (1.0 - t) + seg_b[np.newaxis] * t  # (n_sub, S, 3)
    seg_pts = seg_pts.reshape(-1, 3)

    tree = cKDTree(seg_pts)
    dists, _ = tree.query(points, k=1)
    return dists


def compute_boundary_alignment_error(
    orig_vertices,
    orig_faces,
    orig_face_labels,
    quad_vertices,
    quad_faces,
    n_samples_per_edge=5,
    symmetric=False,
):
    """Compute Boundary Alignment Error.

    Parameters
    ----------
    orig_vertices : (V_o, 3)
    orig_faces : (F_o, 3) int – triangle faces
    orig_face_labels : (F_o,) int – per-face semantic labels
    quad_vertices : (V_q, 3)
    quad_faces : (F_q, 4) int – quad faces
    n_samples_per_edge : int – sampling density along boundary edges
    symmetric : bool – if True, also compute reverse direction (Chamfer)

    Returns
    -------
    dict with keys:
        ``bae``          – mean boundary alignment error (forward)
        ``bae_reverse``  – mean reverse error (only if symmetric=True)
        ``bae_chamfer``  – mean of forward and reverse (only if symmetric)
        ``n_boundary_edges`` – number of semantic boundary edges found
        ``n_quad_edges``     – number of quad mesh edges
    """
    orig_vertices = np.asarray(orig_vertices, dtype=np.float64)
    orig_faces = np.asarray(orig_faces, dtype=np.int64)
    orig_face_labels = np.asarray(orig_face_labels, dtype=np.int64)
    quad_vertices = np.asarray(quad_vertices, dtype=np.float64)
    quad_faces = np.asarray(quad_faces, dtype=np.int64)

    boundary_edges = _find_boundary_edges_from_labels(orig_faces, orig_face_labels)
    n_boundary = len(boundary_edges)

    if n_boundary == 0:
        result = {
            "bae": 0.0,
            "n_boundary_edges": 0,
            "n_quad_edges": 0,
        }
        if symmetric:
            result["bae_reverse"] = 0.0
            result["bae_chamfer"] = 0.0
        return result

    boundary_pts = _sample_points_on_edges(
        orig_vertices, boundary_edges, n_samples_per_edge)

    quad_edges = _edges_from_faces(quad_faces)
    n_quad = len(quad_edges)

    qa = quad_vertices[quad_edges[:, 0]]
    qb = quad_vertices[quad_edges[:, 1]]

    forward_dist = _point_to_segment_distance_batch(boundary_pts, qa, qb)
    bae_forward = float(forward_dist.mean())

    result = {
        "bae": bae_forward,
        "n_boundary_edges": n_boundary,
        "n_quad_edges": n_quad,
    }

    if symmetric:
        quad_edge_pts = _sample_points_on_edges(
            quad_vertices, quad_edges, n_samples_per_edge)
        ba = orig_vertices[boundary_edges[:, 0]]
        bb = orig_vertices[boundary_edges[:, 1]]
        reverse_dist = _point_to_segment_distance_batch(quad_edge_pts, ba, bb)
        bae_reverse = float(reverse_dist.mean())
        result["bae_reverse"] = bae_reverse
        result["bae_chamfer"] = (bae_forward + bae_reverse) / 2.0

    return result


def compute_boundary_alignment_error_from_file(
    orig_mesh_path,
    orig_label_path,
    quad_mesh_path,
    **kwargs,
):
    """Convenience wrapper loading files from disk.

    Parameters
    ----------
    orig_mesh_path : str – path to original triangle mesh (.obj)
    orig_label_path : str – path to per-face labels (.npy, integer array)
    quad_mesh_path : str – path to quad mesh (.obj)
    **kwargs – forwarded to ``compute_boundary_alignment_error``
    """
    import trimesh
    from .angle_distortion import _load_quad_faces_from_obj

    orig_mesh = trimesh.load(orig_mesh_path, process=False)
    labels = np.load(orig_label_path)
    quad_mesh = trimesh.load(quad_mesh_path, process=False)
    quad_faces = _load_quad_faces_from_obj(quad_mesh_path)

    if quad_faces is None or len(quad_faces) == 0:
        raise ValueError("No quad faces found in quad mesh.")

    return compute_boundary_alignment_error(
        orig_vertices=orig_mesh.vertices,
        orig_faces=orig_mesh.faces,
        orig_face_labels=labels,
        quad_vertices=quad_mesh.vertices,
        quad_faces=quad_faces,
        **kwargs,
    )
