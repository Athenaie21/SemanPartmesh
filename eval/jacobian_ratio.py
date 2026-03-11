"""
Jacobian Ratio (JR) metric for quad meshes.

For each quad face the *scaled Jacobian* is evaluated at each of the four
corners.  At corner *k* the Jacobian is the determinant of the 2x3 matrix
formed by the two edge vectors emanating from that corner, normalized by
the product of their lengths.  The scaled Jacobian per quad is
``min_corner / max_corner`` (both in absolute value).

A perfect planar square yields JR = 1.  Degenerate or inverted elements
yield JR close to or below 0.

References
----------
Knupp, P. M. (2003).  Algebraic mesh quality metrics for unstructured
initial meshes.  *Finite Elements in Analysis and Design*.
"""

import numpy as np


def _scaled_jacobians_at_corners(vertices, quad_faces):
    """Return the scaled Jacobian at each corner of every quad.

    Parameters
    ----------
    vertices : (V, 3) ndarray
    quad_faces : (F, 4) ndarray

    Returns
    -------
    sj : (F, 4) ndarray  – scaled Jacobian per corner
    """
    n = quad_faces.shape[0]
    sj = np.empty((n, 4), dtype=np.float64)

    for k in range(4):
        k_prev = (k - 1) % 4
        k_next = (k + 1) % 4

        v0 = vertices[quad_faces[:, k]]
        v_prev = vertices[quad_faces[:, k_prev]]
        v_next = vertices[quad_faces[:, k_next]]

        e1 = v_next - v0   # (F, 3)
        e2 = v_prev - v0   # (F, 3)

        cross = np.cross(e1, e2)           # (F, 3)
        cross_norm = np.linalg.norm(cross, axis=1)  # |e1 x e2|

        l1 = np.linalg.norm(e1, axis=1).clip(min=1e-12)
        l2 = np.linalg.norm(e2, axis=1).clip(min=1e-12)

        sj[:, k] = cross_norm / (l1 * l2)

    return sj


def compute_jacobian_ratio(vertices, quad_faces):
    """Compute the Jacobian Ratio for every quad face.

    Parameters
    ----------
    vertices : (V, 3) array-like
    quad_faces : (F, 4) array-like

    Returns
    -------
    dict with keys:
        ``mean_jr``      – mean Jacobian Ratio over all faces
        ``min_jr``       – worst (minimum) Jacobian Ratio
        ``std_jr``       – standard deviation
        ``per_face_jr``  – (F,) Jacobian Ratio per face
        ``mean_scaled_jacobian`` – mean of per-face minimum scaled Jacobian
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    quad_faces = np.asarray(quad_faces, dtype=np.int64)

    if quad_faces.shape[1] != 4:
        raise ValueError(f"Expected quad faces (F,4), got shape {quad_faces.shape}")

    sj = _scaled_jacobians_at_corners(vertices, quad_faces)

    sj_min = sj.min(axis=1)
    sj_max = sj.max(axis=1).clip(min=1e-12)

    jr = sj_min / sj_max

    return {
        "mean_jr": float(jr.mean()),
        "min_jr": float(jr.min()),
        "std_jr": float(jr.std()),
        "per_face_jr": jr,
        "mean_scaled_jacobian": float(sj_min.mean()),
    }


def compute_jacobian_ratio_from_file(quad_mesh_path):
    """Convenience wrapper that loads a quad mesh OBJ file."""
    from .angle_distortion import _load_quad_faces_from_obj
    import trimesh

    mesh = trimesh.load(quad_mesh_path, process=False)
    quad_faces = _load_quad_faces_from_obj(quad_mesh_path)

    if quad_faces is None or len(quad_faces) == 0:
        raise ValueError("No quad faces found in the mesh file.")

    return compute_jacobian_ratio(mesh.vertices, quad_faces)
