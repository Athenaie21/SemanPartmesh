"""
Angle Distortion metric for quad meshes.

Measures the deviation of interior angles in quad faces from the ideal 90
degrees.  For a perfect planar square every corner is exactly pi/2; the
metric reports how far the actual mesh deviates from that ideal.

Two statistics are returned:
    * **mean** – average absolute deviation (radians) over all corners.
    * **max**  – worst-case absolute deviation.
"""

import os

import numpy as np
import trimesh


def _quad_corner_angles(vertices, quad_faces):
    """Return interior angles (radians) at every corner of every quad.

    Parameters
    ----------
    vertices : (V, 3) ndarray
    quad_faces : (F, 4) ndarray  – vertex indices per quad

    Returns
    -------
    angles : (F, 4) ndarray  – interior angle at each corner
    """
    n_faces = quad_faces.shape[0]
    angles = np.empty((n_faces, 4), dtype=np.float64)

    for k in range(4):
        i_prev = (k - 1) % 4
        i_next = (k + 1) % 4

        v_cur = vertices[quad_faces[:, k]]
        v_prev = vertices[quad_faces[:, i_prev]]
        v_next = vertices[quad_faces[:, i_next]]

        e1 = v_prev - v_cur
        e2 = v_next - v_cur

        e1_len = np.linalg.norm(e1, axis=1, keepdims=True).clip(min=1e-12)
        e2_len = np.linalg.norm(e2, axis=1, keepdims=True).clip(min=1e-12)
        e1 = e1 / e1_len
        e2 = e2 / e2_len

        cos_val = np.sum(e1 * e2, axis=1).clip(-1.0, 1.0)
        angles[:, k] = np.arccos(cos_val)

    return angles


def compute_angle_distortion(vertices, quad_faces):
    """Compute angle distortion of a quad mesh.

    Parameters
    ----------
    vertices : (V, 3) array-like
    quad_faces : (F, 4) array-like  – vertex indices per quad face

    Returns
    -------
    dict with keys:
        ``mean_deviation``  – mean |angle - pi/2| in radians
        ``max_deviation``   – max  |angle - pi/2| in radians
        ``mean_deviation_deg`` – same in degrees
        ``max_deviation_deg``  – same in degrees
        ``per_face_mean_deviation`` – (F,) mean deviation per face
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    quad_faces = np.asarray(quad_faces, dtype=np.int64)

    if quad_faces.shape[1] != 4:
        raise ValueError(f"Expected quad faces (F,4), got shape {quad_faces.shape}")

    angles = _quad_corner_angles(vertices, quad_faces)
    deviation = np.abs(angles - np.pi / 2.0)

    return {
        "mean_deviation": float(deviation.mean()),
        "max_deviation": float(deviation.max()),
        "mean_deviation_deg": float(np.degrees(deviation.mean())),
        "max_deviation_deg": float(np.degrees(deviation.max())),
        "per_face_mean_deviation": deviation.mean(axis=1),
    }


def compute_angle_distortion_from_file(quad_mesh_path):
    """Convenience wrapper that loads a quad mesh file.

    Supports OBJ files with quad faces.  Mixed tri/quad meshes are filtered
    to quads only.
    """
    mesh = trimesh.load(quad_mesh_path, process=False)

    if hasattr(mesh, "faces") and mesh.faces.shape[1] == 3:
        faces_raw = _load_quad_faces_from_obj(quad_mesh_path)
        if faces_raw is None or len(faces_raw) == 0:
            raise ValueError("No quad faces found in the mesh file.")
        return compute_angle_distortion(mesh.vertices, faces_raw)

    return compute_angle_distortion(mesh.vertices, mesh.faces)


def _load_quad_faces_from_obj(obj_path):
    """Parse an OBJ file and return only valid quad faces as (F, 4) array.

    Skips faces that reference out-of-range vertex indices (a known libQEx
    artefact on some meshes).
    """
    n_verts = 0
    quads = []
    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                n_verts += 1
            elif line.startswith("f "):
                parts = line.strip().split()
                indices = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                if len(indices) == 4:
                    quads.append(indices)

    if not quads:
        return None

    arr = np.array(quads, dtype=np.int64)
    valid = np.all((arr >= 0) & (arr < n_verts), axis=1)
    if valid.sum() < len(arr):
        n_bad = len(arr) - int(valid.sum())
        print(f"  WARNING: skipped {n_bad} quad face(s) with out-of-range "
              f"vertex indices in {os.path.basename(obj_path)}")
    arr = arr[valid]
    return arr if len(arr) > 0 else None
