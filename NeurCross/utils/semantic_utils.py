import numpy as np


def compute_semantic_gradient(points, normals, vertex_neighbors, part_features):
    """
    Compute semantic gradient direction and boundary strength at each face center
    using a structure tensor built from PartField features.

    For every face, the structure tensor T aggregates squared feature differences
    weighted by spatial direction.  Projecting T onto the tangent plane and taking
    the dominant eigenvector gives the direction of maximum feature change; its
    eigenvalue (normalized) serves as boundary strength.

    Args:
        points: (N, 3) face center positions
        normals: (N, 3) face normals (normalized)
        vertex_neighbors: list of length N, vertex_neighbors[i] = list of
                          neighbor face indices
        part_features: (N, D) PartField feature vectors

    Returns:
        grad_dir: (N, 3) normalized semantic gradient direction (tangent plane)
        grad_weight: (N,) boundary strength in [0, 1]
    """
    N = points.shape[0]
    grad_dir = np.zeros((N, 3), dtype=np.float32)
    grad_weight = np.zeros(N, dtype=np.float32)

    for i in range(N):
        neighbors = vertex_neighbors[i]
        if len(neighbors) == 0:
            continue

        n_i = normals[i]
        P = np.eye(3) - np.outer(n_i, n_i)

        T = np.zeros((3, 3))
        for j in neighbors:
            dp = points[j] - points[i]
            dp_norm_sq = np.dot(dp, dp)
            if dp_norm_sq < 1e-12:
                continue
            df_sq = np.sum((part_features[j] - part_features[i]) ** 2)
            T += df_sq * np.outer(dp, dp) / dp_norm_sq

        T_tangent = P @ T @ P

        eigenvalues, eigenvectors = np.linalg.eigh(T_tangent)
        max_idx = np.argmax(eigenvalues)
        grad_dir[i] = eigenvectors[:, max_idx]
        grad_weight[i] = eigenvalues[max_idx]

    max_w = grad_weight.max()
    if max_w > 1e-10:
        grad_weight = grad_weight / max_w

    return grad_dir, grad_weight
