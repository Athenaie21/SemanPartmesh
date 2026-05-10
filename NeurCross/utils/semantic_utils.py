import numpy as np


def reduce_features_pca(features, n_components=32):
    """Reduce feature dimension with a small NumPy PCA fallback."""
    features = np.asarray(features, dtype=np.float32)
    if n_components is None or n_components <= 0 or features.shape[1] <= n_components:
        return features

    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components].T
    return np.matmul(centered, components).astype(np.float32)


def compute_semantic_gradient(points, normals, vertex_neighbors, part_features,
                              method="structure_tensor", normalize=True,
                              pca_dim=None, distance_sigma=None):
    """
    Compute semantic gradient direction and boundary strength at each face center
    from local PartField feature variation.

    ``structure_tensor`` accumulates squared feature differences along neighbor
    directions. ``jacobian`` fits a local least-squares feature Jacobian and uses
    the dominant eigenvector of J J^T. ``gradient_avg`` averages weighted tangent
    neighbor directions.

    Args:
        points: (N, 3) face center positions
        normals: (N, 3) face normals (normalized)
        vertex_neighbors: list of length N, vertex_neighbors[i] = list of
                          neighbor face indices
        part_features: (N, D) PartField feature vectors
        method: structure_tensor | jacobian | gradient_avg
        normalize: whether to L2-normalize features before differencing
        pca_dim: optional PCA dimension reduction before differencing
        distance_sigma: optional Gaussian spatial falloff for neighbor pairs

    Returns:
        grad_dir: (N, 3) normalized semantic gradient direction (tangent plane)
        grad_weight: (N,) boundary strength in [0, 1]
    """
    if normalize:
        feat_norm = np.linalg.norm(part_features, axis=-1, keepdims=True)
        part_features = part_features / np.clip(feat_norm, 1e-12, None)
    if pca_dim is not None:
        part_features = reduce_features_pca(part_features, n_components=pca_dim)

    method = str(method).lower()
    if method not in {"structure_tensor", "jacobian", "gradient_avg"}:
        raise ValueError("Unsupported semantic gradient method: {}".format(method))

    N = points.shape[0]
    grad_dir = np.zeros((N, 3), dtype=np.float32)
    grad_weight = np.zeros(N, dtype=np.float32)

    for i in range(N):
        neighbors = vertex_neighbors[i]
        if len(neighbors) == 0:
            continue

        n_i = normals[i]
        P = np.eye(3) - np.outer(n_i, n_i)

        if method == "jacobian":
            tangent_offsets = []
            feature_offsets = []
            pair_weights = []
            for j in neighbors:
                dp = points[j] - points[i]
                dp = P @ dp
                dp_norm_sq = np.dot(dp, dp)
                if dp_norm_sq < 1e-12:
                    continue
                weight = 1.0
                if distance_sigma is not None and distance_sigma > 0:
                    weight = np.exp(-dp_norm_sq / (2.0 * distance_sigma * distance_sigma))
                tangent_offsets.append(dp)
                feature_offsets.append(part_features[j] - part_features[i])
                pair_weights.append(weight)

            if len(tangent_offsets) == 0:
                continue

            X = np.asarray(tangent_offsets, dtype=np.float64)
            Y = np.asarray(feature_offsets, dtype=np.float64)
            W = np.sqrt(np.asarray(pair_weights, dtype=np.float64))[:, None]
            try:
                J, _, _, _ = np.linalg.lstsq(X * W, Y * W, rcond=None)
            except np.linalg.LinAlgError:
                continue
            T_tangent = P @ (J @ J.T) @ P
            eigenvalues, eigenvectors = np.linalg.eigh(T_tangent)
            max_idx = np.argmax(eigenvalues)
            grad_dir[i] = eigenvectors[:, max_idx]
            grad_weight[i] = eigenvalues[max_idx]
            continue

        if method == "gradient_avg":
            direction = np.zeros(3, dtype=np.float64)
            weight_sum = 0.0
            for j in neighbors:
                dp = points[j] - points[i]
                dp = P @ dp
                dp_norm = np.linalg.norm(dp)
                if dp_norm < 1e-12:
                    continue
                df_norm = np.linalg.norm(part_features[j] - part_features[i])
                weight = df_norm
                if distance_sigma is not None and distance_sigma > 0:
                    weight *= np.exp(-(dp_norm ** 2) / (2.0 * distance_sigma * distance_sigma))
                direction += weight * (dp / dp_norm)
                weight_sum += abs(weight)
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-12:
                grad_dir[i] = direction / dir_norm
                grad_weight[i] = weight_sum
            continue

        T = np.zeros((3, 3))
        for j in neighbors:
            dp = points[j] - points[i]
            dp_norm_sq = np.dot(dp, dp)
            if dp_norm_sq < 1e-12:
                continue
            df_sq = np.sum((part_features[j] - part_features[i]) ** 2)
            weight = 1.0
            if distance_sigma is not None and distance_sigma > 0:
                weight = np.exp(-dp_norm_sq / (2.0 * distance_sigma * distance_sigma))
            T += weight * df_sq * np.outer(dp, dp) / dp_norm_sq

        T_tangent = P @ T @ P

        eigenvalues, eigenvectors = np.linalg.eigh(T_tangent)
        max_idx = np.argmax(eigenvalues)
        grad_dir[i] = eigenvectors[:, max_idx]
        grad_weight[i] = eigenvalues[max_idx]

    max_w = grad_weight.max()
    if max_w > 1e-10:
        grad_weight = grad_weight / max_w

    return grad_dir, grad_weight
