"""
Utilities for generating pseudo-GT semantic labels from PartField features.

PartField produces dense per-face feature vectors.  Clustering these
features yields semantic part labels that serve as the "ground truth"
for evaluating mIoU and Boundary Alignment Error — no human annotation
required.

Two clustering strategies are provided:
    * **K-Means** with automatic K selection via silhouette score.
    * **Fixed K** when the desired number of parts is known.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_features(features, k_range=(2, 15), method="best_silhouette"):
    """Cluster PartField features into semantic part labels.

    Parameters
    ----------
    features : (N, D) ndarray – PartField per-face feature vectors.
    k_range : (int, int) – min and max cluster count to try.
    method : str
        ``"best_silhouette"`` – sweep K in *k_range*, pick the K with the
        highest silhouette score.
        ``"all"`` – return labels for every K (for mIoU sweep).

    Returns
    -------
    If method == "best_silhouette":
        dict with ``labels`` (N,), ``k``, ``silhouette``.
    If method == "all":
        dict mapping K -> (N,) label array.
    """
    features = np.asarray(features, dtype=np.float64)
    feat_norm = np.linalg.norm(features, axis=-1, keepdims=True)
    feat_norm = np.clip(feat_norm, 1e-12, None)
    features = features / feat_norm

    k_min, k_max = k_range

    if method == "all":
        result = {}
        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, n_init=5, random_state=42)
            result[k] = km.fit_predict(features).astype(np.int64)
        return result

    best_k = k_min
    best_score = -1.0
    best_labels = None

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        labels_k = km.fit_predict(features)
        if len(np.unique(labels_k)) < 2:
            continue
        n_sample = min(5000, len(features))
        idx = np.random.RandomState(42).choice(
            len(features), n_sample, replace=False)
        score = silhouette_score(features[idx], labels_k[idx])
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels_k.astype(np.int64)

    return {"labels": best_labels, "k": best_k, "silhouette": best_score}


def generate_labels_from_features(feat_path, k=None, k_range=(2, 15)):
    """Load PartField features and cluster into part labels.

    Parameters
    ----------
    feat_path : str – path to .npy feature file.
    k : int or None – if given, use exactly this many clusters.
    k_range : (int, int) – used when k is None for auto-selection.

    Returns
    -------
    labels : (N,) int64 ndarray – per-face cluster labels.
    k_used : int – number of clusters.
    """
    features = np.load(feat_path)

    if k is not None:
        feat_norm = np.linalg.norm(features, axis=-1, keepdims=True)
        features = features / np.clip(feat_norm, 1e-12, None)
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        labels = km.fit_predict(features).astype(np.int64)
        return labels, k

    result = cluster_features(features, k_range=k_range,
                              method="best_silhouette")
    return result["labels"], result["k"]


def transfer_features_to_quad(orig_face_centers, orig_features,
                              quad_face_centers):
    """Transfer PartField features from original mesh to quad mesh faces
    via nearest-neighbor lookup.

    Returns (M, D) features on quad faces.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(orig_face_centers))
    _, idx = tree.query(np.asarray(quad_face_centers), k=1)
    return orig_features[idx]
