"""
Class-agnostic mIoU (Part-aware mean Intersection over Union).

Evaluates how well the quad mesh faces align with ground-truth semantic
part labels.  Works in a *class-agnostic* fashion: predicted clusters are
matched to GT parts via best-overlap assignment (Hungarian or greedy),
so no class label correspondence is needed.

Workflow
--------
1. Each quad face inherits a GT part label from the nearest original mesh
   face (via barycentric closest-point lookup).
2. Predicted part labels are either provided directly or obtained by
   clustering PartField features on quad faces (K-Means with automatic K
   sweep, keeping the best mIoU).
3. For every GT part, the best-matching predicted cluster is found and
   IoU is computed.  mIoU = mean of these best IoUs.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


def _transfer_labels(src_points, src_labels, tgt_points):
    """Transfer integer labels from *src* faces to *tgt* faces via nearest
    neighbor in Euclidean space.

    Parameters
    ----------
    src_points : (N, 3) – source face centers
    src_labels : (N,) int – source per-face labels
    tgt_points : (M, 3) – target face centers

    Returns
    -------
    tgt_labels : (M,) int
    """
    tree = cKDTree(src_points)
    _, idx = tree.query(tgt_points, k=1)
    return src_labels[idx]


def _compute_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return inter / union if union > 0 else 0.0


def _best_match_miou(gt_labels, pred_labels):
    """Compute class-agnostic mIoU using Hungarian matching.

    For each GT part, find the predicted cluster that yields the highest
    IoU.  Return mean over all GT parts.
    """
    gt_ids = np.unique(gt_labels)
    gt_ids = gt_ids[gt_ids >= 0]  # ignore -1 (unlabeled)
    pred_ids = np.unique(pred_labels)
    pred_ids = pred_ids[pred_ids >= 0]

    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return 0.0, {}

    n_gt = len(gt_ids)
    n_pred = len(pred_ids)

    iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float64)
    for i, gid in enumerate(gt_ids):
        mask_g = gt_labels == gid
        for j, pid in enumerate(pred_ids):
            mask_p = pred_labels == pid
            iou_matrix[i, j] = _compute_iou(mask_g, mask_p)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matched_ious = {}
    total_iou = 0.0
    for r, c in zip(row_ind, col_ind):
        matched_ious[int(gt_ids[r])] = float(iou_matrix[r, c])
        total_iou += iou_matrix[r, c]

    for i, gid in enumerate(gt_ids):
        if int(gid) not in matched_ious:
            matched_ious[int(gid)] = 0.0

    miou = total_iou / n_gt
    return miou, matched_ious


def compute_class_agnostic_miou(
    gt_labels,
    pred_labels=None,
    pred_features=None,
    quad_face_centers=None,
    orig_face_centers=None,
    orig_labels=None,
    k_range=(2, 20),
):
    """Compute class-agnostic mIoU.

    Supports two workflows:

    **A) Direct labels** – provide ``gt_labels`` and ``pred_labels``
    on the same mesh (same length).

    **B) Transfer + cluster** – provide ``orig_face_centers`` /
    ``orig_labels`` for the source triangle mesh, ``quad_face_centers``
    for the target quad mesh, and ``pred_features`` (PartField features
    on quad faces).  GT labels are transferred from source to target;
    predicted labels are obtained by K-Means clustering on features.

    Parameters
    ----------
    gt_labels : (M,) int  – GT labels on evaluation mesh, or None if
                 transferring from original.
    pred_labels : (M,) int, optional – predicted labels on evaluation mesh.
    pred_features : (M, D), optional – features to cluster when
                    pred_labels is None.
    quad_face_centers : (M, 3), optional
    orig_face_centers : (N, 3), optional
    orig_labels : (N,) int, optional
    k_range : tuple (k_min, k_max) – range of cluster counts to try.

    Returns
    -------
    dict with keys:
        ``miou``           – class-agnostic mIoU in [0, 1]
        ``per_part_iou``   – dict mapping GT part id -> IoU
        ``best_k``         – number of clusters used (if clustering)
    """
    if gt_labels is None:
        if orig_face_centers is None or orig_labels is None or quad_face_centers is None:
            raise ValueError(
                "Provide either gt_labels directly, or (orig_face_centers, "
                "orig_labels, quad_face_centers) for label transfer.")
        gt_labels = _transfer_labels(
            np.asarray(orig_face_centers),
            np.asarray(orig_labels),
            np.asarray(quad_face_centers),
        )

    gt_labels = np.asarray(gt_labels, dtype=np.int64)

    if pred_labels is not None:
        pred_labels = np.asarray(pred_labels, dtype=np.int64)
        miou, per_part = _best_match_miou(gt_labels, pred_labels)
        return {"miou": miou, "per_part_iou": per_part, "best_k": None}

    if pred_features is None:
        raise ValueError("Provide pred_labels or pred_features.")

    from sklearn.cluster import KMeans

    pred_features = np.asarray(pred_features, dtype=np.float64)
    best_miou = -1.0
    best_result = None

    k_min, k_max = k_range
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        labels_k = km.fit_predict(pred_features)
        miou_k, per_part_k = _best_match_miou(gt_labels, labels_k)
        if miou_k > best_miou:
            best_miou = miou_k
            best_result = {
                "miou": miou_k,
                "per_part_iou": per_part_k,
                "best_k": k,
            }

    return best_result
