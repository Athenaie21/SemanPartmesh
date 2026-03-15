#!/usr/bin/env python3
"""Quick script to compare baseline vs ours on cheburashka quad meshes."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from eval.angle_distortion import compute_angle_distortion, _load_quad_faces_from_obj
from eval.jacobian_ratio import compute_jacobian_ratio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_vertices_from_obj(path):
    """Parse vertices directly from OBJ, bypassing trimesh."""
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float64)


pairs = {
    "BASELINE": os.path.join(ROOT, "pipeline_output/eval_compare/baseline_quads/cheburashka_quad.obj"),
    "OURS":     os.path.join(ROOT, "pipeline_output/eval_compare/ours_quads/cheburashka_quad.obj"),
}

results = {}
for label, path in pairs.items():
    if not os.path.isfile(path):
        print(f"  {label}: file not found at {path}")
        continue

    v = load_vertices_from_obj(path)
    qf = _load_quad_faces_from_obj(path)
    if qf is None or len(qf) == 0:
        print(f"  {label}: no valid quad faces found")
        continue

    ad = compute_angle_distortion(v, qf)
    jr = compute_jacobian_ratio(v, qf)

    results[label] = {
        "n_quads": len(qf),
        "n_verts": len(v),
        "AD_mean_deg": ad["mean_deviation_deg"],
        "AD_max_deg": ad["max_deviation_deg"],
        "JR_mean": jr["mean_jr"],
        "JR_min": jr["min_jr"],
        "JR_std": jr["std_jr"],
        "SJ_mean": jr["mean_scaled_jacobian"],
    }

print()
print("=" * 70)
print("  Cheburashka: Baseline vs Ours (Semantic-aware)")
print("=" * 70)

metric_labels = {
    "n_quads":     "Num Quads",
    "n_verts":     "Num Vertices",
    "AD_mean_deg": "Angle Dist. mean (deg)",
    "AD_max_deg":  "Angle Dist. max  (deg)",
    "JR_mean":     "Jacobian Ratio mean",
    "JR_min":      "Jacobian Ratio min",
    "JR_std":      "Jacobian Ratio std",
    "SJ_mean":     "Scaled Jacobian mean",
}

better_lower = {"AD_mean_deg", "AD_max_deg"}
better_higher = {"JR_mean", "JR_min", "SJ_mean"}

header = f"  {'Metric':<28s} {'Baseline':>12s} {'Ours':>12s} {'Better':>10s}"
print(header)
print("  " + "-" * 66)

bl = results.get("BASELINE", {})
ours = results.get("OURS", {})

for key, label in metric_labels.items():
    v_bl = bl.get(key)
    v_ours = ours.get(key)

    s_bl = f"{v_bl:.4f}" if isinstance(v_bl, float) else str(v_bl) if v_bl is not None else "N/A"
    s_ours = f"{v_ours:.4f}" if isinstance(v_ours, float) else str(v_ours) if v_ours is not None else "N/A"

    marker = ""
    if v_bl is not None and v_ours is not None and isinstance(v_bl, float):
        if key in better_lower:
            marker = "<- Ours" if v_ours < v_bl else ("<- BL" if v_bl < v_ours else "  tie")
        elif key in better_higher:
            marker = "<- Ours" if v_ours > v_bl else ("<- BL" if v_bl > v_ours else "  tie")

    print(f"  {label:<28s} {s_bl:>12s} {s_ours:>12s} {marker:>10s}")

print("=" * 70)
