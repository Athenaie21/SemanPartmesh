#!/usr/bin/env python
"""
Pipeline: PartField  -->  NeurCross  -->  Quad Mesh Extraction.

Three stages:
    1. ``partfield``  — Semantic feature extraction (conda env: partfield)
    2. ``neurcross``  — Cross field training        (conda env: neurcross)
    3. ``extract``    — MIQ param + libQEx quad mesh (C++ tool)

Supports both single-mesh and batch (directory) modes.

Usage
-----
    python run_pipeline.py --input_dir history_input/
    python run_pipeline.py --input_dir input/
    python run_pipeline.py --input_mesh input/armadillo.obj
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import hashlib
from datetime import datetime

import numpy as np
import trimesh

from build_complexity_map import (
    compute_semantic_component,
    get_face_neighbors,
    robust_normalize,
)
from eval.label_utils import cluster_features
from extract_quad import load_quad_obj, write_quad_obj
try:
    from instruction_guidance import build_instruction_metadata
    from instruction_guidance import load_instruction_prototypes
    from instruction_guidance import derive_mesh_basename
    from instruction_guidance import infer_dataset_paths
    from instruction_guidance import load_instruction_metadata
    from instruction_guidance import recommend_extraction_policy
    from instruction_guidance import save_instruction_metadata
    INSTRUCTION_GUIDANCE_IMPORT_ERROR = None
except ImportError as exc:
    INSTRUCTION_GUIDANCE_IMPORT_ERROR = exc

    def _raise_missing_instruction_guidance(*args, **kwargs):
        raise RuntimeError(
            "instruction_guidance is required for --guidance_mode instruction"
        ) from INSTRUCTION_GUIDANCE_IMPORT_ERROR

    build_instruction_metadata = _raise_missing_instruction_guidance
    load_instruction_prototypes = _raise_missing_instruction_guidance
    infer_dataset_paths = _raise_missing_instruction_guidance
    load_instruction_metadata = _raise_missing_instruction_guidance
    recommend_extraction_policy = _raise_missing_instruction_guidance
    save_instruction_metadata = _raise_missing_instruction_guidance

    def derive_mesh_basename(path):
        return os.path.splitext(os.path.basename(path))[0]

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARTFIELD_DIR = os.path.join(PROJECT_ROOT, "PartField")
NEURCROSS_DIR = os.path.join(PROJECT_ROOT, "NeurCross")
EXTRACT_QUAD_PY = os.path.join(PROJECT_ROOT, "extract_quad.py")
QUAD_EXTRACT_BIN = os.path.join(
    PROJECT_ROOT, "quad_extract", "build", "extract_quad_mesh")

PARTFIELD_NATIVE_EXTS = {".obj", ".glb", ".off"}
ALL_MESH_EXTS = {".obj", ".glb", ".off", ".ply", ".stl"}

DEFAULT_MAX_FACES = 100001


# ---------------------------------------------------------------------------
#  Conda / environment helpers
# ---------------------------------------------------------------------------

def find_conda_prefix():
    """Return the root of the conda installation."""
    conda_exe = shutil.which("conda")
    if conda_exe:
        return os.path.dirname(os.path.dirname(os.path.realpath(conda_exe)))
    for candidate in [
        os.path.expanduser("~/.conda"),
        os.path.expanduser("~/mambaforge"),
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/anaconda3"),
        "/base/mambaforge",
    ]:
        if os.path.isdir(os.path.join(candidate, "envs")):
            return candidate
    return None


def resolve_env_python(env_name, conda_prefix=None):
    """Return the absolute path to *python* inside a conda environment."""
    search_dirs = []
    if conda_prefix:
        search_dirs.append(os.path.join(conda_prefix, "envs", env_name))
    search_dirs += [
        os.path.join(os.path.expanduser("~/.conda/envs"), env_name),
        os.path.join("/base/mambaforge/envs", env_name),
    ]
    for d in search_dirs:
        py = os.path.join(d, "bin", "python")
        if os.path.isfile(py):
            return py
    sys.exit(f"Cannot locate python for conda env '{env_name}'.\n"
             f"  Searched: {search_dirs}\n"
             f"  Use --python_partfield / --python_neurcross to specify.")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Guided quad-meshing pipeline with feature and instruction modes")

    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input_dir",
                     help="Directory of mesh files (.obj/.ply/.off/.glb). "
                          "All meshes are processed sequentially.")
    inp.add_argument("--input_mesh",
                     help="Path to a single input mesh file")

    # -- Conda environments -------------------------------------------------
    p.add_argument("--env_partfield", default="seman",
                   help="Conda env name for PartField  (default: partfield)")
    p.add_argument("--env_neurcross", default="seman",
                   help="Conda env name for NeurCross  (default: neurcross)")
    p.add_argument("--python_partfield", default=None,
                   help="Explicit python path for PartField env")
    p.add_argument("--python_neurcross", default=None,
                   help="Explicit python path for NeurCross env")

    # -- PartField ----------------------------------------------------------
    p.add_argument("--partfield_ckpt",
                   default=os.path.join(PARTFIELD_DIR, "model",
                                        "model_objaverse.ckpt"),
                   help="PartField model checkpoint")
    p.add_argument("--partfield_config",
                   default=os.path.join(PARTFIELD_DIR, "configs", "final",
                                        "demo.yaml"),
                   help="PartField YAML config")
    p.add_argument("--n_point_per_face", type=int, default=1000,
                   help="Samples per face for PartField feature averaging")

    # -- NeurCross ----------------------------------------------------------
    p.add_argument("--n_samples", type=int, default=10000)
    p.add_argument("--n_points", type=int, default=15000)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num_workers", type=int, default=0,
                   help="NeurCross DataLoader worker count; 0 is slower but avoids worker hangs")
    p.add_argument("--loss_weights", nargs="+", type=float,
                   default=[7e3, 6e2, 10, 5e1, 30, 3, 2000],
                   help="[sdf, inter, theta_hess, eikonal, theta_neigh, "
                        "morse, guidance]")
    p.add_argument("--guidance_mode", choices=["none", "feature", "instruction"],
                   default="feature",
                   help="Guidance mode for cross-field learning and extraction")
    p.add_argument("--semantic_gradient_method", default="jacobian",
                   choices=["structure_tensor", "gradient_avg", "jacobian"],
                   help="Method for computing semantic gradient field")
    p.add_argument("--semantic_boundary_weight", type=float, default=1.2,
                   help="Internal weight for semantic boundary alignment")
    p.add_argument("--semantic_intra_weight", type=float, default=1.0,
                   help="Internal weight for semantic intra-part consistency")
    p.add_argument("--semantic_neighbor_weight", type=float, default=1.5,
                   help="Internal weight for semantic-aware neighbor smoothness")
    p.add_argument("--semantic_cross_part_gamma", type=float, default=0.6,
                   help="Neighbor smoothness attenuation across semantic parts")
    p.add_argument("--semantic_boundary_reward", type=float, default=0.0,
                   help="Optional cross-boundary override/reward; 0 uses semantic_cross_part_gamma")
    p.add_argument("--semantic_pca_dim", type=int, default=0,
                   help="PCA dimensionality for feature reduction (0=disable)")
    p.add_argument("--semantic_normalize_features", type=int, default=1,
                   help="L2 normalize features before gradient computation (1=yes, 0=no)")
    p.add_argument("--semantic_distance_sigma", type=float, default=0.0,
                   help="Gaussian distance sigma for semantic gradient weighting (0=disable)")
    p.add_argument("--semantic_diversity_weight", type=float, default=0.5,
                   help="Internal weight for cross-part diversity loss")
    p.add_argument("--semantic_diversity_margin", type=float, default=0.3,
                   help="Margin threshold for cross-part diversity loss")
    p.add_argument("--semantic_soft_boundary_temp", type=float, default=0.1,
                   help="Temperature for soft boundary weight from feature similarity")
    p.add_argument("--semantic_soft_boundary_chunk_size", type=int, default=1024,
                   help="Pair count per chunk for soft boundary cosine similarity (0=disable chunking)")
    p.add_argument("--alignment_margin", type=float, default=0.0,
                   help="Margin for semantic boundary alignment loss")
    p.add_argument("--semantic_spatial_cluster_weight", type=float, default=0.0,
                   help="Spatial weight for spatially-regularized clustering (0=pure feature clustering)")
    p.add_argument("--intra_temperature", type=float, default=1.0,
                   help="Temperature for intra consistency loss")
    p.add_argument("--intra_hard_ratio", type=float, default=0.0,
                   help="Fraction of worst-aligned intra pairs to focus on (0=all)")
    p.add_argument("--guidance_warmup_fraction", type=float, default=0.15,
                   help="Fraction of training for guidance weight warmup (0=no warmup)")
    p.add_argument("--guidance_cap_ratio", type=float, default=0.5,
                   help="Cap weighted guidance to this ratio of geometric loss (0=disabled)")
    p.add_argument("--guidance_eikonal_guard", type=float, default=0.0,
                   help="Scale down guidance when eikonal exceeds this threshold (0=disabled)")
    p.add_argument("--guidance_warmup_type", choices=["linear", "cosine"], default="linear",
                   help="Warmup schedule type for guidance weight")
    p.add_argument("--instruction_boundary_weight", type=float, default=1.5,
                   help="Internal weight for instruction boundary alignment")
    p.add_argument("--instruction_intra_weight", type=float, default=1.0,
                   help="Internal weight for instruction instance consistency")
    p.add_argument("--instruction_cross_instance_gamma", type=float, default=0.4,
                   help="Cross-instance neighbor smoothness weight (0=no smoothness, 1=full)")
    p.add_argument("--instruction_operation_align_weight", type=float, default=1.0,
                   help="Internal weight for operation direction alignment")
    p.add_argument("--instruction_anchor_weight", type=float, default=0.3,
                   help="Internal weight for instance anchor direction alignment")
    p.add_argument("--instruction_prototype_path", default=None,
                   help="Path to mined instruction prototype JSON")
    p.add_argument("--instruction_timeline_buckets", type=int, default=4,
                   help="Number of buckets used for normalized instruction timeline indices")
    p.add_argument("--instruction_patch_extract", action="store_true",
                   help="Additionally extract local target patches for inspection without merging them")
    p.add_argument("--instruction_patch_context_rings", type=int, default=2,
                   help="Number of face-neighbor rings added around each target instruction patch")
    p.add_argument("--instruction_patch_min_faces", type=int, default=96,
                   help="Minimum face count required for a local instruction patch")
    p.add_argument("--instruction_patch_max_patches", type=int, default=8,
                   help="Maximum number of local instruction patches exported per mesh")

    # -- Quad Extraction (Stage 3) -----------------------------------------
    p.add_argument("--gradient_size", type=float, default=30.0,
                   help="MIQ gradient size (controls quad density)")
    p.add_argument("--extract_timeout", type=int, default=600,
                   help="Timeout in seconds for each Stage 3 extract attempt")
    p.add_argument("--size_field_path", default=None, help=argparse.SUPPRESS)
    p.add_argument("--size_field_strength", type=float, default=0.75,
                   help=argparse.SUPPRESS)
    p.add_argument("--size_field_smooth_iters", type=int, default=6,
                   help=argparse.SUPPRESS)
    p.add_argument("--disable_semantic_size_field", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--semantic_size_k", type=int, default=None,
                   help="Fixed cluster count for semantic size-field generation")
    p.add_argument("--semantic_size_k_min", type=int, default=2,
                   help="Minimum K when auto-selecting semantic clusters for size field")
    p.add_argument("--semantic_size_k_max", type=int, default=15,
                   help="Maximum K when auto-selecting semantic clusters for size field")
    p.add_argument("--semantic_size_grad_mix", type=float, default=0.7,
                   help="Blend between PartField gradient strength and pseudo-label boundaries")
    p.add_argument("--semantic_size_robust_percentile", type=float, default=95.0,
                   help="Robust percentile used to normalize semantic complexity")
    p.add_argument("--semantic_size_min", type=float, default=0.35,
                   help="Minimum relative size in semantically complex regions")
    p.add_argument("--semantic_size_max", type=float, default=1.0,
                   help="Maximum relative size in semantically simple regions")
    p.add_argument("--semantic_size_save_vis", action="store_true",
                   help="Export a colored semantic complexity preview mesh for debugging")
    p.add_argument("--disable_auto_sweep", action="store_true",
                   help="Disable safer Stage 3 auto-sweep extraction")
    p.add_argument("--disable_extract_retry", action="store_true",
                   help="Disable fallback retry extraction with alternative settings")
    p.add_argument("--disable_size_field_relax", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--sweep_values", nargs="+", type=float, default=None,
                   help="Optional gradient_size sweep values for safer extraction")
    p.add_argument("--keep_sweep_outputs", action="store_true",
                   help="Keep all intermediate sweep OBJ files")
    p.add_argument("--keep_extract_intermediates", action="store_true",
                   help="Keep Stage 3 repair/chunk intermediates after successful extraction")
    p.add_argument("--min_quads", type=int, default=None,
                   help="Minimum acceptable quad count during safer extraction")
    p.add_argument("--max_ad", type=float, default=15.0,
                   help="Maximum acceptable mean angle distortion during safer extraction")
    p.add_argument("--min_jr", type=float, default=0.15,
                   help="Minimum acceptable Jacobian ratio during safer extraction")
    p.add_argument("--catmull_clark_iters", type=int, default=0,
                   help="Fixed Catmull-Clark subdivision iterations after extraction")
    p.add_argument("--target_quad_ratio", type=float, default=0.5,
                   help="Target final quad count as a ratio of input triangle faces")
    p.add_argument("--budget_tolerance_ratio", type=float, default=0.2,
                   help="Relative tolerance band around the extraction quad budget")
    p.add_argument("--max_catmull_clark_iters", type=int, default=2,
                   help="Maximum automatic Catmull-Clark subdivision iterations")
    p.add_argument("--enable_chunked_extract",
                   dest="disable_chunked_extract",
                   action="store_false",
                   help="Enable experimental chunked extraction for assembled meshes")
    p.add_argument("--disable_chunked_extract",
                   dest="disable_chunked_extract",
                   action="store_true",
                   help=argparse.SUPPRESS)
    p.set_defaults(disable_chunked_extract=True)
    p.add_argument("--allow_per_chunk_size", action="store_true",
                   help="Allow variable chunk density by merging each chunk's local auto-sweep best")
    p.add_argument("--extract_chunk_min_faces", type=int, default=200,
                   help="Minimum face count required for a semantic extraction chunk")
    p.add_argument("--extract_chunk_max_chunks", type=int, default=24,
                   help="Maximum number of extraction chunks before falling back to global extraction")
    p.add_argument("--skip_extract", action="store_true",
                   help="Skip Stage 3 (quad mesh extraction)")
    p.add_argument("--snap_to_boundary", action="store_true", default=True,
                   help="Post-process: snap quad vertices to nearby semantic boundaries")
    p.add_argument("--no_snap_to_boundary", dest="snap_to_boundary", action="store_false",
                   help="Disable post-process boundary snapping")
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

    # -- General ------------------------------------------------------------
    p.add_argument("--output_dir", default="pipeline_output",
                   help="Root directory for all outputs "
                        "(a timestamp suffix is appended automatically)")
    p.add_argument("--no_timestamp", action="store_true",
                   help="Do not append timestamp to output_dir")
    p.add_argument("--skip_partfield", action="store_true",
                   help="Skip Stage 1; only valid with --input_mesh")
    p.add_argument("--part_feat_path", default=None,
                   help="Pre-computed PartField feature .npy (with --skip_partfield)")
    p.add_argument("--instruction_dataset_root", default=None,
                   help="Instruction dataset root: either legacy meshes/timeline_info or flat reconstruction json/obj")
    p.add_argument("--instruction_meta_path", default=None,
                   help="Precomputed instruction metadata .npz (single-mesh override)")
    p.add_argument("--max_faces", type=int, default=DEFAULT_MAX_FACES,
                   help="Decimate meshes exceeding this face count to avoid "
                        "GPU OOM.  Set 0 to disable.  (default: %(default)s)")
    p.add_argument("--gpu_id", default="0",
                   help="CUDA_VISIBLE_DEVICES value")
    p.add_argument("--reconstruction_whole_mesh", action="store_true", default=False,
                   help="Enable whole-mesh mode for reconstruction (B-rep) datasets: "
                        "increases cross-instance smoothness and stitches quad patches. "
                        "Auto-detected when instruction_dataset_root uses reconstruction layout.")
    p.add_argument("--reconstruction_cross_instance_gamma", type=float, default=0.6,
                   help="Cross-instance neighbor smoothness weight in whole-mesh mode")
    p.add_argument("--reconstruction_global_smooth_weight", type=float, default=0.8,
                   help="Weight for global neighbor smoothness in whole-mesh mode")
    p.add_argument("--reconstruction_boundary_damping", type=float, default=0.3,
                   help="Damping factor for boundary alignment terms in whole-mesh mode")

    args = p.parse_args()

    conda_prefix = find_conda_prefix()
    if args.guidance_mode == "feature" and not args.python_partfield:
        args.python_partfield = resolve_env_python(
            args.env_partfield, conda_prefix)
    if not args.python_neurcross:
        args.python_neurcross = resolve_env_python(
            args.env_neurcross, conda_prefix)

    if args.guidance_mode == "feature":
        if args.skip_partfield and not args.part_feat_path:
            sys.exit("--guidance_mode feature with --skip_partfield requires --part_feat_path")
    elif args.guidance_mode == "instruction":
        if INSTRUCTION_GUIDANCE_IMPORT_ERROR is not None:
            sys.exit(
                "--guidance_mode instruction requires instruction_guidance, "
                f"but it could not be imported: {INSTRUCTION_GUIDANCE_IMPORT_ERROR}"
            )
        if args.skip_partfield and args.part_feat_path:
            print("WARNING: --part_feat_path is ignored in instruction mode")
        if args.input_dir and args.instruction_meta_path:
            sys.exit("--instruction_meta_path only supports single-mesh mode; use --instruction_dataset_root for batches")
        if not args.instruction_dataset_root and not args.instruction_meta_path:
            sys.exit("--guidance_mode instruction requires --instruction_dataset_root or --instruction_meta_path")
        if args.instruction_prototype_path is not None:
            args.instruction_prototype_path = os.path.abspath(args.instruction_prototype_path)
            if not os.path.isfile(args.instruction_prototype_path):
                sys.exit(f"Instruction prototype file not found: {args.instruction_prototype_path}")
    else:
        if args.skip_partfield and args.part_feat_path:
            print("WARNING: --part_feat_path is ignored in guidance_mode none")

    return args


# ---------------------------------------------------------------------------
#  Input collection
# ---------------------------------------------------------------------------

def collect_meshes(args):
    """Return a sorted list of absolute mesh file paths."""
    if args.input_mesh:
        p = os.path.abspath(args.input_mesh)
        if not os.path.isfile(p):
            sys.exit(f"Input file not found: {p}")
        return [p]

    d = os.path.abspath(args.input_dir)
    if not os.path.isdir(d):
        sys.exit(f"Input directory not found: {d}")

    meshes = []
    for f in sorted(os.listdir(d)):
        if os.path.splitext(f)[1].lower() in ALL_MESH_EXTS:
            meshes.append(os.path.join(d, f))
    if not meshes:
        sys.exit(f"No mesh files found in {d}")
    return meshes


# ---------------------------------------------------------------------------
#  Stage 1: PartField feature extraction
# ---------------------------------------------------------------------------

def decimate_mesh(mesh, target_faces):
    """Simplify a mesh to approximately *target_faces* using quadric decimation.

    Returns (simplified_mesh, True) or (original_mesh, False).
    """
    n = len(mesh.faces)
    if n <= target_faces:
        return mesh, False

    import fast_simplification
    target_reduction = 1.0 - target_faces / n
    v_out, f_out = fast_simplification.simplify(
        mesh.vertices, mesh.faces, target_reduction=target_reduction)
    simplified = trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)
    return simplified, True


def weld_duplicate_vertices(mesh, basename="mesh"):
    """Merge duplicate vertices in a mesh so disconnected B-Rep patches
    become a single connected surface.

    Many CAD-exported meshes triangulate each B-Rep face independently,
    duplicating vertices along shared edges.  NeurCross and the quad
    extractor both need a connected mesh for correct results.

    Returns (mesh, did_weld).  The face count and order are preserved.
    """
    n_before = len(mesh.vertices)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    n_after = len(mesh.vertices)
    if n_after < n_before:
        print(f"  {basename}  welded duplicate vertices: {n_before} -> {n_after}"
              f" ({n_before - n_after} removed)")
        return mesh, True
    return mesh, False


def repair_polygon_soup(mesh, basename="mesh", tol_fraction=0.002,
                        min_component_faces=20):
    """Repair polygon-soup meshes produced by neural 3D generators.

    These meshes are visually continuous but topologically fragmented into
    hundreds or thousands of tiny disconnected components.  This function:
      1. Merges vertices within a spatial tolerance (fraction of bbox diagonal).
      2. Removes degenerate faces created by the merge.
      3. Drops tiny connected components (< *min_component_faces*).

    Returns (mesh, did_repair).  If the input already has few components,
    no changes are made.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from collections import Counter
    import numpy as np

    V, F = mesh.vertices, mesh.faces
    n_verts_orig = len(V)
    n_faces_orig = len(F)

    n = len(V)
    row = np.concatenate([F[:, 0], F[:, 1], F[:, 2],
                          F[:, 0], F[:, 1], F[:, 2]])
    col = np.concatenate([F[:, 1], F[:, 2], F[:, 0],
                          F[:, 2], F[:, 0], F[:, 1]])
    adj = csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    n_components = connected_components(adj, return_labels=False)

    if n_components <= 50:
        return mesh, False

    bbox_diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    if bbox_diag < 1e-12:
        return mesh, False
    tol = min(bbox_diag * tol_fraction, 0.01)

    V = V.copy()
    F = F.copy()

    tree = cKDTree(V)
    pairs = tree.query_pairs(r=tol)

    parent = list(range(len(V)))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    canonical = {}
    new_idx = 0
    new_verts = []
    vert_map = np.empty(len(V), dtype=np.int64)
    for i in range(len(V)):
        root = _find(i)
        if root not in canonical:
            canonical[root] = new_idx
            new_verts.append(V[root])
            new_idx += 1
        vert_map[i] = canonical[root]

    new_V = np.asarray(new_verts)
    new_F = vert_map[F]
    valid = ((new_F[:, 0] != new_F[:, 1]) &
             (new_F[:, 1] != new_F[:, 2]) &
             (new_F[:, 0] != new_F[:, 2]))
    new_F = new_F[valid]

    n2 = len(new_V)
    row2 = np.concatenate([new_F[:, 0], new_F[:, 1], new_F[:, 2],
                           new_F[:, 0], new_F[:, 1], new_F[:, 2]])
    col2 = np.concatenate([new_F[:, 1], new_F[:, 2], new_F[:, 0],
                           new_F[:, 2], new_F[:, 0], new_F[:, 1]])
    adj2 = csr_matrix((np.ones(len(row2)), (row2, col2)), shape=(n2, n2))
    nc2, labels2 = connected_components(adj2)

    face_labels = labels2[new_F[:, 0]]
    comp_sizes = Counter(face_labels)
    keep = {c for c, sz in comp_sizes.items() if sz >= min_component_faces}
    keep_mask = np.array([face_labels[i] in keep for i in range(len(new_F))])
    kept_F = new_F[keep_mask]

    used = np.unique(kept_F)
    remap = np.full(len(new_V), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    final_V = new_V[used]
    final_F = remap[kept_F]

    n3 = len(final_V)
    row3 = np.concatenate([final_F[:, 0], final_F[:, 1], final_F[:, 2],
                           final_F[:, 0], final_F[:, 1], final_F[:, 2]])
    col3 = np.concatenate([final_F[:, 1], final_F[:, 2], final_F[:, 0],
                           final_F[:, 2], final_F[:, 0], final_F[:, 1]])
    adj3 = csr_matrix((np.ones(len(row3)), (row3, col3)), shape=(n3, n3))
    nc_final = connected_components(adj3, return_labels=False)

    print(f"  {basename}  repaired polygon soup: "
          f"{n_verts_orig}v/{n_faces_orig}f/{n_components}comp -> "
          f"{len(final_V)}v/{len(final_F)}f/{nc_final}comp  "
          f"(tol={tol:.6f})")

    result = trimesh.Trimesh(vertices=final_V, faces=final_F, process=False)
    return result, True


def prepare_meshes(mesh_paths, output_dir, max_faces):
    """Load meshes, optionally decimate, and prepare for both stages.

    For PartField: write OBJ files to a staging directory.
    For NeurCross: write decimated meshes to a processed directory
                   (or use the original if no decimation was needed).

    Returns (staging_dir, info_list) where each info entry is
    (basename, neurcross_mesh_path, n_faces).
    """
    staging_dir = os.path.join(output_dir, "partfield_input")
    processed_dir = os.path.join(output_dir, "processed_meshes")
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    info_list = []
    skipped = []
    for mp in mesh_paths:
        basename = os.path.splitext(os.path.basename(mp))[0]
        ext = os.path.splitext(mp)[1].lower()
        try:
            loaded = trimesh.load(mp, process=False)
            if isinstance(loaded, trimesh.Scene):
                mesh = loaded.to_mesh()
                if mesh is None or len(mesh.faces) == 0:
                    mesh = trimesh.util.concatenate(
                        [g for g in loaded.geometry.values()
                         if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
                    )
            else:
                mesh = loaded
        except Exception as exc:
            skipped.append((mp, str(exc)))
            print(f"  WARNING: skip unreadable mesh {basename}{ext}: {exc}")
            continue
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            skipped.append((mp, "failed to extract triangle mesh from file"))
            print(f"  WARNING: skip {basename}{ext}: could not extract triangle mesh")
            continue
        n_orig = len(mesh.faces)

        decimated = False
        if max_faces > 0 and n_orig > max_faces:
            mesh, decimated = decimate_mesh(mesh, max_faces)

        mesh, welded = weld_duplicate_vertices(mesh, basename)
        mesh, repaired = repair_polygon_soup(mesh, basename)
        n_faces = len(mesh.faces)

        modified = decimated or welded or repaired
        pf_dst = os.path.join(staging_dir, f"{basename}.obj")
        if modified:
            if not os.path.exists(pf_dst):
                mesh.export(pf_dst)
            nc_path = os.path.join(processed_dir, f"{basename}.obj")
            if not os.path.exists(nc_path):
                mesh.export(nc_path)
            if decimated:
                print(f"  {basename}{ext}  decimated {n_orig} -> {n_faces} faces")
            elif welded and not repaired:
                print(f"  {basename}{ext}  welded  (faces={n_faces})")
        else:
            if ext in PARTFIELD_NATIVE_EXTS:
                if not os.path.exists(pf_dst):
                    os.symlink(os.path.abspath(mp), pf_dst)
            else:
                if not os.path.exists(pf_dst):
                    mesh.export(pf_dst)
            nc_path = mp
            print(f"  {basename}{ext}  kept as-is  (faces={n_faces})")

        info_list.append((basename, nc_path, n_faces))

    if skipped:
        print("\n  Skipped meshes:")
        for path, reason in skipped:
            print(f"    - {os.path.basename(path)}: {reason}")
        print()

    return staging_dir, info_list


def run_partfield(staging_dir, args):
    """Run PartField inference on all meshes in *staging_dir*."""
    result_tag = "partfield_features"
    # Keep PartField outputs inside its exp_results tree. Some Lightning/DDP
    # launches in this copy do not preserve cross-tree relative result_name
    # paths such as ../../../shared-nvme/..., which leaves downstream stages
    # pointing at feature files that were never written.
    output_feature_dir = os.path.join(os.path.abspath(args.output_dir), result_tag)
    output_key = hashlib.sha1(os.path.abspath(args.output_dir).encode("utf-8")).hexdigest()[:10]
    result_name = f"{os.path.basename(os.path.abspath(args.output_dir))}_{output_key}_{result_tag}"
    feat_dir = os.path.join(PARTFIELD_DIR, "exp_results", result_name)
    if os.path.lexists(feat_dir):
        if os.path.isdir(feat_dir) and not os.path.islink(feat_dir):
            shutil.rmtree(feat_dir)
        else:
            os.unlink(feat_dir)

    cmd = [
        args.python_partfield, "partfield_inference.py",
        "-c", os.path.abspath(args.partfield_config),
        "--opts",
        "continue_ckpt",   os.path.abspath(args.partfield_ckpt),
        "result_name",      result_name,
        "dataset.data_path", os.path.abspath(staging_dir),
        "is_pc",            "False",
        "preprocess_mesh",  "False",
        "vertex_feature",   "False",
        "n_point_per_face", str(args.n_point_per_face),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print(f"\n[Stage 1] PartField inference")
    print(f"  python : {args.python_partfield}")
    print(f"  cwd    : {PARTFIELD_DIR}")
    print(f"  cmd    : {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=PARTFIELD_DIR, env=env, check=True)

    has_features = (
        os.path.isdir(feat_dir)
        and any(name.endswith(".npy") for name in os.listdir(feat_dir))
    )
    if not has_features:
        raise FileNotFoundError(f"PartField did not produce feature files in {feat_dir}")
    if os.path.lexists(output_feature_dir):
        if os.path.isdir(output_feature_dir) and not os.path.islink(output_feature_dir):
            shutil.rmtree(output_feature_dir)
        else:
            os.unlink(output_feature_dir)
    os.symlink(feat_dir, output_feature_dir)
    return feat_dir


def find_feature_file(feat_dir, basename):
    """Locate the .npy feature file that PartField produced for *basename*."""
    candidates = [
        os.path.join(feat_dir, f"part_feat_{basename}_0.npy"),
        os.path.join(feat_dir, f"part_feat_{basename}_0_batch.npy"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Feature file for '{basename}' not found in {feat_dir}.\n"
        f"  Looked for: {[os.path.basename(c) for c in candidates]}")


# ---------------------------------------------------------------------------
#  Stage 2: NeurCross training
# ---------------------------------------------------------------------------

def run_neurcross(input_mesh, feat_path, instruction_meta_path, mesh_name, args):
    """Run NeurCross training for a single mesh.

    NeurCross's train_quad_mesh.py internally appends the mesh filename
    to --logdir, so we only pass the parent directory here.
    Final output lands in: <output_dir>/neurcross_logs/<mesh_name>/
    """
    logdir_parent = os.path.join(
        os.path.abspath(args.output_dir), "neurcross_logs")

    cmd = [
        args.python_neurcross, "train_quad_mesh.py",
        "--data_path",      os.path.abspath(input_mesh),
        "--logdir",         logdir_parent,
        "--guidance_mode",  args.guidance_mode,
        "--n_samples",      str(args.n_samples),
        "--n_points",       str(args.n_points),
        "--num_epochs",     str(args.num_epochs),
        "--lr",             str(args.lr),
        "--num_workers",    str(args.num_workers),
        "--loss_weights",   *[str(w) for w in args.loss_weights],
        "--morse_near",
    ]

    if args.guidance_mode == "feature":
        cmd.extend([
            "--part_feat_path", os.path.abspath(feat_path),
            "--semantic_gradient_method", str(args.semantic_gradient_method),
            "--semantic_boundary_weight", str(args.semantic_boundary_weight),
            "--semantic_intra_weight", str(args.semantic_intra_weight),
            "--semantic_neighbor_weight", str(args.semantic_neighbor_weight),
            "--semantic_cross_part_gamma", str(args.semantic_cross_part_gamma),
            "--semantic_boundary_reward", str(args.semantic_boundary_reward),
            "--semantic_pca_dim", str(args.semantic_pca_dim),
            "--semantic_normalize_features", str(args.semantic_normalize_features),
            "--semantic_distance_sigma", str(args.semantic_distance_sigma),
            "--semantic_diversity_weight", str(args.semantic_diversity_weight),
            "--semantic_diversity_margin", str(args.semantic_diversity_margin),
            "--semantic_soft_boundary_temp", str(args.semantic_soft_boundary_temp),
            "--semantic_soft_boundary_chunk_size", str(args.semantic_soft_boundary_chunk_size),
            "--alignment_margin", str(args.alignment_margin),
            "--semantic_spatial_cluster_weight", str(args.semantic_spatial_cluster_weight),
            "--intra_temperature", str(args.intra_temperature),
            "--intra_hard_ratio", str(args.intra_hard_ratio),
            "--guidance_warmup_fraction", str(args.guidance_warmup_fraction),
            "--guidance_cap_ratio", str(args.guidance_cap_ratio),
            "--guidance_eikonal_guard", str(args.guidance_eikonal_guard),
            "--guidance_warmup_type", str(args.guidance_warmup_type),
        ])
    elif args.guidance_mode == "instruction":
        cmd.extend([
            "--instruction_meta_path", os.path.abspath(instruction_meta_path),
            "--instruction_boundary_weight", str(args.instruction_boundary_weight),
            "--instruction_intra_weight", str(args.instruction_intra_weight),
            "--instruction_cross_instance_gamma", str(args.instruction_cross_instance_gamma),
            "--instruction_operation_align_weight", str(args.instruction_operation_align_weight),
            "--instruction_anchor_weight", str(args.instruction_anchor_weight),
        ])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    quad_mesh_dir = os.path.join(NEURCROSS_DIR, "quad_mesh")
    print(f"\n[Stage 2] NeurCross training  [{mesh_name}]")
    print(f"  python : {args.python_neurcross}")
    print(f"  cwd    : {quad_mesh_dir}")
    print(f"  cmd    : {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=quad_mesh_dir, env=env, check=True)

    actual_logdir = os.path.join(logdir_parent, mesh_name)
    print(f"[Stage 2] Training complete  ->  {actual_logdir}")
    return actual_logdir


# ---------------------------------------------------------------------------
#  Stage 3: Quad mesh extraction  (MIQ + libQEx)
# ---------------------------------------------------------------------------

def find_latest_crossfield(crossfield_dir):
    """Find the cross field txt with the highest iteration number."""
    import glob
    files = glob.glob(os.path.join(crossfield_dir, "*_iter_*.txt"))
    if not files:
        return None
    def iter_num(f):
        base = os.path.splitext(os.path.basename(f))[0]
        return int(base.rsplit("_iter_", 1)[1])
    return max(files, key=iter_num)


def get_face_connected_components(mesh):
    face_count = len(mesh.faces)
    if face_count == 0:
        return []

    neighbors = get_face_neighbors(mesh)
    visited = np.zeros(face_count, dtype=bool)
    components = []

    for seed in range(face_count):
        if visited[seed]:
            continue
        stack = [seed]
        visited[seed] = True
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in neighbors[current]:
                neighbor = int(neighbor)
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.int64))

    components.sort(key=len, reverse=True)
    return components


def split_face_mask_into_connected_chunks(neighbors, mask):
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(len(mask), dtype=bool)
    chunks = []

    valid_ids = np.flatnonzero(mask)
    for seed in valid_ids:
        if visited[seed]:
            continue
        stack = [int(seed)]
        visited[seed] = True
        chunk = []
        while stack:
            current = stack.pop()
            chunk.append(current)
            for neighbor in neighbors[current]:
                neighbor = int(neighbor)
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        chunks.append(np.asarray(sorted(chunk), dtype=np.int64))

    chunks.sort(key=len, reverse=True)
    return chunks


def expand_face_mask_by_rings(neighbors, mask, num_rings):
    mask = np.asarray(mask, dtype=bool)
    if num_rings <= 0:
        return mask.copy()

    expanded = mask.copy()
    frontier = np.flatnonzero(mask)
    for _ in range(int(num_rings)):
        next_frontier = []
        for face_idx in frontier:
            for neighbor in neighbors[int(face_idx)]:
                neighbor = int(neighbor)
                if not expanded[neighbor]:
                    expanded[neighbor] = True
                    next_frontier.append(neighbor)
        if not next_frontier:
            break
        frontier = np.asarray(next_frontier, dtype=np.int64)
    return expanded


def merge_overlapping_face_sets(face_sets):
    merged = [
        np.unique(np.asarray(face_ids, dtype=np.int64))
        for face_ids in face_sets
        if len(face_ids) > 0
    ]
    if not merged:
        return []

    changed = True
    while changed:
        changed = False
        next_sets = []
        while merged:
            current = set(merged.pop().tolist())
            remaining = []
            for other in merged:
                if np.intersect1d(
                        np.fromiter(current, dtype=np.int64),
                        other,
                        assume_unique=False).size > 0:
                    current.update(other.tolist())
                    changed = True
                else:
                    remaining.append(other)
            merged = remaining
            next_sets.append(np.asarray(sorted(current), dtype=np.int64))
        merged = next_sets

    merged.sort(key=len, reverse=True)
    return merged


def compute_semantic_face_chunks(mesh, feat_path, args):
    features = np.load(feat_path)
    face_count = len(mesh.faces)
    if len(features) != face_count:
        raise ValueError(
            f"Feature count ({len(features)}) does not match face count ({face_count}).")

    if args.semantic_size_k is not None:
        from sklearn.cluster import KMeans
        feat_norm = np.linalg.norm(features, axis=-1, keepdims=True)
        features_norm = features / np.clip(feat_norm, 1e-12, None)
        labels = KMeans(
            n_clusters=args.semantic_size_k,
            n_init=5,
            random_state=42
        ).fit_predict(features_norm).astype(np.int64)
    else:
        result = cluster_features(
            features,
            k_range=(args.semantic_size_k_min, args.semantic_size_k_max),
            method="best_silhouette"
        )
        labels = result["labels"]

    if labels is None:
        return None

    neighbors = get_face_neighbors(mesh)
    chunks = []
    for label in np.unique(labels):
        label_mask = labels == label
        chunks.extend(split_face_mask_into_connected_chunks(neighbors, label_mask))

    if len(chunks) <= 1:
        return None
    if len(chunks) > args.extract_chunk_max_chunks:
        return None
    if any(len(chunk) < args.extract_chunk_min_faces for chunk in chunks):
        return None

    chunks.sort(key=len, reverse=True)
    return chunks


def compute_instruction_target_patches(mesh, instruction_meta_path, args):
    metadata = load_instruction_metadata(instruction_meta_path)
    instance_ids = np.asarray(metadata["feature_instance_id"], dtype=np.int64)
    target_region_mask = np.asarray(
        metadata.get("target_region_mask", np.ones(len(instance_ids), dtype=np.uint8)),
        dtype=np.uint8,
    )
    if len(instance_ids) != len(mesh.faces):
        raise ValueError(
            f"Instruction metadata face count ({len(instance_ids)}) does not match mesh faces ({len(mesh.faces)}).")
    if len(target_region_mask) != len(mesh.faces):
        raise ValueError(
            f"Instruction target_region_mask count ({len(target_region_mask)}) does not match mesh faces ({len(mesh.faces)}).")

    target_mask = target_region_mask.astype(bool)
    if not np.any(target_mask):
        return None

    neighbors = get_face_neighbors(mesh)
    raw_patches = []
    for instance_id in np.unique(instance_ids[target_mask]):
        seed_mask = (instance_ids == instance_id) & target_mask
        expanded_mask = expand_face_mask_by_rings(
            neighbors, seed_mask, args.instruction_patch_context_rings)
        face_ids = np.flatnonzero(expanded_mask).astype(np.int64)
        if len(face_ids) == 0:
            continue
        raw_patches.append(face_ids)

    merged_patches = merge_overlapping_face_sets(raw_patches)
    patch_records = []
    for face_ids in merged_patches:
        if len(face_ids) < args.instruction_patch_min_faces:
            continue
        patch_records.append({
            "face_ids": face_ids,
            "n_faces": int(len(face_ids)),
            "n_target_faces": int(target_mask[face_ids].sum()),
        })

    if not patch_records:
        return None

    patch_records.sort(key=lambda record: record["n_faces"], reverse=True)
    return patch_records[:args.instruction_patch_max_patches]


def compute_instruction_face_chunks(mesh, instruction_meta_path, args):
    metadata = load_instruction_metadata(instruction_meta_path)
    instance_ids = np.asarray(metadata["feature_instance_id"], dtype=np.int64)
    feature_types = np.asarray(metadata["feature_type_id"], dtype=np.int64)
    target_region_mask = np.asarray(
        metadata.get("target_region_mask", np.ones(len(instance_ids), dtype=np.uint8)),
        dtype=np.uint8,
    )
    if len(instance_ids) != len(mesh.faces):
        raise ValueError(
            f"Instruction metadata face count ({len(instance_ids)}) does not match mesh faces ({len(mesh.faces)}).")
    if len(target_region_mask) != len(mesh.faces):
        raise ValueError(
            f"Instruction target_region_mask count ({len(target_region_mask)}) does not match mesh faces ({len(mesh.faces)}).")

    chunks = []
    target_mask = target_region_mask.astype(bool)
    residual_mask = ~target_mask
    small_target_face_ids = []

    for instance_id in np.unique(instance_ids[target_mask]):
        instance_face_ids = np.flatnonzero((instance_ids == instance_id) & target_mask)
        chunk_type = int(np.bincount(feature_types[instance_face_ids]).argmax()) if len(instance_face_ids) > 0 else 0
        min_faces = args.extract_chunk_min_faces
        if chunk_type in (2, 3):
            min_faces = max(16, min_faces // 4)
        if len(instance_face_ids) < min_faces:
            small_target_face_ids.append(instance_face_ids)
            continue
        chunks.append(instance_face_ids.astype(np.int64))

    if small_target_face_ids:
        merged_target = np.unique(np.concatenate(small_target_face_ids)).astype(np.int64)
        if len(merged_target) < args.extract_chunk_min_faces:
            return None
        chunks.append(merged_target)

    if np.any(residual_mask):
        residual_chunk = np.flatnonzero(residual_mask).astype(np.int64)
        if len(residual_chunk) < args.extract_chunk_min_faces:
            return None
        chunks.append(residual_chunk)

    if len(chunks) <= 1:
        return None
    if len(chunks) > args.extract_chunk_max_chunks:
        return None

    covered_face_ids = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
    covered_face_ids = np.unique(covered_face_ids)
    if len(covered_face_ids) != len(mesh.faces):
        return None

    chunks.sort(key=len, reverse=True)
    return chunks


def plan_extraction_chunks(mesh, feat_path, instruction_meta_path, args):
    disable_chunked = getattr(args, "disable_chunked_extract", False)
    if disable_chunked:
        return [np.arange(len(mesh.faces), dtype=np.int64)], "global"

    # In instruction mode, ALWAYS extract from the complete mesh.
    # Disconnected components are handled by bridge_connected_components()
    # inside extract_quad.py, which stitches them before MIQ + libQEx.
    # Chunked extraction produces fragmented quad islands with broken seams
    # that cannot be reliably merged.
    if args.guidance_mode == "instruction":
        n_components = len(get_face_connected_components(mesh))
        if n_components > 1:
            print(f"  instruction mode: {n_components} connected components "
                  f"will be bridged during extraction (not chunked)")
        return [np.arange(len(mesh.faces), dtype=np.int64)], "global"

    components = get_face_connected_components(mesh)
    if len(components) > 1:
        return components, "connected_components"

    if args.guidance_mode == "feature" and feat_path is not None:
        semantic_chunks = compute_semantic_face_chunks(mesh, feat_path, args)
        if semantic_chunks is not None:
            return semantic_chunks, "semantic_parts"

    return [np.arange(len(mesh.faces), dtype=np.int64)], "global"


def build_face_subset_mesh(mesh, face_ids):
    face_ids = np.asarray(face_ids, dtype=np.int64)
    subset_faces = np.asarray(mesh.faces, dtype=np.int64)[face_ids]
    used_vertices, inverse = np.unique(subset_faces.reshape(-1), return_inverse=True)
    subset_vertices = np.asarray(mesh.vertices)[used_vertices]
    subset_faces = inverse.reshape(-1, subset_faces.shape[1])
    return trimesh.Trimesh(vertices=subset_vertices, faces=subset_faces, process=False)


def write_face_subset_obj(mesh, face_ids, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    subset_mesh = build_face_subset_mesh(mesh, face_ids)
    subset_mesh.export(output_path)
    return output_path


def load_crossfield_rows(crossfield_txt):
    rows = np.loadtxt(crossfield_txt, dtype=np.float64)
    rows = np.atleast_2d(rows)
    if rows.shape[1] != 6:
        raise ValueError(f"Expected cross-field with 6 columns, got shape {rows.shape}")
    return rows


def write_subset_crossfield(rows, face_ids, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savetxt(output_path, rows[np.asarray(face_ids, dtype=np.int64)], fmt="%.8f")
    return output_path


def merge_quad_chunks(chunk_obj_paths, output_path):
    merged_vertices = []
    merged_quads = []
    vertex_offset = 0
    skipped = []

    for chunk_path in chunk_obj_paths:
        if not os.path.isfile(chunk_path) or os.path.getsize(chunk_path) == 0:
            skipped.append(chunk_path)
            continue
        try:
            vertices, quads = load_quad_obj(chunk_path)
        except (ValueError, Exception):
            skipped.append(chunk_path)
            continue
        merged_vertices.append(vertices)
        merged_quads.append(quads + vertex_offset)
        vertex_offset += len(vertices)

    if skipped:
        print(f"  merge: skipped {len(skipped)}/{len(chunk_obj_paths)} chunks")
        for s in skipped:
            print(f"    - {os.path.basename(s)}")

    if not merged_vertices:
        raise ValueError("merge: no valid chunks to merge")

    vertices = np.vstack(merged_vertices)
    quads = np.vstack(merged_quads)
    write_quad_obj(output_path, vertices, quads)
    print(f"  merge: {len(chunk_obj_paths) - len(skipped)}/{len(chunk_obj_paths)} chunks -> "
          f"{len(vertices)} verts, {len(quads)} quads")
    return output_path


def count_quad_topology_issues(quad_path):
    """Count open and non-manifold edges in a quad OBJ."""
    _, quads = load_quad_obj(quad_path)
    edge_count = {}
    for quad in quads:
        for idx in range(4):
            edge = tuple(sorted((int(quad[idx]), int(quad[(idx + 1) % 4]))))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    boundary_edges = sum(1 for count in edge_count.values() if count == 1)
    nonmanifold_edges = sum(1 for count in edge_count.values() if count > 2)
    return int(boundary_edges), int(nonmanifold_edges)


def mesh_surface_area(mesh_path):
    mesh = trimesh.load_mesh(mesh_path, process=False)
    return float(mesh.area)


def select_uniform_density_chunk_candidate(chunk_summaries, input_mesh, output_obj,
                                           candidate_root, args):
    """Merge one candidate per chunk to match a global quad density target."""
    from extract_quad import (
        candidate_budget_terms,
        candidate_violation,
        count_triangle_faces,
        evaluate_quad_mesh,
        get_min_quads_threshold,
        maybe_apply_catmull_clark,
        rank_candidate,
    )

    target_quads = None
    if args.target_quad_ratio > 0:
        target_quads = int(round(count_triangle_faces(input_mesh) * args.target_quad_ratio))

    input_area = max(mesh_surface_area(input_mesh), 1e-12)
    if target_quads is None or target_quads <= 0:
        density_target = None
    else:
        density_target = target_quads / input_area

    selected_paths = []
    selected_chunks = []
    for chunk_idx, summary_path in enumerate(chunk_summaries):
        with open(summary_path, "r") as f:
            summary = json.load(f)

        chunk_mesh = summary["mesh_path"]
        chunk_area = max(mesh_surface_area(chunk_mesh), 1e-12)
        chunk_target_quads = None
        if density_target is not None:
            chunk_target_quads = max(1, int(round(chunk_area * density_target)))

        candidates = []
        for candidate in summary.get("candidates", []):
            path = candidate.get("output_path")
            if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
                continue
            violation = candidate.get("constraint_violation", {})
            failed = int(violation.get("num_failed_constraints", 0))
            n_quads = int(candidate["n_quads"])
            target_abs_diff = 0 if chunk_target_quads is None else abs(n_quads - chunk_target_quads)
            target_overshoot = 0 if chunk_target_quads is None else max(0, n_quads - chunk_target_quads)
            candidates.append((
                (
                    failed,
                    float(violation.get("quad_deficit", 0)),
                    float(violation.get("ad_excess", 0.0)),
                    float(violation.get("jr_deficit", 0.0)),
                    target_abs_diff,
                    target_overshoot,
                    float(candidate.get("AD_mean_deg", 0.0)),
                    -float(candidate.get("JR_mean", 0.0)),
                    n_quads,
                ),
                candidate,
                path,
            ))

        if not candidates:
            raise RuntimeError(f"no usable chunk candidates in {summary_path}")

        candidates.sort(key=lambda item: item[0])
        _, selected, selected_path = candidates[0]
        selected_paths.append(selected_path)
        selected_chunks.append({
            "chunk_index": int(chunk_idx),
            "summary_path": os.path.abspath(summary_path),
            "mesh_path": os.path.abspath(chunk_mesh),
            "area": float(chunk_area),
            "target_quads": chunk_target_quads,
            "selected_gradient_size": float(selected["gradient_size"]),
            "selected_n_quads": int(selected["n_quads"]),
            "selected_output_path": os.path.abspath(selected_path),
        })

    os.makedirs(candidate_root, exist_ok=True)
    merged_path = os.path.join(candidate_root, "merged_uniform_density.obj")
    merge_quad_chunks(selected_paths, merged_path)

    subdiv_info = maybe_apply_catmull_clark(merged_path, input_mesh, args)
    metrics = evaluate_quad_mesh(merged_path)
    if subdiv_info is not None:
        metrics.update(subdiv_info)

    min_quads = get_min_quads_threshold(input_mesh, args.min_quads)
    boundary_edges, nonmanifold_edges = count_quad_topology_issues(merged_path)
    violation = candidate_violation(metrics, min_quads, args.max_ad, args.min_jr)
    budget_terms = candidate_budget_terms(
        metrics,
        target_quads=target_quads,
        slack_ratio=args.budget_tolerance_ratio,
    )
    metrics["output_path"] = os.path.abspath(merged_path)
    metrics["boundary_edges"] = boundary_edges
    metrics["nonmanifold_edges"] = nonmanifold_edges
    metrics["constraint_violation"] = violation
    metrics["budget_terms"] = budget_terms
    metrics["rank_key"] = list(rank_candidate(metrics, violation, budget_terms))

    shutil.copyfile(merged_path, output_obj)

    summary_path = os.path.join(candidate_root, "uniform_chunk_density_summary.json")
    summary = {
        "mesh_path": os.path.abspath(input_mesh),
        "output_path": os.path.abspath(output_obj),
        "target_quads": target_quads,
        "input_area": input_area,
        "target_quad_density": density_target,
        "selected_chunks": selected_chunks,
        "merged": metrics,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("[Uniform Chunk Density] Merged candidate selected")
    print(f"  target_quads : {'disabled' if target_quads is None else target_quads}")
    print(f"  n_quads      : {metrics['n_quads']}")
    print(f"  AD_mean_deg  : {metrics['AD_mean_deg']:.3f}")
    print(f"  JR_min       : {metrics['JR_min']:.4f}")
    print(f"  saved output : {output_obj}")
    print(f"  saved summary: {summary_path}")
    return output_obj


def select_merged_chunk_candidate(chunk_summaries, input_mesh, output_obj,
                                  candidate_root, args):
    """Choose chunk extraction candidates after merging whole-mesh outputs."""
    from extract_quad import (
        candidate_budget_terms,
        candidate_violation,
        count_triangle_faces,
        evaluate_quad_mesh,
        get_min_quads_threshold,
        maybe_apply_catmull_clark,
        rank_candidate,
        sanitize_float_tag,
    )

    by_chunk = []
    for summary_path in chunk_summaries:
        with open(summary_path, "r") as f:
            summary = json.load(f)
        candidates = {}
        for candidate in summary.get("candidates", []):
            path = candidate.get("output_path")
            if path and os.path.isfile(path) and os.path.getsize(path) > 0:
                candidates[float(candidate["gradient_size"])] = path
        if not candidates:
            raise RuntimeError(f"no usable chunk candidates in {summary_path}")
        by_chunk.append(candidates)

    common_sizes = sorted(set.intersection(*[set(c.keys()) for c in by_chunk]))
    if not common_sizes:
        raise RuntimeError("no common gradient_size candidates across chunks")

    os.makedirs(candidate_root, exist_ok=True)
    min_quads = get_min_quads_threshold(input_mesh, args.min_quads)
    target_quads = None
    if args.target_quad_ratio > 0:
        target_quads = int(round(count_triangle_faces(input_mesh) * args.target_quad_ratio))

    print("\n[Global Chunk Sweep]")
    print(f"  common sweep values: {common_sizes}")
    print(f"  min_quads          : {min_quads}")
    print(f"  target_quads       : {'disabled' if target_quads is None else target_quads}")
    print(f"  output             : {output_obj}")

    candidates = []
    for gs in common_sizes:
        tag = sanitize_float_tag(gs)
        merged_path = os.path.join(candidate_root, f"merged_gs{tag}.obj")
        merge_quad_chunks([chunk[gs] for chunk in by_chunk], merged_path)

        subdiv_info = maybe_apply_catmull_clark(merged_path, input_mesh, args)
        metrics = evaluate_quad_mesh(merged_path)
        if subdiv_info is not None:
            metrics.update(subdiv_info)

        boundary_edges, nonmanifold_edges = count_quad_topology_issues(merged_path)
        violation = candidate_violation(metrics, min_quads, args.max_ad, args.min_jr)
        budget_terms = candidate_budget_terms(
            metrics,
            target_quads=target_quads,
            slack_ratio=args.budget_tolerance_ratio,
        )
        topology_rank = (
            int(boundary_edges > 0) + int(nonmanifold_edges > 0),
            int(nonmanifold_edges),
            int(boundary_edges),
        )
        metrics["gradient_size"] = float(gs)
        metrics["output_path"] = os.path.abspath(merged_path)
        metrics["boundary_edges"] = boundary_edges
        metrics["nonmanifold_edges"] = nonmanifold_edges
        metrics["topology_violation"] = {
            "boundary_edges": boundary_edges,
            "nonmanifold_edges": nonmanifold_edges,
            "num_failed_constraints": topology_rank[0],
        }
        metrics["constraint_violation"] = violation
        metrics["budget_terms"] = budget_terms
        metrics["is_valid"] = (
            violation["num_failed_constraints"] == 0 and topology_rank[0] == 0
        )
        metrics["rank_key"] = list(
            tuple(rank_candidate(metrics, violation, budget_terms)[:4]) +
            topology_rank +
            tuple(rank_candidate(metrics, violation, budget_terms)[4:])
        )
        candidates.append(metrics)

        status = "valid" if metrics["is_valid"] else "constraint-violating"
        print(
            f"  RESULT [{status}] gs={gs} "
            f"n_quads={metrics['n_quads']} "
            f"AD_mean={metrics['AD_mean_deg']:.3f}deg "
            f"JR_min={metrics['JR_min']:.4f} "
            f"boundary_edges={boundary_edges} "
            f"nonmanifold_edges={nonmanifold_edges}"
        )

    candidates.sort(key=lambda m: tuple(m["rank_key"]))
    best = candidates[0]
    shutil.copyfile(best["output_path"], output_obj)

    summary_path = os.path.join(candidate_root, "global_chunk_sweep_summary.json")
    summary = {
        "mesh_path": os.path.abspath(input_mesh),
        "output_path": os.path.abspath(output_obj),
        "selection_rule": {
            "priority": [
                "fewest failed constraints",
                "smallest quad deficit",
                "smallest angle-distortion excess",
                "smallest Jacobian deficit",
                "closed topology before budget/quality",
                "fewest non-manifold edges",
                "fewest boundary edges",
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
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("[Global Chunk Sweep] Best merged candidate selected")
    print(f"  gradient_size: {best['gradient_size']}")
    print(f"  n_quads      : {best['n_quads']}")
    print(f"  AD_mean_deg  : {best['AD_mean_deg']:.3f}")
    print(f"  JR_min       : {best['JR_min']:.4f}")
    print(f"  saved output : {output_obj}")
    print(f"  saved summary: {summary_path}")
    return output_obj


MIN_CHUNK_FACES = 10


def _merge_tiny_chunks(mesh, chunks):
    """Merge chunks with fewer than MIN_CHUNK_FACES faces into nearest
    larger neighbour.  Returns the new (possibly smaller) chunk list."""
    from scipy.spatial import cKDTree

    big = [(i, c) for i, c in enumerate(chunks) if len(c) >= MIN_CHUNK_FACES]
    tiny = [(i, c) for i, c in enumerate(chunks) if len(c) < MIN_CHUNK_FACES]
    if not tiny or not big:
        return chunks

    big_centroids = []
    for _, c in big:
        verts = mesh.vertices[mesh.faces[c].reshape(-1)]
        big_centroids.append(verts.mean(axis=0))
    tree = cKDTree(np.array(big_centroids))

    merged = {i: list(c) for i, c in big}
    for _, c in tiny:
        verts = mesh.vertices[mesh.faces[c].reshape(-1)]
        centroid = verts.mean(axis=0)
        _, nearest = tree.query(centroid)
        target_idx = big[nearest][0]
        merged[target_idx].extend(c.tolist())

    result = [np.asarray(sorted(v), dtype=np.int64) for v in merged.values()]
    result.sort(key=len, reverse=True)
    n_merged = len(tiny)
    if n_merged > 0:
        print(f"  merged {n_merged} tiny chunk(s) (<{MIN_CHUNK_FACES} faces) into neighbours")
    return result


def run_chunked_quad_extract(input_mesh, feat_path, instruction_meta_path,
                             crossfield_txt, output_obj, args):
    mesh = trimesh.load_mesh(input_mesh, process=False)
    chunks, strategy = plan_extraction_chunks(mesh, feat_path, instruction_meta_path, args)

    if len(chunks) <= 1:
        run_quad_extract(input_mesh, crossfield_txt, output_obj, args,
                         feat_path=feat_path)
        return {
            "strategy": strategy,
            "n_chunks": 1,
            "chunk_face_counts": [int(len(chunks[0]))],
            "chunk_dir": None,
        }

    chunks = _merge_tiny_chunks(mesh, chunks)

    print(f"  extraction strategy: {strategy} ({len(chunks)} chunks)")
    print(f"  chunk face counts  : {[int(len(chunk)) for chunk in chunks]}")

    crossfield_rows = load_crossfield_rows(crossfield_txt)
    if len(crossfield_rows) != len(mesh.faces):
        raise ValueError(
            f"Cross-field row count ({len(crossfield_rows)}) does not match face count ({len(mesh.faces)}).")

    chunk_root = os.path.splitext(output_obj)[0] + "_chunks"
    chunk_input_dir = os.path.join(chunk_root, "inputs")
    chunk_output_dir = os.path.join(chunk_root, "outputs")
    os.makedirs(chunk_input_dir, exist_ok=True)
    os.makedirs(chunk_output_dir, exist_ok=True)

    chunk_outputs = []
    chunk_summaries = []
    succeeded = 0
    failed_chunks = []
    for chunk_idx, face_ids in enumerate(chunks):
        chunk_name = f"chunk_{chunk_idx:03d}"
        chunk_mesh = os.path.join(chunk_input_dir, f"{chunk_name}.obj")
        chunk_crossfield = os.path.join(chunk_input_dir, f"{chunk_name}_crossfield.txt")
        chunk_output = os.path.join(chunk_output_dir, f"{chunk_name}_quad.obj")
        chunk_work_dir = os.path.join(chunk_output_dir, f"{chunk_name}_quad_extract_work")
        chunk_summary = os.path.join(chunk_work_dir, f"{chunk_name}_quad_sweep_summary.json")

        write_face_subset_obj(mesh, face_ids, chunk_mesh)
        write_subset_crossfield(crossfield_rows, face_ids, chunk_crossfield)
        chunk_ok = True
        try:
            if args.disable_auto_sweep:
                run_quad_extract(chunk_mesh, chunk_crossfield, chunk_output, args)
            else:
                chunk_args = argparse.Namespace(**vars(args))
                chunk_args.keep_sweep_outputs = True
                chunk_args.keep_extract_intermediates = True
                chunk_args.intermediate_dir = chunk_work_dir
                chunk_args.summary_json = chunk_summary
                chunk_args.min_quads = None
                chunk_args.target_quad_ratio = 0.0
                chunk_args.catmull_clark_iters = 0
                chunk_args.max_catmull_clark_iters = 0
                run_quad_extract(chunk_mesh, chunk_crossfield, chunk_output, chunk_args)
        except Exception as exc:  # noqa: BLE001
            chunk_ok = False
            failed_chunks.append(f"{chunk_name}: {exc}")
            print(f"  chunk extraction failed ({chunk_name}): {exc}")

        if chunk_ok and os.path.isfile(chunk_output) and os.path.getsize(chunk_output) > 0:
            succeeded += 1
            if not args.disable_auto_sweep and os.path.isfile(chunk_summary):
                chunk_summaries.append(chunk_summary)
        elif chunk_ok:
            failed_chunks.append(f"{chunk_name}: empty output")
            print(f"  chunk extraction failed ({chunk_name}): empty output")
        chunk_outputs.append(chunk_output)

    print(f"\n  chunk summary: {succeeded}/{len(chunks)} OK, {len(failed_chunks)} failed")
    if failed_chunks:
        print(f"  failed chunks: {failed_chunks}")

    keep_intermediates = getattr(args, "keep_extract_intermediates", False)
    if failed_chunks:
        print("  chunk extraction incomplete; retrying once with global extraction")
        if os.path.isfile(output_obj):
            os.remove(output_obj)
        run_quad_extract(input_mesh, crossfield_txt, output_obj, args,
                         feat_path=feat_path)
        if not os.path.isfile(output_obj) or os.path.getsize(output_obj) == 0:
            raise RuntimeError(
                f"global fallback did not produce a valid quad mesh: {output_obj}")
        if not keep_intermediates:
            shutil.rmtree(chunk_root, ignore_errors=True)
        return {
            "strategy": f"{strategy}_fallback_global",
            "n_chunks": int(len(chunks)),
            "chunk_face_counts": [int(len(chunk)) for chunk in chunks],
            "chunk_dir": chunk_root if keep_intermediates else None,
            "chunks_succeeded": succeeded,
            "chunks_failed": len(failed_chunks),
            "failed_chunk_names": failed_chunks,
        }

    if args.disable_auto_sweep:
        merge_quad_chunks(chunk_outputs, output_obj)
    elif getattr(args, "allow_per_chunk_size", False):
        merge_quad_chunks(chunk_outputs, output_obj)
    else:
        if len(chunk_summaries) != len(chunks):
            raise RuntimeError(
                f"uniform density merge needs {len(chunks)} summaries, got {len(chunk_summaries)}")
        select_uniform_density_chunk_candidate(
            chunk_summaries,
            input_mesh,
            output_obj,
            os.path.join(chunk_root, "merged_candidates"),
            args,
        )
    if not os.path.isfile(output_obj) or os.path.getsize(output_obj) == 0:
        raise RuntimeError(f"merged quad mesh is empty: {output_obj}")
    if not keep_intermediates:
        shutil.rmtree(chunk_root, ignore_errors=True)
    final_strategy = strategy
    if not args.disable_auto_sweep and getattr(args, "allow_per_chunk_size", False):
        final_strategy = f"{strategy}_per_chunk_size"
    elif not args.disable_auto_sweep:
        final_strategy = f"{strategy}_uniform_density"
    return {
        "strategy": final_strategy,
        "n_chunks": int(len(chunks)),
        "chunk_face_counts": [int(len(chunk)) for chunk in chunks],
        "chunk_dir": chunk_root if keep_intermediates else None,
        "chunks_succeeded": succeeded,
        "chunks_failed": len(failed_chunks),
        "failed_chunk_names": failed_chunks,
    }


def run_instruction_patch_extract(input_mesh, instruction_meta_path, crossfield_txt, output_root, args):
    mesh = trimesh.load_mesh(input_mesh, process=False)
    patch_records = compute_instruction_target_patches(mesh, instruction_meta_path, args)
    if not patch_records:
        return None

    crossfield_rows = load_crossfield_rows(crossfield_txt)
    if len(crossfield_rows) != len(mesh.faces):
        raise ValueError(
            f"Cross-field row count ({len(crossfield_rows)}) does not match face count ({len(mesh.faces)}).")

    patch_input_dir = os.path.join(output_root, "inputs")
    patch_output_dir = os.path.join(output_root, "outputs")
    os.makedirs(patch_input_dir, exist_ok=True)
    os.makedirs(patch_output_dir, exist_ok=True)

    print(f"  local target patches: {len(patch_records)}")
    print(f"  patch face counts   : {[record['n_faces'] for record in patch_records]}")
    print(f"  target face counts  : {[record['n_target_faces'] for record in patch_records]}")

    output_paths = []
    for patch_idx, record in enumerate(patch_records):
        face_ids = record["face_ids"]
        patch_name = f"patch_{patch_idx:03d}"
        patch_mesh = os.path.join(patch_input_dir, f"{patch_name}.obj")
        patch_crossfield = os.path.join(patch_input_dir, f"{patch_name}_crossfield.txt")
        patch_output = os.path.join(patch_output_dir, f"{patch_name}_quad.obj")

        write_face_subset_obj(mesh, face_ids, patch_mesh)
        write_subset_crossfield(crossfield_rows, face_ids, patch_crossfield)
        run_quad_extract(patch_mesh, patch_crossfield, patch_output, args)
        output_paths.append(patch_output)

    return {
        "strategy": "instruction_target_patches",
        "n_patches": int(len(patch_records)),
        "patch_face_counts": [int(record["n_faces"]) for record in patch_records],
        "patch_target_face_counts": [int(record["n_target_faces"]) for record in patch_records],
        "patch_dir": output_root,
        "patch_outputs": output_paths,
    }


def run_quad_extract(input_mesh, crossfield_txt, output_obj, args,
                     feat_path=None):
    """Run safer quad extraction through extract_quad.py."""
    if not os.path.isfile(EXTRACT_QUAD_PY):
        sys.exit(f"Quad extraction script not found: {EXTRACT_QUAD_PY}")
    cmd = [
        sys.executable,
        EXTRACT_QUAD_PY,
        "--mesh", os.path.abspath(input_mesh),
        "--crossfield", os.path.abspath(crossfield_txt),
        "--output", os.path.abspath(output_obj),
        "--gradient_size", str(args.gradient_size),
        "--timeout", str(args.extract_timeout),
        "--max_ad", str(args.max_ad),
        "--min_jr", str(args.min_jr),
    ]
    if args.min_quads is not None:
        cmd.extend(["--min_quads", str(args.min_quads)])
    if not args.disable_auto_sweep:
        cmd.append("--auto_sweep")
    if args.sweep_values:
        cmd.extend(["--sweep_values", *[str(v) for v in args.sweep_values]])
    if args.keep_sweep_outputs:
        cmd.append("--keep_sweep_outputs")
    if getattr(args, "intermediate_dir", None):
        cmd.extend(["--intermediate_dir", os.path.abspath(args.intermediate_dir)])
    if getattr(args, "summary_json", None):
        cmd.extend(["--summary_json", os.path.abspath(args.summary_json)])
    if getattr(args, "keep_extract_intermediates", False):
        cmd.append("--keep_intermediates")
    if not args.disable_extract_retry:
        cmd.append("--retry")
    if getattr(args, "max_fallback_attempts", None) is not None:
        cmd.extend(["--max_fallback_attempts", str(args.max_fallback_attempts)])
    if getattr(args, "max_repair_tiers", None) is not None:
        cmd.extend(["--max_repair_tiers", str(args.max_repair_tiers)])
    if args.catmull_clark_iters > 0:
        cmd.extend(["--catmull_clark_iters", str(args.catmull_clark_iters)])
    if args.target_quad_ratio is not None:
        cmd.extend(["--target_quad_ratio", str(args.target_quad_ratio)])
    if args.budget_tolerance_ratio is not None:
        cmd.extend(["--budget_tolerance_ratio", str(args.budget_tolerance_ratio)])
    if args.max_catmull_clark_iters is not None:
        cmd.extend(["--max_catmull_clark_iters", str(args.max_catmull_clark_iters)])
    if getattr(args, 'snap_to_boundary', False) and feat_path is not None:
        cmd.extend([
            "--snap_to_boundary",
            "--snap_feat_path", os.path.abspath(feat_path),
            "--snap_fraction", str(args.snap_fraction),
            "--snap_threshold_ratio", str(args.snap_threshold_ratio),
            "--snap_min_angle", str(getattr(args, 'snap_min_angle', 25.0)),
            "--snap_smooth_iters", str(getattr(args, 'snap_smooth_iters', 3)),
            "--snap_smooth_weight", str(getattr(args, 'snap_smooth_weight', 0.3)),
        ])

    print(f"\n[Stage 3] Quad extraction")
    print(f"  script      : {EXTRACT_QUAD_PY}")
    print(f"  python      : {sys.executable}")
    print(f"  input mesh  : {input_mesh}")
    print(f"  cross field : {crossfield_txt}")
    print(f"  output      : {output_obj}")
    print(f"  gradient_size: {args.gradient_size}\n")
    print(f"  timeout     : {args.extract_timeout}")
    print(f"  auto_sweep  : {not args.disable_auto_sweep}")
    print(f"  retry       : {not args.disable_extract_retry}")
    print(f"  min_quads   : {'default' if args.min_quads is None else args.min_quads}")
    print(f"  max_ad      : {args.max_ad}")
    print(f"  min_jr      : {args.min_jr}\n")
    print(f"  cc_iters    : {args.catmull_clark_iters}")
    print(f"  target_ratio: {args.target_quad_ratio}")
    print(f"  budget_tol  : {args.budget_tolerance_ratio}")
    print(f"  cc_max_auto : {args.max_catmull_clark_iters}\n")

    env = os.environ.copy()
    subprocess.run(cmd, env=env, check=True)
    return output_obj


def derive_instruction_extract_args(base_args, instruction_meta_path):
    if base_args.guidance_mode != "instruction":
        return base_args
    if instruction_meta_path is None:
        return base_args

    metadata = load_instruction_metadata(instruction_meta_path)
    local_args = argparse.Namespace(**vars(base_args))
    policy_enabled = False
    if base_args.instruction_prototype_path is not None:
        prototype_db = load_instruction_prototypes(base_args.instruction_prototype_path)
        policy = recommend_extraction_policy(
            metadata,
            prototype_db,
            num_timeline_buckets=base_args.instruction_timeline_buckets,
        )
        local_args.gradient_size = float(policy["gradient_size"])
        local_args.target_quad_ratio = float(policy["target_quad_ratio"])
        local_args.budget_tolerance_ratio = float(
            policy.get("budget_tolerance_ratio", local_args.budget_tolerance_ratio))
        local_args.extract_chunk_min_faces = int(
            min(local_args.extract_chunk_min_faces, policy["extract_chunk_min_faces"]))
        if not local_args.sweep_values:
            local_args.sweep_values = list(policy["sweep_values"])
        policy_enabled = True

    feature_type = np.asarray(metadata["feature_type_id"], dtype=np.int64)
    target_region = np.asarray(
        metadata.get("target_region_mask", np.ones_like(feature_type, dtype=np.uint8)),
        dtype=np.uint8,
    )
    blend_target_mask = (target_region > 0) & np.isin(feature_type, [2, 3])
    blend_target_ratio = float(blend_target_mask.mean()) if len(blend_target_mask) > 0 else 0.0
    if blend_target_ratio >= 0.05:
        local_args.gradient_size = min(float(local_args.gradient_size), 24.0)
        local_args.target_quad_ratio = max(float(local_args.target_quad_ratio), 0.58)
        local_args.budget_tolerance_ratio = min(float(local_args.budget_tolerance_ratio), 0.16)
        local_args.extract_chunk_min_faces = min(int(local_args.extract_chunk_min_faces), 160)
        if not local_args.sweep_values:
            local_args.sweep_values = [24.0, 20.0, 28.0]
        policy_enabled = True

    if base_args.reconstruction_whole_mesh:
        local_args.stitch_quad_patches = True
        local_args.stitch_threshold_ratio = 0.005
        local_args.stitch_smooth_iters = 2
        local_args.stitch_smooth_weight = 0.2

    if policy_enabled:
        print("  instruction policy: enabled")
        print(f"  policy gradient  : {local_args.gradient_size}")
        print(f"  policy sweep     : {local_args.sweep_values}")
        print(f"  policy quad ratio: {local_args.target_quad_ratio}")
        print(f"  policy budget tol: {local_args.budget_tolerance_ratio}")
        print(f"  policy chunk min : {local_args.extract_chunk_min_faces}")
        print(f"  blend target ratio: {blend_target_ratio:.4f}")
    if base_args.reconstruction_whole_mesh:
        print("  reconstruction whole-mesh: stitch_quad_patches enabled")
    return local_args


def export_scalar_preview(mesh, values, output_path):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) != len(mesh.faces):
        raise ValueError(
            f"Value count ({len(values)}) does not match face count ({len(mesh.faces)}).")

    anchors = np.array([0.0, 0.33, 0.66, 1.0], dtype=np.float64)
    colors = np.array([
        [24, 34, 84],
        [53, 196, 233],
        [244, 208, 63],
        [192, 57, 43],
    ], dtype=np.float64)

    rgba = np.zeros((len(values), 4), dtype=np.uint8)
    clipped = np.clip(values, 0.0, 1.0)
    for c in range(3):
        rgba[:, c] = np.interp(clipped, anchors, colors[:, c]).astype(np.uint8)
    rgba[:, 3] = 255

    preview_mesh = mesh.copy()
    preview_mesh.visual.face_colors = rgba
    preview_mesh.export(output_path)


def build_semantic_size_field(mesh_path, feat_path, basename, output_root, args):
    mesh = trimesh.load_mesh(mesh_path, process=False)
    face_count = len(mesh.faces)
    features = np.load(feat_path)
    if len(features) != face_count:
        raise ValueError(
            f"[{basename}] Feature count ({len(features)}) does not match face count ({face_count}).")
    if args.semantic_size_min <= 0.0 or args.semantic_size_min > args.semantic_size_max:
        raise ValueError("Require 0 < semantic_size_min <= semantic_size_max.")

    vertex_neighbors = get_face_neighbors(mesh)
    semantic_args = argparse.Namespace(
        k=args.semantic_size_k,
        k_min=args.semantic_size_k_min,
        k_max=args.semantic_size_k_max,
        semantic_grad_mix=args.semantic_size_grad_mix,
    )
    semantic_info = compute_semantic_component(mesh, features, vertex_neighbors, semantic_args)
    semantic_complexity = robust_normalize(
        semantic_info["raw"], percentile=args.semantic_size_robust_percentile)
    size_hint = args.semantic_size_max - semantic_complexity * (
        args.semantic_size_max - args.semantic_size_min
    )

    size_dir = os.path.join(output_root, "semantic_size_fields")
    os.makedirs(size_dir, exist_ok=True)
    size_txt = os.path.join(size_dir, f"{basename}_semantic_size.txt")
    np.savetxt(size_txt, size_hint, fmt="%.8f")

    summary = {
        "mesh": os.path.abspath(mesh_path),
        "features": os.path.abspath(feat_path),
        "face_count": int(face_count),
        "k_used": int(semantic_info["k_used"]),
        "silhouette": semantic_info["silhouette"],
        "n_boundary_edges": int(semantic_info["n_boundary_edges"]),
        "semantic_grad_mix": float(args.semantic_size_grad_mix),
        "robust_percentile": float(args.semantic_size_robust_percentile),
        "size_min": float(args.semantic_size_min),
        "size_max": float(args.semantic_size_max),
        "complexity_stats": {
            "min": float(semantic_complexity.min()),
            "max": float(semantic_complexity.max()),
            "mean": float(semantic_complexity.mean()),
            "p90": float(np.percentile(semantic_complexity, 90)),
            "p95": float(np.percentile(semantic_complexity, 95)),
        },
        "size_hint_stats": {
            "min": float(size_hint.min()),
            "max": float(size_hint.max()),
            "mean": float(size_hint.mean()),
            "p10": float(np.percentile(size_hint, 10)),
            "p50": float(np.percentile(size_hint, 50)),
        },
    }
    summary_path = os.path.join(size_dir, f"{basename}_semantic_size_summary.json")
    with open(summary_path, "w") as f:
        import json
        json.dump(summary, f, indent=2)

    preview_path = None
    if args.semantic_size_save_vis:
        preview_path = os.path.join(size_dir, f"{basename}_semantic_complexity.ply")
        export_scalar_preview(mesh, semantic_complexity, preview_path)

    silhouette_str = "n/a"
    if semantic_info["silhouette"] is not None:
        silhouette_str = f"{semantic_info['silhouette']:.4f}"
    print(f"  semantic size field: {size_txt}")
    print(f"  semantic clusters  : K={semantic_info['k_used']}, "
          f"silhouette={silhouette_str}")
    print(f"  size range         : [{size_hint.min():.4f}, {size_hint.max():.4f}]")

    return {
        "size_field_path": size_txt,
        "summary_path": summary_path,
        "preview_path": preview_path,
    }


# ---------------------------------------------------------------------------
#  Verification
# ---------------------------------------------------------------------------

def verify_features(feat_path, expected_faces, name):
    """Sanity-check that feature count equals face count."""
    feats = np.load(feat_path)
    if feats.shape[0] != expected_faces:
        raise ValueError(
            f"[{name}] Feature row count ({feats.shape[0]}) != face count "
            f"({expected_faces}).  Face ordering would be inconsistent.")
    print(f"  [{name}] Feature shape {feats.shape} matches "
          f"{expected_faces} faces.")


def verify_instruction_metadata(meta_path, expected_faces, name):
    metadata = load_instruction_metadata(meta_path)
    face_count = int(metadata["mesh_face_count"][0])
    if face_count != expected_faces:
        raise ValueError(
            f"[{name}] Instruction metadata face count ({face_count}) != face count "
            f"({expected_faces}). Face ordering would be inconsistent.")
    print(f"  [{name}] Instruction metadata matches {expected_faces} faces.")


def build_instruction_metadata_for_mesh(mesh_path, output_path, args):
    if args.instruction_meta_path is not None:
        return os.path.abspath(args.instruction_meta_path)
    if args.instruction_dataset_root is None:
        raise ValueError("Instruction mode requires instruction metadata or dataset root.")

    dataset_paths = infer_dataset_paths(
        os.path.abspath(args.instruction_dataset_root),
        derive_mesh_basename(mesh_path),
    )
    metadata = build_instruction_metadata(
        mesh_path=mesh_path,
        fidx_path=dataset_paths["fidx"],
        timeline_path=dataset_paths["timeline"],
        seg_path=dataset_paths["seg"],
    )
    return save_instruction_metadata(output_path, metadata)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not args.no_timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"{args.output_dir}_{ts}"

    os.makedirs(args.output_dir, exist_ok=True)

    mesh_paths = collect_meshes(args)
    n_total = len(mesh_paths)
    print(f"Found {n_total} mesh(es) to process:\n")
    for mp in mesh_paths:
        print(f"  - {os.path.basename(mp)}")
    print()

    info_list = []
    if args.guidance_mode == "feature":
        # ------------------------------------------------------------------
        # Stage 1: PartField feature extraction (all meshes at once)
        # ------------------------------------------------------------------
        if args.skip_partfield:
            if n_total != 1:
                sys.exit("--skip_partfield only works with a single --input_mesh")
            if not args.part_feat_path or not os.path.isfile(args.part_feat_path):
                sys.exit("--skip_partfield requires a valid --part_feat_path")

            basename = os.path.splitext(os.path.basename(mesh_paths[0]))[0]
            mesh = trimesh.load_mesh(mesh_paths[0], process=False)
            nc_path = mesh_paths[0]
            n_faces = len(mesh.faces)
            if args.max_faces > 0 and n_faces > args.max_faces:
                mesh, _ = decimate_mesh(mesh, args.max_faces)
                processed_dir = os.path.join(args.output_dir, "processed_meshes")
                os.makedirs(processed_dir, exist_ok=True)
                nc_path = os.path.join(processed_dir, f"{basename}.obj")
                mesh.export(nc_path)
                print(f"  Decimated {n_faces} -> {len(mesh.faces)} faces")
                n_faces = len(mesh.faces)
            info_list = [(basename, nc_path, n_faces)]
            feat_dir = None
            feat_override = os.path.abspath(args.part_feat_path)
            print("[Stage 1] Skipped. Using existing features.\n")
        else:
            print("=" * 60)
            print("  Stage 1: PartField feature extraction")
            print("=" * 60)

            staging_dir, info_list = prepare_meshes(
                mesh_paths, args.output_dir, args.max_faces)
            feat_dir = run_partfield(staging_dir, args)
            feat_override = None
    else:
        print("=" * 60)
        print("  Stage 1: Geometry preparation")
        print("=" * 60)
        geometry_max_faces = args.max_faces
        if args.guidance_mode == "instruction" and args.instruction_meta_path is None:
            if args.max_faces > 0:
                print("WARNING: disabling decimation in instruction mode to preserve face-level metadata alignment.")
            geometry_max_faces = 0
        _, info_list = prepare_meshes(mesh_paths, args.output_dir, geometry_max_faces)
        feat_dir = None
        feat_override = None
        print(f"[Stage 1] PartField skipped in guidance mode '{args.guidance_mode}'.\n")

    if len(info_list) == 0:
        sys.exit("No valid meshes remained after input loading/parsing.")

    feat_map = {}
    instruction_meta_map = {}
    if args.guidance_mode == "feature":
        print("\n[Check] Verifying feature files ...")
        for basename, orig_path, n_faces in info_list:
            if feat_override:
                fp = feat_override
            else:
                fp = find_feature_file(feat_dir, basename)
            verify_features(fp, n_faces, basename)
            feat_map[basename] = (orig_path, fp)
            instruction_meta_map[basename] = None
    elif args.guidance_mode == "instruction":
        print("\n[Check] Building / verifying instruction metadata ...")
        meta_dir = os.path.join(args.output_dir, "instruction_meta")
        os.makedirs(meta_dir, exist_ok=True)
        original_mesh_by_name = {
            os.path.splitext(os.path.basename(mp))[0]: mp
            for mp in mesh_paths
        }
        # Auto-detect reconstruction layout and enable whole-mesh mode
        if (not args.reconstruction_whole_mesh
                and args.instruction_dataset_root is not None):
            for basename, _, _ in info_list:
                dataset_paths = infer_dataset_paths(
                    os.path.abspath(args.instruction_dataset_root), basename)
                if dataset_paths.get("layout") == "reconstruction":
                    args.reconstruction_whole_mesh = True
                    print("[Auto] Detected reconstruction layout -> "
                          "enabling --reconstruction_whole_mesh")
                    break
        for basename, nc_path, n_faces in info_list:
            raw_mesh = original_mesh_by_name.get(basename, nc_path)
            meta_path = build_instruction_metadata_for_mesh(
                raw_mesh,
                os.path.join(meta_dir, f"{basename}_instruction_meta.npz"),
                args
            )
            verify_instruction_metadata(meta_path, n_faces, basename)
            feat_map[basename] = (nc_path, None)
            instruction_meta_map[basename] = meta_path
    else:
        for basename, orig_path, _ in info_list:
            feat_map[basename] = (orig_path, None)
            instruction_meta_map[basename] = None

    # ------------------------------------------------------------------
    # Stage 2: NeurCross training (one per mesh)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Stage 2: NeurCross training")
    print("=" * 60)

    results = {}
    for idx, (basename, (orig_path, fp)) in enumerate(feat_map.items(), 1):
        print(f"\n--- [{idx}/{n_total}] {basename} "
              + "-" * (45 - len(basename)))
        instruction_meta_path = instruction_meta_map[basename]
        logdir = run_neurcross(orig_path, fp, instruction_meta_path, basename, args)
        results[basename] = {
            "input": orig_path,
            "features": fp,
            "instruction_meta": instruction_meta_path,
            "logdir": logdir,
        }

    # ------------------------------------------------------------------
    # Stage 3: Quad mesh extraction  (MIQ + libQEx)
    # ------------------------------------------------------------------
    if not args.skip_extract:
        print("\n" + "=" * 60)
        print("  Stage 3: Quad mesh extraction (MIQ + libQEx)")
        print("=" * 60)

        quad_output_dir = os.path.join(
            os.path.abspath(args.output_dir), "quad_meshes")
        os.makedirs(quad_output_dir, exist_ok=True)

        for idx, (basename, r) in enumerate(results.items(), 1):
            print(f"\n--- [{idx}/{n_total}] {basename} "
                  + "-" * (45 - len(basename)))

            cross_field_dir = os.path.join(r["logdir"], "save_crossField")
            cf_txt = find_latest_crossfield(cross_field_dir)
            if cf_txt is None:
                print(f"  WARNING: No cross field found in {cross_field_dir}")
                print(f"           Skipping quad extraction for {basename}")
                continue

            input_mesh_obj = r["input"]
            ext = os.path.splitext(input_mesh_obj)[1].lower()
            if ext != ".obj":
                obj_path = os.path.join(quad_output_dir,
                                        f"{basename}_input.obj")
                mesh = trimesh.load_mesh(input_mesh_obj, process=False)
                mesh.export(obj_path)
                input_mesh_obj = obj_path

            output_obj = os.path.join(quad_output_dir, f"{basename}_quad.obj")
            try:
                extract_args = derive_instruction_extract_args(args, r.get("instruction_meta"))
                extract_meta = run_chunked_quad_extract(
                    input_mesh_obj,
                    r["features"],
                    r.get("instruction_meta"),
                    cf_txt,
                    output_obj,
                    extract_args
                )
                results[basename]["quad_mesh"] = output_obj
                results[basename]["extract_strategy"] = extract_meta["strategy"]
                results[basename]["extract_chunks"] = extract_meta["n_chunks"]
                results[basename]["extract_chunk_faces"] = extract_meta["chunk_face_counts"]
                if extract_meta["chunk_dir"] is not None:
                    results[basename]["extract_chunk_dir"] = extract_meta["chunk_dir"]
                if (
                        args.guidance_mode == "instruction" and
                        args.instruction_patch_extract and
                        r.get("instruction_meta") is not None):
                    patch_root = os.path.join(
                        quad_output_dir, f"{basename}_target_patches")
                    patch_meta = run_instruction_patch_extract(
                        input_mesh_obj,
                        r["instruction_meta"],
                        cf_txt,
                        patch_root,
                        extract_args,
                    )
                    if patch_meta is not None:
                        results[basename]["patch_extract_strategy"] = patch_meta["strategy"]
                        results[basename]["patch_extract_count"] = patch_meta["n_patches"]
                        results[basename]["patch_extract_face_counts"] = patch_meta["patch_face_counts"]
                        results[basename]["patch_extract_target_face_counts"] = patch_meta["patch_target_face_counts"]
                        results[basename]["patch_extract_dir"] = patch_meta["patch_dir"]
            except (subprocess.CalledProcessError, RuntimeError, ValueError) as e:
                print(f"  ERROR: Quad extraction failed for {basename}: {e}")
                results[basename]["quad_mesh"] = None
    else:
        print("\n[Stage 3] Skipped (--skip_extract)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    abs_output = os.path.abspath(args.output_dir)
    print("\n" + "=" * 60)
    print("  Pipeline finished successfully")
    print("=" * 60)
    print(f"\nOutput directory: {abs_output}\n")
    if args.guidance_mode == "feature":
        print(f"  partfield_input/        Staged mesh files for PartField")
        print(f"  partfield_features/     Per-face features (.npy) + PCA vis (.ply)")
    if args.guidance_mode == "instruction":
        print(f"  instruction_meta/       Per-face instruction metadata (.npz)")
    print(f"  neurcross_logs/         NeurCross training results")
    print(f"  quad_meshes/*_chunks/   Per-chunk intermediates when --enable_chunked_extract is used")
    print(f"  quad_meshes/*_target_patches/  Local target-patch extraction outputs when enabled")
    print(f"  quad_meshes/            Extracted quad meshes (.obj)")
    print()
    for name, r in results.items():
        cross_field_dir = os.path.join(r["logdir"], "save_crossField")
        print(f"  [{name}]")
        print(f"    guidance   : {args.guidance_mode}")
        if r.get("features") is not None and os.path.isfile(r["features"]):
            print(f"    features   : {os.path.relpath(r['features'], abs_output)}")
        if r.get("instruction_meta") is not None and os.path.isfile(r["instruction_meta"]):
            print(f"    instr meta : {os.path.relpath(r['instruction_meta'], abs_output)}")
        print(f"    cross field: {os.path.relpath(cross_field_dir, abs_output)}/")
        strategy = r.get("extract_strategy")
        if strategy:
            print(f"    extraction : {strategy} ({r.get('extract_chunks', 1)} chunks)")
        patch_strategy = r.get("patch_extract_strategy")
        if patch_strategy:
            print(f"    patches    : {patch_strategy} ({r.get('patch_extract_count', 0)} patches)")
        qm = r.get("quad_mesh")
        if qm and os.path.isfile(qm):
            print(f"    quad mesh  : {os.path.relpath(qm, abs_output)}")
        print()


if __name__ == "__main__":
    main()
