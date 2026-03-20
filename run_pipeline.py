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
from datetime import datetime

import numpy as np
import trimesh

from build_complexity_map import (
    compute_semantic_component,
    get_face_neighbors,
    robust_normalize,
)
from eval.label_utils import cluster_features

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
        description="PartField + NeurCross semantic quad-meshing pipeline")

    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input_dir",
                     help="Directory of mesh files (.obj/.ply/.off/.glb). "
                          "All meshes are processed sequentially.")
    inp.add_argument("--input_mesh",
                     help="Path to a single input mesh file")

    # -- Conda environments -------------------------------------------------
    p.add_argument("--env_partfield", default="partfield",
                   help="Conda env name for PartField  (default: partfield)")
    p.add_argument("--env_neurcross", default="neurcross",
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
    p.add_argument("--loss_weights", nargs="+", type=float,
                   default=[7e3, 6e2, 10, 5e1, 30, 3, 1000],
                   help="[sdf, inter, theta_hess, eikonal, theta_neigh, "
                        "morse, semantic]")
    p.add_argument("--semantic_boundary_weight", type=float, default=1.0,
                   help="Internal weight for semantic boundary alignment")
    p.add_argument("--semantic_intra_weight", type=float, default=1.0,
                   help="Internal weight for semantic intra-part consistency")
    p.add_argument("--semantic_neighbor_weight", type=float, default=1.0,
                   help="Internal weight for semantic-aware neighbor smoothness")
    p.add_argument("--semantic_cross_part_gamma", type=float, default=0.2,
                   help="Neighbor smoothness attenuation across semantic parts")

    # -- Quad Extraction (Stage 3) -----------------------------------------
    p.add_argument("--gradient_size", type=float, default=30.0,
                   help="MIQ gradient size (controls quad density)")
    p.add_argument("--size_field_path", default=None,
                   help="Deprecated and ignored. Stage 3 now follows the pure MIQ -> libQEx pipeline.")
    p.add_argument("--size_field_strength", type=float, default=0.75,
                   help="Deprecated and ignored.")
    p.add_argument("--size_field_smooth_iters", type=int, default=6,
                   help="Deprecated and ignored.")
    p.add_argument("--disable_semantic_size_field", action="store_true",
                   help="Deprecated and ignored. Semantic size-field generation is disabled.")
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
                   help="Deprecated and ignored.")
    p.add_argument("--sweep_values", nargs="+", type=float, default=None,
                   help="Optional gradient_size sweep values for safer extraction")
    p.add_argument("--keep_sweep_outputs", action="store_true",
                   help="Keep all intermediate sweep OBJ files")
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
    p.add_argument("--max_catmull_clark_iters", type=int, default=2,
                   help="Maximum automatic Catmull-Clark subdivision iterations")
    p.add_argument("--disable_chunked_extract", action="store_true",
                   help="Disable chunked extraction for assembled meshes")
    p.add_argument("--extract_chunk_min_faces", type=int, default=200,
                   help="Minimum face count required for a semantic extraction chunk")
    p.add_argument("--extract_chunk_max_chunks", type=int, default=24,
                   help="Maximum number of extraction chunks before falling back to global extraction")
    p.add_argument("--skip_extract", action="store_true",
                   help="Skip Stage 3 (quad mesh extraction)")

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
    p.add_argument("--max_faces", type=int, default=DEFAULT_MAX_FACES,
                   help="Decimate meshes exceeding this face count to avoid "
                        "GPU OOM.  Set 0 to disable.  (default: %(default)s)")
    p.add_argument("--gpu_id", default="0",
                   help="CUDA_VISIBLE_DEVICES value")

    args = p.parse_args()

    conda_prefix = find_conda_prefix()
    if not args.python_partfield:
        args.python_partfield = resolve_env_python(
            args.env_partfield, conda_prefix)
    if not args.python_neurcross:
        args.python_neurcross = resolve_env_python(
            args.env_neurcross, conda_prefix)

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
    for mp in mesh_paths:
        basename = os.path.splitext(os.path.basename(mp))[0]
        ext = os.path.splitext(mp)[1].lower()
        mesh = trimesh.load_mesh(mp, process=False)
        n_orig = len(mesh.faces)

        decimated = False
        if max_faces > 0 and n_orig > max_faces:
            mesh, decimated = decimate_mesh(mesh, max_faces)

        n_faces = len(mesh.faces)

        pf_dst = os.path.join(staging_dir, f"{basename}.obj")
        if decimated:
            if not os.path.exists(pf_dst):
                mesh.export(pf_dst)
            nc_path = os.path.join(processed_dir, f"{basename}.obj")
            if not os.path.exists(nc_path):
                mesh.export(nc_path)
            print(f"  {basename}{ext}  decimated {n_orig} -> {n_faces} faces")
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

    return staging_dir, info_list


def run_partfield(staging_dir, args):
    """Run PartField inference on all meshes in *staging_dir*."""
    result_tag = "partfield_features"
    result_name = os.path.relpath(
        os.path.join(os.path.abspath(args.output_dir), result_tag),
        os.path.join(PARTFIELD_DIR, "exp_results"))

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

    feat_dir = os.path.join(PARTFIELD_DIR, "exp_results", result_name)
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

def run_neurcross(input_mesh, feat_path, mesh_name, args):
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
        "--part_feat_path", os.path.abspath(feat_path),
        "--logdir",         logdir_parent,
        "--n_samples",      str(args.n_samples),
        "--n_points",       str(args.n_points),
        "--num_epochs",     str(args.num_epochs),
        "--lr",             str(args.lr),
        "--loss_weights",   *[str(w) for w in args.loss_weights],
        "--semantic_boundary_weight", str(args.semantic_boundary_weight),
        "--semantic_intra_weight", str(args.semantic_intra_weight),
        "--semantic_neighbor_weight", str(args.semantic_neighbor_weight),
        "--semantic_cross_part_gamma", str(args.semantic_cross_part_gamma),
        "--morse_near",
    ]

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


def plan_extraction_chunks(mesh, feat_path, args):
    if args.disable_chunked_extract:
        return [np.arange(len(mesh.faces), dtype=np.int64)], "global"

    components = get_face_connected_components(mesh)
    if len(components) > 1:
        return components, "connected_components"

    if feat_path is not None:
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


def write_subset_scalar_field(values, face_ids, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savetxt(output_path, values[np.asarray(face_ids, dtype=np.int64)], fmt="%.8f")
    return output_path


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
                indices = [int(p.split("/")[0]) - 1 for p in parts]
                if len(indices) == 4:
                    quads.append(indices)

    if not vertices or not quads:
        raise ValueError(f"No valid quad data in {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(quads, dtype=np.int64)


def write_quad_obj(path, vertices, quads):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for vertex in np.asarray(vertices, dtype=np.float64):
            f.write(f"v {vertex[0]:.12g} {vertex[1]:.12g} {vertex[2]:.12g}\n")
        for quad in np.asarray(quads, dtype=np.int64):
            f.write(f"f {quad[0] + 1} {quad[1] + 1} {quad[2] + 1} {quad[3] + 1}\n")


def merge_quad_chunks(chunk_obj_paths, output_path):
    merged_vertices = []
    merged_quads = []
    vertex_offset = 0

    for chunk_path in chunk_obj_paths:
        vertices, quads = load_quad_obj(chunk_path)
        merged_vertices.append(vertices)
        merged_quads.append(quads + vertex_offset)
        vertex_offset += len(vertices)

    vertices = np.vstack(merged_vertices)
    quads = np.vstack(merged_quads)
    write_quad_obj(output_path, vertices, quads)
    return output_path


def run_chunked_quad_extract(input_mesh, feat_path, crossfield_txt, output_obj, args,
                             size_field_path=None):
    mesh = trimesh.load_mesh(input_mesh, process=False)
    chunks, strategy = plan_extraction_chunks(mesh, feat_path, args)

    if len(chunks) <= 1:
        run_quad_extract(input_mesh, crossfield_txt, output_obj, args)
        return {
            "strategy": strategy,
            "n_chunks": 1,
            "chunk_face_counts": [int(len(chunks[0]))],
            "chunk_dir": None,
        }

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
    for chunk_idx, face_ids in enumerate(chunks):
        chunk_name = f"chunk_{chunk_idx:03d}"
        chunk_mesh = os.path.join(chunk_input_dir, f"{chunk_name}.obj")
        chunk_crossfield = os.path.join(chunk_input_dir, f"{chunk_name}_crossfield.txt")
        chunk_output = os.path.join(chunk_output_dir, f"{chunk_name}_quad.obj")

        write_face_subset_obj(mesh, face_ids, chunk_mesh)
        write_subset_crossfield(crossfield_rows, face_ids, chunk_crossfield)
        run_quad_extract(chunk_mesh, chunk_crossfield, chunk_output, args)
        chunk_outputs.append(chunk_output)

    merge_quad_chunks(chunk_outputs, output_obj)
    return {
        "strategy": strategy,
        "n_chunks": int(len(chunks)),
        "chunk_face_counts": [int(len(chunk)) for chunk in chunks],
        "chunk_dir": chunk_root,
    }


def run_quad_extract(input_mesh, crossfield_txt, output_obj, args, size_field_path=None):
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
        "--timeout", "600",
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
    if not args.disable_extract_retry:
        cmd.append("--retry")
    if args.catmull_clark_iters > 0:
        cmd.extend(["--catmull_clark_iters", str(args.catmull_clark_iters)])
    if args.target_quad_ratio is not None:
        cmd.extend(["--target_quad_ratio", str(args.target_quad_ratio)])
    if args.max_catmull_clark_iters is not None:
        cmd.extend(["--max_catmull_clark_iters", str(args.max_catmull_clark_iters)])

    print(f"\n[Stage 3] Quad extraction")
    print(f"  script      : {EXTRACT_QUAD_PY}")
    print(f"  python      : {sys.executable}")
    print(f"  input mesh  : {input_mesh}")
    print(f"  cross field : {crossfield_txt}")
    print(f"  output      : {output_obj}")
    print(f"  gradient_size: {args.gradient_size}\n")
    if size_field_path is not None or args.size_field_path is not None:
        print("  warning     : deprecated size-field options are ignored")
    if not args.disable_semantic_size_field:
        print("  warning     : semantic size-field generation is disabled for pure MIQ -> libQEx extraction")
    print()

    print(f"  auto_sweep  : {not args.disable_auto_sweep}")
    print(f"  retry       : {not args.disable_extract_retry}")
    print(f"  min_quads   : {'default' if args.min_quads is None else args.min_quads}")
    print(f"  max_ad      : {args.max_ad}")
    print(f"  min_jr      : {args.min_jr}\n")
    print(f"  cc_iters    : {args.catmull_clark_iters}")
    print(f"  target_ratio: {args.target_quad_ratio}")
    print(f"  cc_max_auto : {args.max_catmull_clark_iters}\n")

    env = os.environ.copy()
    subprocess.run(cmd, env=env, check=True)
    return output_obj


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

    # Verify features
    print("\n[Check] Verifying feature files ...")
    feat_map = {}
    for basename, orig_path, n_faces in info_list:
        if feat_override:
            fp = feat_override
        else:
            fp = find_feature_file(feat_dir, basename)
        verify_features(fp, n_faces, basename)
        feat_map[basename] = (orig_path, fp)

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
        logdir = run_neurcross(orig_path, fp, basename, args)
        results[basename] = {
            "input": orig_path,
            "features": fp,
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
                extract_meta = run_chunked_quad_extract(
                    input_mesh_obj,
                    r["features"],
                    cf_txt,
                    output_obj,
                    args
                )
                results[basename]["quad_mesh"] = output_obj
                results[basename]["extract_strategy"] = extract_meta["strategy"]
                results[basename]["extract_chunks"] = extract_meta["n_chunks"]
                results[basename]["extract_chunk_faces"] = extract_meta["chunk_face_counts"]
                if extract_meta["chunk_dir"] is not None:
                    results[basename]["extract_chunk_dir"] = extract_meta["chunk_dir"]
            except subprocess.CalledProcessError as e:
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
    print(f"  partfield_input/        Staged mesh files for PartField")
    print(f"  partfield_features/     Per-face features (.npy) + PCA vis (.ply)")
    print(f"  neurcross_logs/         NeurCross training results")
    print(f"  quad_meshes/*_chunks/   Per-chunk extraction intermediates when chunking is used")
    print(f"  quad_meshes/            Extracted quad meshes (.obj)")
    print()
    for name, r in results.items():
        cross_field_dir = os.path.join(r["logdir"], "save_crossField")
        print(f"  [{name}]")
        print(f"    features   : {os.path.relpath(r['features'], abs_output)}")
        print(f"    cross field: {os.path.relpath(cross_field_dir, abs_output)}/")
        strategy = r.get("extract_strategy")
        if strategy:
            print(f"    extraction : {strategy} ({r.get('extract_chunks', 1)} chunks)")
        qm = r.get("quad_mesh")
        if qm and os.path.isfile(qm):
            print(f"    quad mesh  : {os.path.relpath(qm, abs_output)}")
        print()


if __name__ == "__main__":
    main()
