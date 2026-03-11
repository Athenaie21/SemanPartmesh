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
    python run_pipeline.py --input_dir input/
    python run_pipeline.py --input_mesh input/armadillo.obj
"""

import os
import sys
import shutil
import argparse
import subprocess

import numpy as np
import trimesh

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARTFIELD_DIR = os.path.join(PROJECT_ROOT, "PartField")
NEURCROSS_DIR = os.path.join(PROJECT_ROOT, "NeurCross")
QUAD_EXTRACT_BIN = os.path.join(
    PROJECT_ROOT, "quad_extract", "build", "extract_quad_mesh")

PARTFIELD_NATIVE_EXTS = {".obj", ".glb", ".off"}
ALL_MESH_EXTS = {".obj", ".glb", ".off", ".ply", ".stl"}

DEFAULT_MAX_FACES = 30000


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
                   default=[7e3, 6e2, 10, 5e1, 30, 3, 20],
                   help="[sdf, inter, theta_hess, eikonal, theta_neigh, "
                        "morse, semantic]")

    # -- Quad Extraction (Stage 3) -----------------------------------------
    p.add_argument("--gradient_size", type=float, default=30.0,
                   help="MIQ gradient size (controls quad density)")
    p.add_argument("--skip_extract", action="store_true",
                   help="Skip Stage 3 (quad mesh extraction)")

    # -- General ------------------------------------------------------------
    p.add_argument("--output_dir", default="pipeline_output",
                   help="Root directory for all outputs")
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


def run_quad_extract(input_mesh, crossfield_txt, output_obj, args,
                     gradient_size=None, timeout=600):
    """Call the C++ extract_quad_mesh tool.

    Returns output_obj on success, None on failure.
    """
    if not os.path.isfile(QUAD_EXTRACT_BIN):
        sys.exit(f"Quad extraction binary not found: {QUAD_EXTRACT_BIN}\n"
                 f"  Build it with: cd quad_extract/build && cmake .. && make")

    gs = gradient_size if gradient_size is not None else args.gradient_size

    cmd = [
        QUAD_EXTRACT_BIN,
        os.path.abspath(input_mesh),
        os.path.abspath(crossfield_txt),
        os.path.abspath(output_obj),
        str(gs),
    ]

    print(f"\n[Stage 3] Quad extraction")
    print(f"  binary      : {QUAD_EXTRACT_BIN}")
    print(f"  input mesh  : {input_mesh}")
    print(f"  cross field : {crossfield_txt}")
    print(f"  output      : {output_obj}")
    print(f"  gradient_size: {gs}  timeout: {timeout}s\n")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        "/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", ""))
    subprocess.run(cmd, env=env, check=True, timeout=timeout)
    return output_obj


def run_quad_extract_with_retry(input_mesh, crossfield_dir, output_obj, args):
    """Try quad extraction with multiple gradient sizes and cross-field
    iterations, returning the first successful output path or None."""
    gradient_sizes = [args.gradient_size, 50.0, 15.0, 80.0]
    seen = set()
    gradient_sizes = [g for g in gradient_sizes
                      if not (g in seen or seen.add(g))]

    import glob as _glob
    cf_files = sorted(
        _glob.glob(os.path.join(crossfield_dir, "*_iter_*.txt")),
        key=lambda f: int(
            os.path.splitext(os.path.basename(f))[0].rsplit("_iter_", 1)[1]),
        reverse=True,
    )
    if not cf_files:
        print(f"  WARNING: No cross field found in {crossfield_dir}")
        return None

    for cf_txt in cf_files[:3]:
        cf_label = os.path.basename(cf_txt)
        for gs in gradient_sizes:
            print(f"\n  >> Trying cross_field={cf_label}  gradient_size={gs}")
            try:
                run_quad_extract(input_mesh, cf_txt, output_obj, args,
                                 gradient_size=gs, timeout=600)
                if os.path.isfile(output_obj) and os.path.getsize(output_obj) > 0:
                    print(f"  >> SUCCESS with {cf_label} gs={gs}")
                    return output_obj
            except subprocess.TimeoutExpired:
                print(f"  >> TIMEOUT with {cf_label} gs={gs}")
            except subprocess.CalledProcessError as e:
                print(f"  >> FAILED with {cf_label} gs={gs}: {e}")

    print(f"  WARNING: All extraction attempts failed.")
    return None


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

            input_mesh_obj = r["input"]
            ext = os.path.splitext(input_mesh_obj)[1].lower()
            if ext != ".obj":
                obj_path = os.path.join(quad_output_dir,
                                        f"{basename}_input.obj")
                mesh = trimesh.load_mesh(input_mesh_obj, process=False)
                mesh.export(obj_path)
                input_mesh_obj = obj_path

            output_obj = os.path.join(quad_output_dir, f"{basename}_quad.obj")
            result_path = run_quad_extract_with_retry(
                input_mesh_obj, cross_field_dir, output_obj, args)
            results[basename]["quad_mesh"] = result_path
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
    print(f"  quad_meshes/            Extracted quad meshes (.obj)")
    print()
    for name, r in results.items():
        cross_field_dir = os.path.join(r["logdir"], "save_crossField")
        print(f"  [{name}]")
        print(f"    features   : {os.path.relpath(r['features'], abs_output)}")
        print(f"    cross field: {os.path.relpath(cross_field_dir, abs_output)}/")
        qm = r.get("quad_mesh")
        if qm and os.path.isfile(qm):
            print(f"    quad mesh  : {os.path.relpath(qm, abs_output)}")
        print()


if __name__ == "__main__":
    main()
