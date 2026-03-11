#!/usr/bin/env python
"""
Standalone quad mesh extraction from NeurCross cross-field output.

Reads a triangle mesh (.obj) and a NeurCross cross-field .txt file,
then runs MIQ parameterization + libQEx to produce a quad mesh.

Usage
-----
    # Single cross-field file:
    python extract_quad.py --mesh input/cheburashka.obj \
        --crossfield pipeline_output/neurcross_logs/cheburashka/save_crossField/cheburashka_iter_9999.txt \
        --output output_quad.obj

    # Auto-detect latest cross-field from a NeurCross log directory:
    python extract_quad.py --mesh input/cheburashka.obj \
        --crossfield_dir pipeline_output/neurcross_logs/cheburashka/save_crossField/

    # Batch mode — process all meshes under a directory:
    python extract_quad.py --mesh_dir input/ \
        --crossfield_root pipeline_output/neurcross_logs/ \
        --output_dir pipeline_output/quad_meshes/
"""

import os
import sys
import glob
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
QUAD_EXTRACT_BIN = os.path.join(
    PROJECT_ROOT, "quad_extract", "build", "extract_quad_mesh")

ALL_MESH_EXTS = {".obj", ".glb", ".off", ".ply", ".stl"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone quad mesh extraction (MIQ + libQEx)")

    mesh_grp = p.add_mutually_exclusive_group(required=True)
    mesh_grp.add_argument("--mesh", help="Path to a single input triangle mesh (.obj)")
    mesh_grp.add_argument("--mesh_dir", help="Directory of triangle meshes for batch mode")

    cf_grp = p.add_mutually_exclusive_group()
    cf_grp.add_argument("--crossfield", help="Path to a single cross-field .txt file")
    cf_grp.add_argument(
        "--crossfield_dir",
        help="Directory containing *_iter_*.txt files; the latest iteration is used")
    cf_grp.add_argument(
        "--crossfield_root",
        help="Root directory of NeurCross logs (batch mode). "
             "Expects <root>/<mesh_name>/save_crossField/ structure")

    p.add_argument("--output", default=None,
                   help="Output quad mesh path (single-mesh mode)")
    p.add_argument("--output_dir", default=None,
                   help="Output directory for quad meshes (batch mode)")
    p.add_argument("--gradient_size", type=float, default=30.0,
                   help="MIQ gradient size — controls quad density (default: 30.0)")
    p.add_argument("--timeout", type=int, default=600,
                   help="Timeout per extraction attempt in seconds (default: 600)")
    p.add_argument("--retry", action="store_true",
                   help="Auto-retry with multiple gradient sizes and iterations on failure")

    return p.parse_args()


def find_latest_crossfield(crossfield_dir):
    """Find the cross-field txt with the highest iteration number."""
    files = glob.glob(os.path.join(crossfield_dir, "*_iter_*.txt"))
    if not files:
        return None

    def iter_num(f):
        base = os.path.splitext(os.path.basename(f))[0]
        return int(base.rsplit("_iter_", 1)[1])
    return max(files, key=iter_num)


def run_extract(mesh_path, crossfield_path, output_path,
                gradient_size=30.0, timeout=600):
    """Run the C++ extract_quad_mesh binary once.

    Returns True on success, False on failure.
    """
    cmd = [
        QUAD_EXTRACT_BIN,
        os.path.abspath(mesh_path),
        os.path.abspath(crossfield_path),
        os.path.abspath(output_path),
        str(gradient_size),
    ]

    print(f"  binary       : {QUAD_EXTRACT_BIN}")
    print(f"  input mesh   : {mesh_path}")
    print(f"  cross field  : {crossfield_path}")
    print(f"  output       : {output_path}")
    print(f"  gradient_size: {gradient_size}  timeout: {timeout}s\n")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        "/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", ""))

    try:
        subprocess.run(cmd, env=env, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {e}")
        return False

    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        return True
    return False


def extract_single(mesh_path, crossfield_path, output_path,
                   gradient_size, timeout, retry):
    """Extract a quad mesh, optionally retrying with different parameters."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if not retry:
        print(f"\n[Quad Extraction]")
        ok = run_extract(mesh_path, crossfield_path, output_path,
                         gradient_size, timeout)
        if ok:
            print(f"SUCCESS -> {output_path}")
        else:
            print(f"FAILED for {mesh_path}")
        return ok

    cf_dir = os.path.dirname(crossfield_path)
    cf_files = sorted(
        glob.glob(os.path.join(cf_dir, "*_iter_*.txt")),
        key=lambda f: int(
            os.path.splitext(os.path.basename(f))[0].rsplit("_iter_", 1)[1]),
        reverse=True,
    )
    if not cf_files:
        cf_files = [crossfield_path]

    gradient_sizes = [gradient_size, 50.0, 15.0, 80.0]
    seen = set()
    gradient_sizes = [g for g in gradient_sizes
                      if not (g in seen or seen.add(g))]

    for cf_txt in cf_files[:3]:
        cf_label = os.path.basename(cf_txt)
        for gs in gradient_sizes:
            print(f"\n[Quad Extraction] Trying {cf_label}  gs={gs}")
            ok = run_extract(mesh_path, cf_txt, output_path, gs, timeout)
            if ok:
                print(f"SUCCESS -> {output_path}")
                return True

    print(f"FAILED: all attempts exhausted for {mesh_path}")
    return False


def main():
    args = parse_args()

    if not os.path.isfile(QUAD_EXTRACT_BIN):
        sys.exit(
            f"Quad extraction binary not found: {QUAD_EXTRACT_BIN}\n"
            f"  Build it with:\n"
            f"    cd libQEx/build && cmake .. -DCMAKE_CXX_FLAGS='-DNDEBUG' && make\n"
            f"    cd quad_extract/build && cmake .. -DCMAKE_CXX_FLAGS='-DNDEBUG -O2' && make")

    # --- Single mesh mode ---
    if args.mesh:
        if not os.path.isfile(args.mesh):
            sys.exit(f"Mesh not found: {args.mesh}")

        if args.crossfield:
            cf_path = args.crossfield
        elif args.crossfield_dir:
            cf_path = find_latest_crossfield(args.crossfield_dir)
            if cf_path is None:
                sys.exit(f"No *_iter_*.txt files found in {args.crossfield_dir}")
            print(f"Auto-selected cross field: {cf_path}")
        elif args.crossfield_root:
            basename = os.path.splitext(os.path.basename(args.mesh))[0]
            cf_dir = os.path.join(args.crossfield_root, basename, "save_crossField")
            cf_path = find_latest_crossfield(cf_dir)
            if cf_path is None:
                sys.exit(f"No cross-field files in {cf_dir}")
            print(f"Auto-selected cross field: {cf_path}")
        else:
            sys.exit("Provide --crossfield, --crossfield_dir, or --crossfield_root")

        if not os.path.isfile(cf_path):
            sys.exit(f"Cross-field file not found: {cf_path}")

        basename = os.path.splitext(os.path.basename(args.mesh))[0]
        output = args.output or f"{basename}_quad.obj"

        ok = extract_single(args.mesh, cf_path, output,
                            args.gradient_size, args.timeout, args.retry)
        sys.exit(0 if ok else 1)

    # --- Batch mode ---
    mesh_dir = os.path.abspath(args.mesh_dir)
    if not os.path.isdir(mesh_dir):
        sys.exit(f"Mesh directory not found: {mesh_dir}")

    if not args.crossfield_root:
        sys.exit("Batch mode (--mesh_dir) requires --crossfield_root")

    output_dir = args.output_dir or "quad_meshes_output"
    os.makedirs(output_dir, exist_ok=True)

    mesh_files = sorted(
        f for f in os.listdir(mesh_dir)
        if os.path.splitext(f)[1].lower() in ALL_MESH_EXTS
    )
    if not mesh_files:
        sys.exit(f"No mesh files found in {mesh_dir}")

    n = len(mesh_files)
    success, failed = 0, 0

    for idx, mf in enumerate(mesh_files, 1):
        basename = os.path.splitext(mf)[0]
        mesh_path = os.path.join(mesh_dir, mf)

        cf_dir = os.path.join(
            os.path.abspath(args.crossfield_root), basename, "save_crossField")
        cf_path = find_latest_crossfield(cf_dir)

        print(f"\n{'='*60}")
        print(f"  [{idx}/{n}] {basename}")
        print(f"{'='*60}")

        if cf_path is None:
            print(f"  SKIP: no cross-field found in {cf_dir}")
            failed += 1
            continue

        output_path = os.path.join(output_dir, f"{basename}_quad.obj")
        ok = extract_single(mesh_path, cf_path, output_path,
                            args.gradient_size, args.timeout, args.retry)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Done: {success} succeeded, {failed} failed out of {n}")
    print(f"  Output directory: {os.path.abspath(output_dir)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
