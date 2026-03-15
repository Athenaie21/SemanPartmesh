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
import json
import shutil

import trimesh

from eval.angle_distortion import compute_angle_distortion, _load_quad_faces_from_obj
from eval.jacobian_ratio import compute_jacobian_ratio

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
    p.add_argument("--size_field", default=None,
                   help="Optional per-face or per-vertex local size/density text file.")
    p.add_argument("--size_field_strength", type=float, default=0.25,
                   help="Strength of local size adaptation in [0, +inf). "
                        "0 disables local adaptation while preserving the file load path.")
    p.add_argument("--size_field_smooth_iters", type=int, default=6,
                   help="Number of face-neighbor smoothing iterations for the local size field.")
    p.add_argument("--size_field_relax", action="store_true",
                   help="If size-field extraction fails or times out, automatically retry with weaker local adaptation.")
    p.add_argument("--timeout", type=int, default=600,
                   help="Timeout per extraction attempt in seconds (default: 600)")
    p.add_argument("--retry", action="store_true",
                   help="Auto-retry with multiple gradient sizes and iterations on failure")
    p.add_argument("--auto_sweep", action="store_true",
                   help="Sweep multiple gradient sizes, evaluate each result, and keep the best one")
    p.add_argument("--sweep_values", nargs="+", type=float, default=None,
                   help="Explicit gradient_size values for auto sweep "
                        "(example: --sweep_values 8 12 16 24 30 40 60)")
    p.add_argument("--keep_sweep_outputs", action="store_true",
                   help="Keep all intermediate sweep OBJ files instead of only the best one")
    p.add_argument("--summary_json", default=None,
                   help="Optional JSON path to save auto-sweep metrics and ranking summary")
    p.add_argument("--min_quads", type=int, default=None,
                   help="Hard lower bound for acceptable quad count in auto-sweep mode. "
                        "Default: max(100, 5%% of input triangle faces)")
    p.add_argument("--max_ad", type=float, default=15.0,
                   help="Hard upper bound for acceptable mean angle distortion in degrees "
                        "during auto sweep (default: 15.0)")
    p.add_argument("--min_jr", type=float, default=0.15,
                   help="Hard lower bound for acceptable minimum Jacobian Ratio "
                        "during auto sweep (default: 0.15)")

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


def _run_extract_once(mesh_path, crossfield_path, output_path,
                      gradient_size=30.0, timeout=600, size_field_path=None,
                      size_field_strength=0.25, size_field_smooth_iters=6):
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
    if size_field_path is not None:
        cmd.extend([
            f"--size_field={os.path.abspath(size_field_path)}",
            f"--size_strength={size_field_strength}",
            f"--size_smooth_iters={size_field_smooth_iters}",
        ])

    print(f"  binary       : {QUAD_EXTRACT_BIN}")
    print(f"  input mesh   : {mesh_path}")
    print(f"  cross field  : {crossfield_path}")
    print(f"  output       : {output_path}")
    print(f"  gradient_size: {gradient_size}  timeout: {timeout}s\n")
    if size_field_path is not None:
        print(f"  size_field   : {size_field_path}")
        print(f"  size_strength: {size_field_strength}")
        print(f"  smooth_iters : {size_field_smooth_iters}\n")

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


def build_size_field_attempts(size_field_path, size_field_strength,
                              size_field_smooth_iters, relax_enabled):
    if size_field_path is None:
        return [(None, 0.0, 0)]

    attempts = [(size_field_path, size_field_strength, size_field_smooth_iters)]
    if not relax_enabled:
        return attempts

    relaxed = [
        (size_field_path, min(size_field_strength, 0.15), max(size_field_smooth_iters, 8)),
        (size_field_path, min(size_field_strength, 0.08), max(size_field_smooth_iters, 10)),
        (None, 0.0, 0),
    ]
    seen = set()
    merged = []
    for path, strength, smooth_iters in attempts + relaxed:
        key = (
            None if path is None else os.path.abspath(path),
            round(float(strength), 6),
            int(smooth_iters),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append((path, float(strength), int(smooth_iters)))
    return merged


def run_extract(mesh_path, crossfield_path, output_path,
                gradient_size=30.0, timeout=600, size_field_path=None,
                size_field_strength=0.25, size_field_smooth_iters=6,
                size_field_relax=False):
    attempts = build_size_field_attempts(
        size_field_path,
        size_field_strength,
        size_field_smooth_iters,
        relax_enabled=size_field_relax,
    )

    for idx, (path, strength, smooth_iters) in enumerate(attempts, 1):
        if os.path.isfile(output_path):
            os.remove(output_path)

        if path is None:
            print(f"  local size attempt {idx}/{len(attempts)}: disabled")
        else:
            print(
                f"  local size attempt {idx}/{len(attempts)}: "
                f"strength={strength}, smooth_iters={smooth_iters}"
            )

        ok = _run_extract_once(
            mesh_path,
            crossfield_path,
            output_path,
            gradient_size=gradient_size,
            timeout=timeout,
            size_field_path=path,
            size_field_strength=strength,
            size_field_smooth_iters=smooth_iters,
        )
        if ok:
            return True

    return False


def sanitize_float_tag(value):
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def default_sweep_values(base_value):
    """Return a compact set of sweep candidates around *base_value*."""
    multipliers = [0.25, 0.4, 0.55, 0.75, 1.0, 1.35, 1.8, 2.4]
    values = []
    seen = set()
    for m in multipliers:
        v = round(base_value * m, 4)
        if v <= 0:
            continue
        key = round(v, 6)
        if key in seen:
            continue
        seen.add(key)
        values.append(v)
    return values


def evaluate_quad_mesh(quad_path):
    """Return basic geometric metrics for a quad OBJ."""
    quad_faces = _load_quad_faces_from_obj(quad_path)
    if quad_faces is None or len(quad_faces) == 0:
        raise ValueError(f"No valid quad faces found in {quad_path}")

    mesh = trimesh.load(quad_path, process=False)
    vertices = mesh.vertices
    ad = compute_angle_distortion(vertices, quad_faces)
    jr = compute_jacobian_ratio(vertices, quad_faces)

    return {
        "n_quads": int(len(quad_faces)),
        "n_verts": int(len(vertices)),
        "AD_mean_deg": float(ad["mean_deviation_deg"]),
        "AD_max_deg": float(ad["max_deviation_deg"]),
        "JR_mean": float(jr["mean_jr"]),
        "JR_min": float(jr["min_jr"]),
        "JR_std": float(jr["std_jr"]),
        "SJ_mean": float(jr["mean_scaled_jacobian"]),
    }


def count_triangle_faces(mesh_path):
    mesh = trimesh.load(mesh_path, process=False)
    return int(len(mesh.faces))


def build_candidate_output_path(output_path, gradient_size):
    base, ext = os.path.splitext(output_path)
    return f"{base}_gs{sanitize_float_tag(gradient_size)}{ext}"


def get_min_quads_threshold(mesh_path, min_quads):
    if min_quads is not None:
        return min_quads
    tri_faces = count_triangle_faces(mesh_path)
    return max(100, int(round(0.05 * tri_faces)))


def candidate_violation(metrics, min_quads, max_ad, min_jr):
    quad_deficit = max(0, min_quads - metrics["n_quads"])
    ad_excess = max(0.0, metrics["AD_mean_deg"] - max_ad)
    jr_deficit = max(0.0, min_jr - metrics["JR_min"])
    return {
        "quad_deficit": quad_deficit,
        "ad_excess": ad_excess,
        "jr_deficit": jr_deficit,
        "num_failed_constraints": int(quad_deficit > 0) + int(ad_excess > 0.0) + int(jr_deficit > 0.0),
    }


def rank_candidate(metrics, violation):
    """Lower tuple is better."""
    return (
        violation["num_failed_constraints"],
        violation["quad_deficit"],
        violation["ad_excess"],
        violation["jr_deficit"],
        -metrics["n_quads"],
        metrics["AD_mean_deg"],
        -metrics["JR_mean"],
        -metrics["SJ_mean"],
    )


def auto_sweep_single(mesh_path, crossfield_path, output_path, args):
    """Sweep gradient_size values, evaluate each output, and keep the best."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    sweep_values = args.sweep_values or default_sweep_values(args.gradient_size)
    min_quads = get_min_quads_threshold(mesh_path, args.min_quads)

    print("\n[Auto Sweep]")
    print(f"  mesh        : {mesh_path}")
    print(f"  cross field : {crossfield_path}")
    print(f"  output      : {output_path}")
    print(f"  sweep_values: {sweep_values}")
    print(f"  thresholds  : min_quads={min_quads}, max_ad={args.max_ad}, min_jr={args.min_jr}\n")

    candidates = []
    for gs in sweep_values:
        candidate_output = build_candidate_output_path(output_path, gs)
        print(f"[Auto Sweep] Trying gradient_size={gs}")
        ok = run_extract(
            mesh_path,
            crossfield_path,
            candidate_output,
            gs,
            args.timeout,
            size_field_path=args.size_field,
            size_field_strength=args.size_field_strength,
            size_field_smooth_iters=args.size_field_smooth_iters,
            size_field_relax=args.size_field_relax,
        )
        if not ok:
            print(f"  FAILED: extraction failed for gs={gs}\n")
            continue

        try:
            metrics = evaluate_quad_mesh(candidate_output)
        except Exception as exc:
            print(f"  FAILED: evaluation failed for gs={gs}: {exc}\n")
            continue

        violation = candidate_violation(metrics, min_quads, args.max_ad, args.min_jr)
        metrics["gradient_size"] = float(gs)
        metrics["output_path"] = os.path.abspath(candidate_output)
        metrics["constraint_violation"] = violation
        metrics["is_valid"] = violation["num_failed_constraints"] == 0
        metrics["rank_key"] = list(rank_candidate(metrics, violation))
        candidates.append(metrics)

        status = "valid" if metrics["is_valid"] else "constraint-violating"
        print(
            f"  RESULT [{status}] "
            f"n_quads={metrics['n_quads']} "
            f"AD_mean={metrics['AD_mean_deg']:.3f}deg "
            f"JR_min={metrics['JR_min']:.4f} "
            f"SJ_mean={metrics['SJ_mean']:.4f}\n"
        )

    if not candidates:
        print("FAILED: no valid sweep candidate produced an evaluable quad mesh")
        return False

    candidates.sort(key=lambda m: tuple(m["rank_key"]))
    best = candidates[0]
    shutil.copyfile(best["output_path"], output_path)

    if not args.keep_sweep_outputs:
        best_abs = os.path.abspath(output_path)
        for candidate in candidates:
            path = candidate["output_path"]
            if os.path.abspath(path) != best_abs and os.path.isfile(path):
                os.remove(path)

    summary = {
        "mesh_path": os.path.abspath(mesh_path),
        "crossfield_path": os.path.abspath(crossfield_path),
        "output_path": os.path.abspath(output_path),
        "selection_rule": {
            "priority": [
                "fewest failed constraints",
                "smallest quad deficit",
                "smallest angle-distortion excess",
                "smallest Jacobian deficit",
                "largest quad count",
                "lowest angle distortion",
                "highest mean Jacobian ratio",
                "highest mean scaled Jacobian",
            ],
            "min_quads": min_quads,
            "max_ad": args.max_ad,
            "min_jr": args.min_jr,
        },
        "best": best,
        "candidates": candidates,
    }

    summary_path = args.summary_json
    if summary_path is None:
        base, _ = os.path.splitext(output_path)
        summary_path = f"{base}_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("[Auto Sweep] Best candidate selected")
    print(f"  gradient_size: {best['gradient_size']}")
    print(f"  n_quads      : {best['n_quads']}")
    print(f"  AD_mean_deg  : {best['AD_mean_deg']:.3f}")
    print(f"  JR_min       : {best['JR_min']:.4f}")
    print(f"  saved output : {output_path}")
    print(f"  saved summary: {summary_path}")
    return True


def extract_single(mesh_path, crossfield_path, output_path,
                   gradient_size, timeout, retry, args=None):
    """Extract a quad mesh, optionally retrying with different parameters."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if args is not None and args.auto_sweep:
        return auto_sweep_single(mesh_path, crossfield_path, output_path, args)

    if not retry:
        print(f"\n[Quad Extraction]")
        ok = run_extract(mesh_path, crossfield_path, output_path,
                         gradient_size, timeout,
                         size_field_path=args.size_field if args is not None else None,
                         size_field_strength=args.size_field_strength if args is not None else 0.25,
                         size_field_smooth_iters=args.size_field_smooth_iters if args is not None else 6,
                         size_field_relax=args.size_field_relax if args is not None else False)
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
            ok = run_extract(
                mesh_path,
                cf_txt,
                output_path,
                gs,
                timeout,
                size_field_path=args.size_field if args is not None else None,
                size_field_strength=args.size_field_strength if args is not None else 0.25,
                size_field_smooth_iters=args.size_field_smooth_iters if args is not None else 6,
                size_field_relax=args.size_field_relax if args is not None else False,
            )
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
                            args.gradient_size, args.timeout, args.retry, args=args)
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
                            args.gradient_size, args.timeout, args.retry, args=args)
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
