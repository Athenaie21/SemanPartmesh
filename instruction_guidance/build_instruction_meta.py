#!/usr/bin/env python
import argparse
import os

from instruction_guidance.metadata import build_instruction_metadata
from instruction_guidance.metadata import derive_mesh_basename
from instruction_guidance.metadata import infer_dataset_paths
from instruction_guidance.metadata import save_instruction_metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build per-triangle instruction metadata from Fusion360 segmentation files.")
    parser.add_argument("--mesh", required=True, help="Path to the mesh used for NeurCross training/extraction")
    parser.add_argument("--dataset_root", required=True,
                        help="Fusion360 segmentation dataset root containing meshes/ and timeline_info/")
    parser.add_argument("--output", required=True, help="Output .npz metadata path")
    parser.add_argument("--mesh_basename", default=None,
                        help="Optional basename override for dataset lookup")
    parser.add_argument("--fidx", default=None, help="Optional explicit .fidx path")
    parser.add_argument("--timeline", default=None, help="Optional explicit timeline_info json path")
    parser.add_argument("--seg", default=None, help="Optional explicit .seg path")
    return parser.parse_args()


def main():
    args = parse_args()
    mesh_path = os.path.abspath(args.mesh)
    if not os.path.isfile(mesh_path):
        raise SystemExit(f"Mesh not found: {mesh_path}")

    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    mesh_basename = args.mesh_basename or derive_mesh_basename(mesh_path)
    inferred = infer_dataset_paths(dataset_root, mesh_basename)

    fidx_path = os.path.abspath(args.fidx) if args.fidx else inferred["fidx"]
    timeline_path = os.path.abspath(args.timeline) if args.timeline else inferred["timeline"]
    seg_path = os.path.abspath(args.seg) if args.seg else inferred["seg"]

    missing = [p for p in [fidx_path, timeline_path] if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"Missing required instruction metadata inputs: {missing}")
    if not os.path.isfile(seg_path):
        seg_path = None

    metadata = build_instruction_metadata(
        mesh_path=mesh_path,
        fidx_path=fidx_path,
        timeline_path=timeline_path,
        seg_path=seg_path,
    )
    output_path = save_instruction_metadata(os.path.abspath(args.output), metadata)
    print(f"Saved instruction metadata to: {output_path}")


if __name__ == "__main__":
    main()
