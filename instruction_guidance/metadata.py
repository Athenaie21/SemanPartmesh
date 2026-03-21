import json
import os

import numpy as np
import trimesh


FEATURE_TYPE_TO_ID = {
    "UnknownFeature": 0,
    "ExtrudeFeature": 1,
    "ChamferFeature": 2,
    "FilletFeature": 3,
    "RevolveFeature": 4,
}

FEATURE_OPERATION_TO_ID = {
    "UnknownOperation": 0,
    "NewBodyFeatureOperation": 1,
    "JoinFeatureOperation": 2,
    "CutFeatureOperation": 3,
    "IntersectFeatureOperation": 4,
}

LOCATION_TO_ID = {
    "UnknownLocation": 0,
    "SideFace": 1,
    "StartFace": 2,
    "EndFace": 3,
}


def derive_mesh_basename(mesh_path):
    return os.path.splitext(os.path.basename(mesh_path))[0]


def load_face_index_file(fidx_path):
    values = np.loadtxt(fidx_path, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    return values


def load_segmentation_file(seg_path):
    values = np.loadtxt(seg_path, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    return values


def load_timeline_info(timeline_path):
    with open(timeline_path, "r") as f:
        return json.load(f)


def infer_dataset_paths(dataset_root, mesh_basename):
    mesh_dir = os.path.join(dataset_root, "meshes")
    timeline_dir = os.path.join(dataset_root, "timeline_info")
    return {
        "mesh_obj": os.path.join(mesh_dir, f"{mesh_basename}.obj"),
        "fidx": os.path.join(mesh_dir, f"{mesh_basename}.fidx"),
        "seg": os.path.join(mesh_dir, f"{mesh_basename}.seg"),
        "timeline": os.path.join(timeline_dir, f"{mesh_basename}.json"),
    }


def _normalize_timeline_indices(values):
    values = np.asarray(values, dtype=np.float32)
    valid = values >= 0
    normalized = np.full(values.shape, -1.0, dtype=np.float32)
    if not np.any(valid):
        return normalized
    valid_values = values[valid]
    lo = float(valid_values.min())
    hi = float(valid_values.max())
    if hi - lo < 1e-8:
        normalized[valid] = 0.0
    else:
        normalized[valid] = (valid_values - lo) / (hi - lo)
    return normalized


def build_instance_boundary_mask(mesh, feature_instance_id):
    feature_instance_id = np.asarray(feature_instance_id, dtype=np.int64)
    mask = np.zeros(len(feature_instance_id), dtype=np.uint8)
    for face_a, face_b in np.asarray(mesh.face_adjacency, dtype=np.int64):
        if feature_instance_id[face_a] != feature_instance_id[face_b]:
            mask[face_a] = 1
            mask[face_b] = 1
    return mask


def build_instruction_metadata(
        mesh_path,
        fidx_path,
        timeline_path,
        seg_path=None):
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a triangle mesh at {mesh_path}")

    fidx = load_face_index_file(fidx_path)
    if len(fidx) != len(mesh.faces):
        raise ValueError(
            f"fidx length ({len(fidx)}) does not match triangle face count ({len(mesh.faces)}).")

    seg_labels = None
    if seg_path is not None and os.path.isfile(seg_path):
        seg_labels = load_segmentation_file(seg_path)
        if len(seg_labels) != len(mesh.faces):
            raise ValueError(
                f"seg length ({len(seg_labels)}) does not match triangle face count ({len(mesh.faces)}).")

    timeline = load_timeline_info(timeline_path)
    brep_faces = timeline.get("faces", [])
    feature_table = timeline.get("features", {})
    if not brep_faces:
        raise ValueError(f"No faces found in timeline info: {timeline_path}")

    max_face_index = int(fidx.max()) if len(fidx) > 0 else -1
    if max_face_index >= len(brep_faces):
        raise ValueError(
            f"fidx references B-Rep face {max_face_index}, but timeline info only has {len(brep_faces)} faces.")

    feature_uuid_per_tri = []
    location_per_tri = []
    feature_type_per_tri = []
    feature_operation_per_tri = []
    timeline_index_per_tri = []

    for brep_face_idx in fidx:
        face_info = brep_faces[int(brep_face_idx)]
        feature_uuid = face_info.get("feature", "")
        feature_meta = feature_table.get(feature_uuid, {})

        feature_uuid_per_tri.append(feature_uuid)
        location_per_tri.append(face_info.get("location_in_feature", "UnknownLocation"))
        feature_type_per_tri.append(feature_meta.get("type", "UnknownFeature"))
        feature_operation_per_tri.append(feature_meta.get("operation", "UnknownOperation"))
        timeline_index_per_tri.append(feature_meta.get("timeline_index", -1))

    unique_feature_uuids = []
    feature_uuid_to_instance_id = {}
    for uuid in feature_uuid_per_tri:
        if uuid not in feature_uuid_to_instance_id:
            feature_uuid_to_instance_id[uuid] = len(unique_feature_uuids)
            unique_feature_uuids.append(uuid)

    feature_instance_id = np.asarray(
        [feature_uuid_to_instance_id[uuid] for uuid in feature_uuid_per_tri],
        dtype=np.int64
    )
    feature_type_id = np.asarray(
        [FEATURE_TYPE_TO_ID.get(name, 0) for name in feature_type_per_tri],
        dtype=np.int64
    )
    feature_operation_id = np.asarray(
        [FEATURE_OPERATION_TO_ID.get(name, 0) for name in feature_operation_per_tri],
        dtype=np.int64
    )
    location_id = np.asarray(
        [LOCATION_TO_ID.get(name, 0) for name in location_per_tri],
        dtype=np.int64
    )
    timeline_index = np.asarray(timeline_index_per_tri, dtype=np.int64)
    timeline_index_norm = _normalize_timeline_indices(timeline_index)
    instance_boundary_mask = build_instance_boundary_mask(mesh, feature_instance_id)

    metadata = {
        "feature_instance_id": feature_instance_id,
        "feature_type_id": feature_type_id,
        "feature_operation_id": feature_operation_id,
        "location_id": location_id,
        "timeline_index": timeline_index,
        "timeline_index_norm": timeline_index_norm,
        "instance_boundary_mask": instance_boundary_mask,
        "brep_face_index": fidx.astype(np.int64),
        "feature_instance_uuid": np.asarray(unique_feature_uuids),
        "mesh_face_count": np.asarray([len(mesh.faces)], dtype=np.int64),
    }
    if seg_labels is not None:
        metadata["coarse_seg_label"] = seg_labels.astype(np.int64)
    return metadata


def save_instruction_metadata(output_path, metadata):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(output_path, **metadata)
    return output_path


def load_instruction_metadata(meta_path):
    data = np.load(meta_path, allow_pickle=False)
    return {key: data[key] for key in data.files}
