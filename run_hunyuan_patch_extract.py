import os
import json
import subprocess
from pathlib import Path
from collections import deque

import numpy as np
import trimesh

ROOT = Path('/root/SemanPartMesh')
OUT_ROOT = Path('/root/shared-nvme/SemanPartMesh/hunyuan_final_extract')
EXTRACT = ROOT / 'extract_quad.py'
PYTHON = '/root/.conda/envs/neurcross/bin/python'

MESHES = ['28', '42', '76']
ANGLE_THR_DEG = 30.0
MIN_PATCH_FACES = 120
MAX_PATCH_FACES = 1400
TARGET_COVERAGE = 0.80


def smooth_components(mesh, angle_thr_deg=30.0):
    fa = np.asarray(mesh.face_adjacency, dtype=np.int64)
    ang = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
    thr = np.deg2rad(float(angle_thr_deg))
    adj = [[] for _ in range(len(mesh.faces))]
    for (a, b), an in zip(fa, ang):
        if an <= thr:
            a = int(a)
            b = int(b)
            adj[a].append(b)
            adj[b].append(a)
    vis = np.zeros(len(mesh.faces), dtype=bool)
    comps = []
    for i in range(len(mesh.faces)):
        if vis[i]:
            continue
        st = [i]
        vis[i] = True
        comp = []
        while st:
            x = st.pop()
            comp.append(x)
            for y in adj[x]:
                if not vis[y]:
                    vis[y] = True
                    st.append(y)
        comps.append(np.asarray(comp, dtype=np.int64))
    comps.sort(key=len, reverse=True)
    return comps, adj


def split_component_by_size(component, full_adj, max_faces):
    comp_set = set(int(x) for x in component.tolist())
    unvisited = set(comp_set)
    chunks = []
    while unvisited:
        seed = next(iter(unvisited))
        q = deque([seed])
        unvisited.remove(seed)
        chunk = []
        while q and len(chunk) < max_faces:
            x = q.popleft()
            chunk.append(x)
            for y in full_adj[x]:
                if y in unvisited:
                    unvisited.remove(y)
                    q.append(y)
                    if len(chunk) + len(q) >= max_faces:
                        break
        chunks.append(np.asarray(chunk, dtype=np.int64))
    return chunks


def compact_submesh(mesh, face_ids):
    faces = np.asarray(mesh.faces, dtype=np.int64)[face_ids]
    used, inv = np.unique(faces.reshape(-1), return_inverse=True)
    verts = np.asarray(mesh.vertices)[used]
    faces = inv.reshape(-1, 3)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def merge_quads(obj_paths, out_path):
    verts_all = []
    quads_all = []
    offset = 0
    for path in obj_paths:
        verts = []
        quads = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    p = line.strip().split()
                    verts.append([float(p[1]), float(p[2]), float(p[3])])
                elif line.startswith('f '):
                    ids = [int(x.split('/')[0]) - 1 for x in line.strip().split()[1:]]
                    if len(ids) == 4:
                        quads.append(ids)
        if not verts or not quads:
            continue
        verts = np.asarray(verts, dtype=np.float64)
        quads = np.asarray(quads, dtype=np.int64)
        verts_all.append(verts)
        quads_all.append(quads + offset)
        offset += len(verts)
    if not verts_all:
        return False
    V = np.vstack(verts_all)
    Q = np.vstack(quads_all)
    with open(out_path, 'w') as f:
        for v in V:
            f.write(f'v {v[0]:.12g} {v[1]:.12g} {v[2]:.12g}\n')
        for q in Q:
            f.write(f'f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n')
    return True


for name in MESHES:
    mesh_path = ROOT / 'pipeline_output_hunyuan_repaired' / 'processed_meshes' / f'{name}.obj'
    cf_path = ROOT / 'pipeline_output_hunyuan_repaired' / 'neurcross_logs' / name / 'save_crossField' / f'{name}_iter_9999.txt'
    out_dir = OUT_ROOT / name
    patch_dir = out_dir / 'patches'
    patch_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load_mesh(mesh_path, process=False)
    rows = np.atleast_2d(np.loadtxt(cf_path, dtype=np.float64))
    comps, full_adj = smooth_components(mesh, ANGLE_THR_DEG)

    planned = []
    for comp in comps:
        if len(comp) < MIN_PATCH_FACES:
            continue
        if len(comp) <= MAX_PATCH_FACES:
            planned.append(comp)
        else:
            planned.extend(split_component_by_size(comp, full_adj, MAX_PATCH_FACES))

    planned = [p for p in planned if len(p) >= MIN_PATCH_FACES]
    planned.sort(key=len, reverse=True)

    success_objs = []
    attempted = []
    success_faces = 0
    total_faces = int(len(mesh.faces))

    for idx, face_ids in enumerate(planned):
        if success_faces / max(total_faces, 1) >= TARGET_COVERAGE:
            break
        patch_name = f'patch_{idx:03d}'
        patch_mesh = patch_dir / f'{patch_name}.obj'
        patch_cf = patch_dir / f'{patch_name}_crossfield.txt'
        patch_quad = patch_dir / f'{patch_name}_quad.obj'

        sub = compact_submesh(mesh, face_ids)
        sub.export(patch_mesh)
        np.savetxt(patch_cf, rows[np.asarray(face_ids, dtype=np.int64)], fmt='%.8f')

        cmd = [
            PYTHON, str(EXTRACT),
            '--mesh', str(patch_mesh),
            '--crossfield', str(patch_cf),
            '--output', str(patch_quad),
            '--gradient_size', '30',
            '--timeout', '1200',
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        attempted.append({
            'patch': patch_name,
            'faces': int(len(face_ids)),
            'returncode': int(proc.returncode),
            'output': str(patch_quad),
        })
        if proc.returncode == 0 and patch_quad.exists() and patch_quad.stat().st_size > 0:
            success_objs.append(str(patch_quad))
            success_faces += int(len(face_ids))
            print(f'[{name}] OK    {patch_name} faces={len(face_ids)} coverage={success_faces}/{total_faces}')
        else:
            log_path = patch_dir / f'{patch_name}.log'
            log_path.write_text(proc.stdout)
            print(f'[{name}] FAIL  {patch_name} faces={len(face_ids)} rc={proc.returncode}')

    merged = out_dir / f'{name}_quad.obj'
    merged_ok = merge_quads(success_objs, merged)
    summary = {
        'mesh': name,
        'total_faces': total_faces,
        'planned_patches': len(planned),
        'attempted_patches': len(attempted),
        'successful_patches': len(success_objs),
        'successful_face_coverage': float(success_faces / max(total_faces, 1)),
        'merged_output': str(merged) if merged_ok else None,
        'attempted': attempted,
    }
    (out_dir / f'{name}_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'[{name}] merged_ok={merged_ok} coverage={summary["successful_face_coverage"]:.3f} output={merged if merged_ok else None}')
