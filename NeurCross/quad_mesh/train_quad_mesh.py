import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
import torch.optim as optim
from torchinfo import summary

from models import Network_predict_angle
from models import MorseLoss_quad_mesh as MorseLoss
import utils.utils as utils
import quad_mesh_args

import quad_mesh_dataset as dataset

# get training parameters
args = quad_mesh_args.get_args()

if args.guidance_mode == 'feature' and args.part_feat_path is None:
    raise ValueError("guidance_mode='feature' requires --part_feat_path")
if args.guidance_mode == 'instruction' and args.instruction_meta_path is None:
    raise ValueError("guidance_mode='instruction' requires --instruction_meta_path")
if args.guidance_mode == 'instruction':
    from instruction_guidance import load_instruction_metadata
    from instruction_guidance.control_targets import compute_instance_boundary_gradient

file_name = os.path.splitext(args.data_path.split('/')[-1])[0]
logdir = os.path.join(args.logdir, file_name)
os.makedirs(logdir, exist_ok=True)

# set up logging
log_file = utils.setup_logdir_only_log(logdir, args)

device = 'cpu' if not torch.cuda.is_available() else 'cuda'

# get data loaders
utils.same_seed(args.seed)
train_set = dataset.ReconDataset(args.data_path, args.n_points, args.n_samples, args.grid_res)

train_dataloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=(device == 'cuda'),
)
# get model
net = Network_predict_angle(in_dim=3, angle_in_dim=12, decoder_hidden_dim=args.decoder_hidden_dim, nl=args.nl,
                            decoder_n_hidden_layers=args.decoder_n_hidden_layers, init_type=args.init_type,
                            sphere_init_params=args.sphere_init_params, udf=args.udf)

net.to(device)
if args.load_path is not None:
    net.load_state_dict(torch.load(args.load_path))
    print('Loaded model from %s' % args.load_path)
summary(net.decoder, (1, 1024, 3))

n_parameters = utils.count_parameters(net)
utils.log_string("Number of parameters in the current model:{}".format(n_parameters), log_file)

# Setup Adam optimizers
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)
n_iterations = args.n_samples * (args.num_epochs)
print('n_iterations: ', n_iterations)

net.to(device)

num_batches = len(train_dataloader)
refine_flag = True
min_cd = np.inf
SAVE_BEST = False

##################################################################################
# get the vertices neighbors of the mesh
vertex_neighbors = utils.get_sample_vers_neighbors_for_face_center_points_or_vertices(args.data_path)
vertex_neighbors_list = utils.calculate_same_neighbors_verts(vertex_neighbors)
if len(vertex_neighbors) != len(train_set.points):
    raise ValueError(
        "Face neighbor list length mismatch for '{}': expected {}, got {}".format(
            args.data_path, len(train_set.points), len(vertex_neighbors)
        )
    )
###################################################################################
axis_angle_R_mat_list = utils.get_rotation_matrix(vertex_neighbors_list, vertex_neighbors, args.data_path)

semantic_grad_dir_tensor = None
semantic_grad_weight_tensor = None
semantic_labels_tensor = None
semantic_features_tensor = None

instruction_instance_labels_tensor = None
instruction_boundary_grad_dir_tensor = None
instruction_boundary_grad_weight_tensor = None
instruction_extrude_dir_tensor = None
instruction_extrude_face_weight_tensor = None
instruction_revolve_dir_tensor = None
instruction_revolve_face_weight_tensor = None
instruction_chamfer_dir_tensor = None
instruction_chamfer_face_weight_tensor = None
instruction_fillet_dir_tensor = None
instruction_fillet_face_weight_tensor = None
instruction_anchor_dir_tensor = None
instruction_anchor_face_weight_tensor = None

if args.guidance_mode == 'feature' and args.part_feat_path is not None:
    part_features = np.load(args.part_feat_path)  # (N_faces, 448)

    face_centers = train_set.points
    face_normals = train_set.mnfld_n
    if len(part_features) != len(face_centers):
        raise ValueError(
            "Part feature count mismatch for '{}': expected {}, got {}".format(
                args.part_feat_path, len(face_centers), len(part_features)
            )
        )

    from utils.semantic_utils import compute_semantic_gradient
    sg_method = getattr(args, 'semantic_gradient_method', 'jacobian')
    sg_normalize = bool(getattr(args, 'semantic_normalize_features', 1))
    sg_pca_dim = getattr(args, 'semantic_pca_dim', 32) or None
    if sg_pca_dim == 0:
        sg_pca_dim = None
    sg_distance_sigma = getattr(args, 'semantic_distance_sigma', 0.0) or None
    if sg_distance_sigma == 0.0:
        sg_distance_sigma = None
    semantic_labels = None
    try:
        from eval.label_utils import cluster_features

        spatial_w = getattr(args, 'semantic_spatial_cluster_weight', 0.0)
        if spatial_w > 0:
            try:
                from eval.label_utils import cluster_features_spatial
            except ImportError as exc:
                raise ImportError(
                    "--semantic_spatial_cluster_weight > 0 requires "
                    "eval.label_utils.cluster_features_spatial"
                ) from exc
            label_result = cluster_features_spatial(
                part_features, face_centers,
                spatial_weight=spatial_w, method="best_silhouette")
        else:
            label_result = cluster_features(part_features, method="best_silhouette")
        semantic_labels = label_result["labels"]
        if semantic_labels is not None and len(semantic_labels) == len(face_centers):
            semantic_labels_tensor = torch.tensor(semantic_labels, dtype=torch.long).to(device)
            utils.log_string(
                "Semantic pseudo labels: K={}, silhouette={:.4f}{}".format(
                    label_result["k"], label_result["silhouette"],
                    " (spatial)" if spatial_w > 0 else ""
                ),
                log_file
            )
    except Exception as exc:
        utils.log_string(
            "Semantic label clustering skipped: {}".format(exc),
            log_file
        )

    print(f"  Semantic gradient: method={sg_method}, normalize={sg_normalize}, "
          f"pca_dim={sg_pca_dim}, distance_sigma={sg_distance_sigma}")
    grad_dir, grad_weight = compute_semantic_gradient(
        face_centers, face_normals, vertex_neighbors, part_features,
        method=sg_method,
        normalize=sg_normalize,
        pca_dim=sg_pca_dim,
        distance_sigma=sg_distance_sigma,
    )
    semantic_grad_dir_tensor = torch.tensor(grad_dir, dtype=torch.float32).to(device)
    semantic_grad_weight_tensor = torch.tensor(grad_weight, dtype=torch.float32).to(device)

    feat_norm = np.linalg.norm(part_features, axis=-1, keepdims=True)
    part_features_normalized = part_features / np.clip(feat_norm, 1e-12, None)
    if sg_pca_dim is not None:
        from utils.semantic_utils import reduce_features_pca
        soft_features = reduce_features_pca(part_features_normalized, n_components=sg_pca_dim)
    else:
        soft_features = part_features_normalized
    semantic_features_tensor = torch.tensor(soft_features, dtype=torch.float32).to(device)

if args.guidance_mode == 'instruction' and args.instruction_meta_path is not None:
    from instruction_guidance.control_targets import build_instruction_control_targets

    instruction_meta = load_instruction_metadata(args.instruction_meta_path)
    instruction_instance_labels_tensor = torch.tensor(
        instruction_meta["feature_instance_id"], dtype=torch.long).to(device)

    face_centers = train_set.points
    face_normals = train_set.mnfld_n
    boundary_grad_dir, boundary_grad_weight = compute_instance_boundary_gradient(
        face_centers, face_normals, vertex_neighbors,
        instruction_meta["feature_instance_id"],
    )
    instruction_boundary_grad_dir_tensor = torch.tensor(
        boundary_grad_dir, dtype=torch.float32).to(device)
    instruction_boundary_grad_weight_tensor = torch.tensor(
        boundary_grad_weight, dtype=torch.float32).to(device)

    control_targets = build_instruction_control_targets(
        instruction_meta, face_centers, face_normals, vertex_neighbors)

    def _to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32).to(device)

    instruction_extrude_dir_tensor = _to_tensor(control_targets["instruction_extrude_dir"])
    instruction_extrude_face_weight_tensor = _to_tensor(control_targets["instruction_extrude_weight"])
    instruction_revolve_dir_tensor = _to_tensor(control_targets["instruction_revolve_dir"])
    instruction_revolve_face_weight_tensor = _to_tensor(control_targets["instruction_revolve_weight"])
    instruction_chamfer_dir_tensor = _to_tensor(control_targets["instruction_chamfer_dir"])
    instruction_chamfer_face_weight_tensor = _to_tensor(control_targets["instruction_chamfer_weight"])
    instruction_fillet_dir_tensor = _to_tensor(control_targets["instruction_fillet_dir"])
    instruction_fillet_face_weight_tensor = _to_tensor(control_targets["instruction_fillet_weight"])
    instruction_anchor_dir_tensor = _to_tensor(control_targets["instruction_anchor_dir"])
    instruction_anchor_face_weight_tensor = _to_tensor(control_targets["instruction_anchor_weight"])

    n_extrude = int((control_targets["instruction_extrude_weight"] > 0).sum())
    n_revolve = int((control_targets["instruction_revolve_weight"] > 0).sum())
    n_chamfer = int((control_targets["instruction_chamfer_weight"] > 0).sum())
    n_fillet = int((control_targets["instruction_fillet_weight"] > 0).sum())
    n_anchor = int((control_targets["instruction_anchor_weight"] > 0).sum())

    num_instances = len(np.unique(instruction_meta["feature_instance_id"]))
    num_boundary_faces = int((boundary_grad_weight > 0).sum())
    utils.log_string(
        "Instruction metadata loaded: {} instances, {} faces, {} boundary faces".format(
            num_instances,
            len(instruction_meta["feature_instance_id"]),
            num_boundary_faces,
        ),
        log_file
    )
    utils.log_string(
        "Operation directions: extrude={}, revolve={}, chamfer={}, fillet={}, anchor={}".format(
            n_extrude, n_revolve, n_chamfer, n_fillet, n_anchor),
        log_file
    )

utils.log_string("Guidance mode: {}".format(args.guidance_mode), log_file)

criterion = MorseLoss(weights=args.loss_weights, loss_type=args.loss_type, div_decay=args.morse_decay,
                      div_type=args.morse_type,
                      vertex_neighbors_list=vertex_neighbors_list,
                      vertex_neighbors=vertex_neighbors, axis_angle_R_mat_list=axis_angle_R_mat_list,
                      device=device,
                      guidance_mode=args.guidance_mode,
                      # Feature mode
                      semantic_grad_dir=semantic_grad_dir_tensor,
                      semantic_grad_weight=semantic_grad_weight_tensor,
                      semantic_labels=semantic_labels_tensor,
                      semantic_boundary_weight=args.semantic_boundary_weight,
                      semantic_intra_weight=args.semantic_intra_weight,
                      semantic_neighbor_weight=args.semantic_neighbor_weight,
                      semantic_cross_part_gamma=args.semantic_cross_part_gamma,
                      semantic_boundary_reward=args.semantic_boundary_reward,
                      semantic_features=semantic_features_tensor,
                      semantic_diversity_weight=args.semantic_diversity_weight,
                      semantic_diversity_margin=args.semantic_diversity_margin,
                      semantic_soft_boundary_temp=args.semantic_soft_boundary_temp,
                      semantic_soft_boundary_chunk_size=args.semantic_soft_boundary_chunk_size,
                      alignment_margin=args.alignment_margin,
                      intra_temperature=args.intra_temperature,
                      intra_hard_ratio=args.intra_hard_ratio,
                      # Instruction mode
                      instruction_instance_labels=instruction_instance_labels_tensor,
                      instruction_boundary_grad_dir=instruction_boundary_grad_dir_tensor,
                      instruction_boundary_grad_weight=instruction_boundary_grad_weight_tensor,
                      instruction_boundary_weight=args.instruction_boundary_weight,
                      instruction_intra_weight=args.instruction_intra_weight,
                      instruction_cross_instance_gamma=args.instruction_cross_instance_gamma,
                      instruction_operation_align_weight=args.instruction_operation_align_weight,
                      instruction_anchor_weight=args.instruction_anchor_weight,
                      instruction_extrude_dir=instruction_extrude_dir_tensor,
                      instruction_extrude_face_weight=instruction_extrude_face_weight_tensor,
                      instruction_revolve_dir=instruction_revolve_dir_tensor,
                      instruction_revolve_face_weight=instruction_revolve_face_weight_tensor,
                      instruction_chamfer_dir=instruction_chamfer_dir_tensor,
                      instruction_chamfer_face_weight=instruction_chamfer_face_weight_tensor,
                      instruction_fillet_dir=instruction_fillet_dir_tensor,
                      instruction_fillet_face_weight=instruction_fillet_face_weight_tensor,
                      instruction_anchor_dir=instruction_anchor_dir_tensor,
                      instruction_anchor_face_weight=instruction_anchor_face_weight_tensor,
                      # Safeguards
                      guidance_warmup_fraction=args.guidance_warmup_fraction,
                      guidance_cap_ratio=args.guidance_cap_ratio,
                      guidance_eikonal_guard=args.guidance_eikonal_guard,
                      guidance_warmup_type=args.guidance_warmup_type)

# For each epoch
for epoch in range(args.num_epochs):
    for batch_idx, data in enumerate(train_dataloader):
        if batch_idx != 0 and (batch_idx % 500 == 0 or batch_idx == len(train_dataloader) - 1):
            SAVE_BEST = True

        net.zero_grad()
        net.train()

        mnfld_points, mnfld_n_gt, nonmnfld_points, near_points, local_coord_u, local_coord_v = data[
            'points'].to(device), data['mnfld_n'].to(device), data['nonmnfld_points'].to(device), data[
            'near_points'].to(device), data['local_coordinates_u'].to(device), data['local_coordinates_v'].to(device)

        mnfld_points.requires_grad_()
        nonmnfld_points.requires_grad_()
        near_points.requires_grad_()

        features = torch.cat((mnfld_points, mnfld_n_gt, local_coord_u, local_coord_v), dim=-1)

        output_pred, mnfld_pts_theta_output_pred = net(nonmnfld_points, mnfld_points,
                                                       near_points=near_points if args.morse_near else None,
                                                       angle_features=features)

        loss_dict = criterion(output_pred, mnfld_points, nonmnfld_points, mnfld_n_gt,
                              near_points=near_points if args.morse_near else None, batch_idx=batch_idx,
                              logdir=logdir, filename=file_name, save_best=SAVE_BEST,
                              mnfld_pts_theta_output_pred=mnfld_pts_theta_output_pred,
                              local_coord_u=local_coord_u, local_coord_v=local_coord_v)

        lr = torch.tensor(optimizer.param_groups[0]['lr'])
        loss_dict["lr"] = lr

        loss_dict["loss"].backward()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip_norm)

        SAVE_BEST = False
        optimizer.step()

        # Output training stats
        if batch_idx % 10 == 0:
            weights = criterion.weights
            utils.log_string("Weights: {}, lr={:.3e}".format(weights, lr), log_file)
            guidance_w = weights[6] if len(weights) > 6 else 0.0
            utils.log_string('Epoch: {} [{:4d}/{} ({:.0f}%)] Loss: {:.5f} = L_Mnfld: {:.5f} + '
                             'L_NonMnfld: {:.5f} + L_Eknl: {:.5f} + L_Morse: {:.5f} + L_thetaHessian: {:.5f} + '
                             'L_thetaNeighbor: {:.5f} + L_Guidance: {:.5f}'.format(
                epoch, batch_idx * args.batch_size, len(train_set), 100. * batch_idx / args.n_samples,
                loss_dict["loss"].item(), weights[0] * loss_dict["sdf_term"].item(),
                       weights[1] * loss_dict["inter_term"].item(),
                       weights[3] * loss_dict["eikonal_term"].item(), weights[5] * loss_dict["morse_term"].item(),
                       weights[2] * loss_dict["theta_hessian_term"].item(),
                       weights[4] * loss_dict['theta_neighbors_term'].item(),
                       guidance_w * loss_dict['guidance_loss'].item()
            ),
                log_file)
            utils.log_string('Epoch: {} [{:4d}/{} ({:.0f}%)] Unweighted L_s : L_Mnfld: {:.5f} + '
                             'L_NonMnfld: {:.5f} + L_Eknl: {:.5f} + L_Morse: {:.5f} + L_thetaHessian: {:.5f} + '
                             'L_thetaNeighbor: {:.5f} + L_Guidance: {:.5f}'.format(
                epoch, batch_idx * args.batch_size, len(train_set), 100. * batch_idx / args.n_samples,
                loss_dict["sdf_term"].item(), loss_dict["inter_term"].item(),
                loss_dict["eikonal_term"].item(), loss_dict["morse_term"].item(),
                loss_dict['theta_hessian_term'].item(), loss_dict['theta_neighbors_term'].item(),
                loss_dict['guidance_loss'].item()),
                log_file)
            if args.guidance_mode == 'feature':
                utils.log_string(
                    'Feature guidance terms: boundary={:.5f}, intra={:.5f}, neighbor={:.5f}, diversity={:.5f}'.format(
                        loss_dict['semantic_boundary_term'].item(),
                        loss_dict['semantic_intra_term'].item(),
                        loss_dict['semantic_neighbor_term'].item(),
                        loss_dict['semantic_diversity_term'].item(),
                    ),
                    log_file
                )
            elif args.guidance_mode == 'instruction':
                utils.log_string(
                    'Instruction guidance terms: boundary_align={:.5f}, intra={:.5f}, op_align={:.5f}, anchor={:.5f}'.format(
                        loss_dict['instruction_boundary_align_term'].item(),
                        loss_dict['instruction_intra_term'].item(),
                        loss_dict['instruction_operation_align_term'].item(),
                        loss_dict['instruction_anchor_term'].item(),
                    ),
                    log_file
                )
            utils.log_string('', log_file)

        criterion.update_morse_weight(epoch * args.n_samples + batch_idx, args.num_epochs * args.n_samples,
                                      args.decay_params)  # assumes batch size of 1
        criterion.update_guidance_weight(epoch * args.n_samples + batch_idx,
                                         args.num_epochs * args.n_samples)

torch.save(net.state_dict(), os.path.join(logdir, file_name + '_model.pth'))
