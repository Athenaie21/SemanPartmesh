import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
import torch.optim as optim
from torchinfo import summary
from instruction_guidance import load_instruction_metadata

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

file_name = os.path.splitext(args.data_path.split('/')[-1])[0]
logdir = os.path.join(args.logdir, file_name)
os.makedirs(logdir, exist_ok=True)

# set up logging
log_file = utils.setup_logdir_only_log(logdir, args)

device = 'cpu' if not torch.cuda.is_available() else 'cuda'

# get data loaders
utils.same_seed(args.seed)
train_set = dataset.ReconDataset(args.data_path, args.n_points, args.n_samples, args.grid_res)

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)
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
###################################################################################
axis_angle_R_mat_list = utils.get_rotation_matrix(vertex_neighbors_list, vertex_neighbors, args.data_path)

semantic_grad_dir_tensor = None
semantic_grad_weight_tensor = None
semantic_labels_tensor = None
instruction_instance_labels_tensor = None
instruction_feature_type_tensor = None
instruction_location_tensor = None

if args.guidance_mode == 'feature' and args.part_feat_path is not None:
    part_features = np.load(args.part_feat_path)  # (N_faces, 448)

    face_centers = train_set.points       # already centered & scaled
    face_normals = train_set.mnfld_n

    from utils.semantic_utils import compute_semantic_gradient
    grad_dir, grad_weight = compute_semantic_gradient(
        face_centers, face_normals, vertex_neighbors, part_features
    )
    semantic_grad_dir_tensor = torch.tensor(grad_dir, dtype=torch.float32).to(device)
    semantic_grad_weight_tensor = torch.tensor(grad_weight, dtype=torch.float32).to(device)

    try:
        from eval.label_utils import cluster_features

        label_result = cluster_features(part_features, method="best_silhouette")
        semantic_labels = label_result["labels"]
        if semantic_labels is not None and len(semantic_labels) == len(face_centers):
            semantic_labels_tensor = torch.tensor(semantic_labels, dtype=torch.long).to(device)
            utils.log_string(
                "Semantic pseudo labels: K={}, silhouette={:.4f}".format(
                    label_result["k"], label_result["silhouette"]
                ),
                log_file
            )
    except Exception as exc:
        utils.log_string(
            "Semantic label clustering skipped: {}".format(exc),
            log_file
        )

if args.guidance_mode == 'instruction' and args.instruction_meta_path is not None:
    instruction_meta = load_instruction_metadata(args.instruction_meta_path)
    instruction_instance_labels_tensor = torch.tensor(
        instruction_meta["feature_instance_id"], dtype=torch.long).to(device)
    instruction_feature_type_tensor = torch.tensor(
        instruction_meta["feature_type_id"], dtype=torch.long).to(device)
    instruction_location_tensor = torch.tensor(
        instruction_meta["location_id"], dtype=torch.long).to(device)
    utils.log_string(
        "Instruction metadata loaded: {} instances, {} faces".format(
            len(np.unique(instruction_meta["feature_instance_id"])),
            len(instruction_meta["feature_instance_id"])
        ),
        log_file
    )

utils.log_string("Guidance mode: {}".format(args.guidance_mode), log_file)

criterion = MorseLoss(weights=args.loss_weights, loss_type=args.loss_type, div_decay=args.morse_decay,
                      div_type=args.morse_type,
                      vertex_neighbors_list=vertex_neighbors_list,
                      vertex_neighbors=vertex_neighbors, axis_angle_R_mat_list=axis_angle_R_mat_list,
                      device=device,
                      guidance_mode=args.guidance_mode,
                      semantic_grad_dir=semantic_grad_dir_tensor,
                      semantic_grad_weight=semantic_grad_weight_tensor,
                      semantic_labels=semantic_labels_tensor,
                      semantic_boundary_weight=args.semantic_boundary_weight,
                      semantic_intra_weight=args.semantic_intra_weight,
                      semantic_neighbor_weight=args.semantic_neighbor_weight,
                      semantic_cross_part_gamma=args.semantic_cross_part_gamma,
                      instruction_instance_labels=instruction_instance_labels_tensor,
                      instruction_feature_type=instruction_feature_type_tensor,
                      instruction_location=instruction_location_tensor,
                      instruction_boundary_weight=args.instruction_boundary_weight,
                      instruction_intra_weight=args.instruction_intra_weight,
                      instruction_type_weight=args.instruction_type_weight,
                      instruction_cross_instance_gamma=args.instruction_cross_instance_gamma)

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
            utils.log_string('Epoch: {} [{:4d}/{} ({:.0f}%)] Loss: {:.5f} = L_Mnfld: {:.5f} + '
                             'L_NonMnfld: {:.5f} + L_Eknl: {:.5f} + L_Morse: {:.5f} + L_thetaHessian: {:.5f} + '
                             'L_thetaNeighbor: {:.5f} + L_Guidance: {:.5f}'.format(
                epoch, batch_idx * args.batch_size, len(train_set), 100. * batch_idx / args.n_samples,
                loss_dict["loss"].item(), weights[0] * loss_dict["sdf_term"].item(),
                       weights[1] * loss_dict["inter_term"].item(),
                       weights[3] * loss_dict["eikonal_term"].item(), weights[5] * loss_dict["morse_term"].item(),
                       weights[2] * loss_dict["theta_hessian_term"].item(),
                       weights[4] * loss_dict['theta_neighbors_term'].item(),
                       weights[6] * loss_dict['guidance_loss'].item()
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
                    'Feature guidance terms: boundary={:.5f}, intra={:.5f}, neighbor={:.5f}'.format(
                        loss_dict['semantic_boundary_term'].item(),
                        loss_dict['semantic_intra_term'].item(),
                        loss_dict['semantic_neighbor_term'].item()
                    ),
                    log_file
                )
            elif args.guidance_mode == 'instruction':
                utils.log_string(
                    'Instruction guidance terms: boundary={:.5f}, intra={:.5f}, type={:.5f}'.format(
                        loss_dict['instruction_boundary_term'].item(),
                        loss_dict['instruction_intra_term'].item(),
                        loss_dict['instruction_type_term'].item()
                    ),
                    log_file
                )
            utils.log_string('', log_file)

        criterion.update_morse_weight(epoch * args.n_samples + batch_idx, args.num_epochs * args.n_samples,
                                      args.decay_params)  # assumes batch size of 1

torch.save(net.state_dict(), os.path.join(logdir, file_name + '_model.pth'))
