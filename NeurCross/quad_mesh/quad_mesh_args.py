import argparse

def add_args(parser):
    parser.add_argument('--logdir', type=str, default='./doubleTorus/', help='log directory')
    parser.add_argument('--model_name', type=str, default='model', help='trained model name')
    parser.add_argument('--seed', type=int, default=3627473, help='random seed')
    parser.add_argument('--data_path', type=str, default='../data/doubleTorus/input/doubleTorus.ply', help='path to input dir')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='numbers of epochs')
    parser.add_argument('--n_points', type=int, default=15000, help='number of points in each point cloud')
    parser.add_argument('--grid_res', type=int, default=256, help='uniform grid resolution')
    parser.add_argument('--nonmnfld_sample_type', type=str, default='gaussian',
                        help='how to sample points off the manifold - grid | gaussian | combined')

    # training parameters
    parser.add_argument('--num_epochs', type=int, default=1, help='always be 1')
    parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate')
    parser.add_argument('--grad_clip_norm', type=float, default=10.0, help='Value to clip gradients to')
    parser.add_argument('--batch_size', type=int, default=1, help='number of samples in a minibatch')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader worker count; use 0 for the most stable single-mesh training')
    parser.add_argument('--load_path', type=str, default=None)

    # Network architecture and loss
    parser.add_argument('--init_type', type=str, default='siren',
                        help='initialization type siren | geometric_sine | geometric_relu | mfgi')
    parser.add_argument('--decoder_hidden_dim', type=int, default=256, help='length of decoder hidden dim')
    parser.add_argument('--decoder_n_hidden_layers', type=int, default=4, help='number of decoder hidden layers')
    parser.add_argument('--latent_size', type=int, default=0)
    parser.add_argument('--nl', type=str, default='sine', help='type of non linearity sine | relu | softplus')
    parser.add_argument('--sphere_init_params', nargs='+', type=float, default=[1.6, 0.1],
                        help='radius and scaling')
    parser.add_argument('--udf', action='store_true')
    parser.add_argument('--output_any', action='store_true')

    parser.add_argument('--loss_type', type=str, default='siren_wo_n_w_morse_w_theta')
    parser.add_argument('--decay_params', nargs='+', type=float, default=[3, 0.2, 3, 0.4, 0.001, 0],
                        help='epoch number to evaluate')
    parser.add_argument('--morse_type', type=str, default='l1', help='divergence term norm l1 | l2')
    parser.add_argument('--morse_decay', type=str, default='linear',
                        help='divergence term importance decay none | step | linear')
    parser.add_argument('--loss_weights', nargs='+', type=float, default=[7e3, 6e2, 10, 5e1, 30, 3, 50],
                        help='loss terms weights sdf | inter | normal | eikonal | div | morse | guidance')
    parser.add_argument('--guidance_mode', type=str, default='feature',
                        choices=['none', 'feature', 'instruction'],
                        help='extra guidance mode for cross-field learning')
    parser.add_argument('--part_feat_path', type=str, default=None, help='path to precomputed PartField features (.npy)')
    parser.add_argument('--instruction_meta_path', type=str, default=None,
                        help='path to precomputed instruction metadata (.npz)')

    # Feature mode params
    parser.add_argument('--semantic_gradient_method', type=str, default='jacobian',
                        choices=['structure_tensor', 'gradient_avg', 'jacobian'],
                        help='method for computing semantic gradient field')
    parser.add_argument('--semantic_boundary_weight', type=float, default=2.0,
                        help='internal weight for semantic boundary alignment')
    parser.add_argument('--semantic_intra_weight', type=float, default=1.0,
                        help='internal weight for semantic intra-part consistency')
    parser.add_argument('--semantic_neighbor_weight', type=float, default=1.0,
                        help='internal weight for semantic-aware neighbor smoothness')
    parser.add_argument('--semantic_cross_part_gamma', type=float, default=0.2,
                        help='neighbor smoothness attenuation across semantic parts')
    parser.add_argument('--semantic_boundary_reward', type=float, default=0.0,
                        help='optional cross-boundary override/reward; 0 uses semantic_cross_part_gamma')
    parser.add_argument('--semantic_pca_dim', type=int, default=0,
                        help='PCA dimensionality for feature reduction (0=disable)')
    parser.add_argument('--semantic_normalize_features', type=int, default=1,
                        help='L2 normalize features before gradient computation (1=yes, 0=no)')
    parser.add_argument('--semantic_distance_sigma', type=float, default=0.0,
                        help='Gaussian distance sigma for multi-scale gradient weighting (0=disable)')
    parser.add_argument('--semantic_diversity_weight', type=float, default=0.5,
                        help='weight for cross-part diversity loss')
    parser.add_argument('--semantic_diversity_margin', type=float, default=0.3,
                        help='margin threshold for cross-part diversity loss')
    parser.add_argument('--semantic_soft_boundary_temp', type=float, default=0.1,
                        help='temperature for soft boundary weight from feature similarity')
    parser.add_argument('--semantic_soft_boundary_chunk_size', type=int, default=1024,
                        help='pair count per chunk for soft boundary cosine similarity (0=disable chunking)')
    parser.add_argument('--alignment_margin', type=float, default=0.0,
                        help='margin for boundary alignment loss')
    parser.add_argument('--semantic_spatial_cluster_weight', type=float, default=0.0,
                        help='spatial weight for spatially-regularized clustering (0=pure feature clustering)')
    parser.add_argument('--intra_temperature', type=float, default=1.0,
                        help='temperature for intra consistency loss (feature mode)')
    parser.add_argument('--intra_hard_ratio', type=float, default=0.0,
                        help='fraction of worst-aligned pairs to focus on (feature mode, 0=all)')

    # Instruction mode params
    parser.add_argument('--instruction_boundary_weight', type=float, default=1.5,
                        help='internal weight for instruction boundary alignment')
    parser.add_argument('--instruction_intra_weight', type=float, default=1.0,
                        help='internal weight for instruction instance consistency')
    parser.add_argument('--instruction_cross_instance_gamma', type=float, default=0.4,
                        help='cross-instance neighbor smoothness weight (0=no smoothness, 1=full)')
    parser.add_argument('--instruction_operation_align_weight', type=float, default=1.0,
                        help='internal weight for operation direction alignment (extrude/revolve/chamfer/fillet)')
    parser.add_argument('--instruction_anchor_weight', type=float, default=0.3,
                        help='internal weight for instance anchor direction alignment')

    # Guidance safeguards
    parser.add_argument('--guidance_warmup_fraction', type=float, default=0.15,
                        help='fraction of training for guidance weight warmup (0=no warmup)')
    parser.add_argument('--guidance_cap_ratio', type=float, default=0.5,
                        help='cap weighted guidance to this ratio of geometric loss (0=disabled)')
    parser.add_argument('--guidance_eikonal_guard', type=float, default=0.0,
                        help='scale down guidance when eikonal exceeds this threshold (0=disabled)')
    parser.add_argument('--guidance_warmup_type', type=str, default='linear',
                        choices=['linear', 'cosine'],
                        help='warmup schedule type for guidance weight')

    # Morse
    parser.add_argument('--morse_near', action='store_true')
    parser.add_argument('--weight_for_morse', action='store_true',
                        help='if true, Weighting A according to the distance of the sampling point')
    parser.add_argument('--use_morse_nonmnfld_grad', type=bool, default=True, help='if True, use morse loss on nonmnfld')
    parser.add_argument('--relax_morse', type=float, default=0.5, help='the max value of relax Morse')
    parser.add_argument('--use_vertices', type=bool, default=False, help='if False, sample points to overfitting')
    parser.add_argument('--featureLine_threshold', type=float, default=1.0)

    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    return args
