import os
import torch
import torch.nn as nn

import utils.utils as utils

def _normalize_vectors(vectors, eps=1e-8):
    return vectors / (vectors.norm(dim=-1, keepdim=True) + eps)


def semantic_boundary_alignment_loss(
        vector_alpha,
        vector_beta,
        semantic_grad_dir,
        surface_normal,
        weight,
        eps=1e-8):
    alpha = vector_alpha.squeeze(1)
    beta = vector_beta.squeeze(1)

    nb = _normalize_vectors(semantic_grad_dir, eps)
    ns = _normalize_vectors(surface_normal, eps)
    tb = torch.linalg.cross(ns, nb, dim=-1)
    tb = _normalize_vectors(tb, eps)

    score1 = (alpha * nb).sum(-1).abs() + (beta * tb).sum(-1).abs()
    score2 = (alpha * tb).sum(-1).abs() + (beta * nb).sum(-1).abs()
    score = 0.5 * torch.maximum(score1, score2)

    penalty = 1.0 - score
    return (weight * penalty).sum() / (weight.sum() + eps)


def semantic_neighbor_weight(labels_i, labels_j, gamma=0.2):
    same_part = labels_i.unsqueeze(-1) == labels_j
    return torch.where(
        same_part,
        torch.ones_like(labels_j, dtype=torch.float32),
        torch.full_like(labels_j, gamma, dtype=torch.float32)
    )


def semantic_neighbor_smoothness_loss(
        neighbors_term,
        semantic_labels_i,
        semantic_labels_j,
        gamma=0.2):
    weight = semantic_neighbor_weight(semantic_labels_i, semantic_labels_j, gamma=gamma)
    return (weight * neighbors_term).sum(), weight.sum()


def semantic_intra_consistency_loss(
        vector_alpha_i,
        vector_alpha_j,
        semantic_labels_i,
        semantic_labels_j):
    alpha_i = vector_alpha_i.squeeze(1).squeeze(1)
    alpha_j = vector_alpha_j.squeeze(2)
    same_part = semantic_labels_i.unsqueeze(-1) == semantic_labels_j

    similarity = (alpha_i.unsqueeze(1) * alpha_j).sum(-1).abs()
    penalty = 1.0 - similarity
    same_part_weight = same_part.to(penalty.dtype)
    return (same_part_weight * penalty).sum(), same_part_weight.sum()

def eikonal_loss(nonmnfld_grad, mnfld_grad, eikonal_type='abs'):
    # Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
    # shape is (bs, num_points, dim=3) for both grads
    # Eikonal
    if nonmnfld_grad is not None and mnfld_grad is not None:
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad

    if eikonal_type == 'abs':
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).abs()).mean()
    else:
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).square()).mean()
    # eikonal_term = (-torch.log(all_grads.norm(2, dim=2))).mean()
    return eikonal_term

class MorseLoss_quad_mesh(nn.Module):
    def __init__(self, weights=None, loss_type='siren_wo_n_w_morse', div_decay='none',
                 div_type='l1', vertex_neighbors_list=None,
                 vertex_neighbors=None, axis_angle_R_mat_list=None, device=None,
                 semantic_grad_dir=None, semantic_grad_weight=None, semantic_labels=None,
                 semantic_boundary_weight=1.0, semantic_intra_weight=1.0,
                 semantic_neighbor_weight=1.0, semantic_cross_part_gamma=0.2):
        super().__init__()
        if weights is None:
            weights = [7e3, 6e2, 10, 5e1, 30, 3, 20]
        self.weights = weights  # sdf, intern, normal, eikonal, div
        self.loss_type = loss_type
        self.div_decay = div_decay
        self.div_type = div_type
        self.use_morse = True if 'morse' in self.loss_type else False
        self.vertex_neighbors_list = vertex_neighbors_list
        self.vertex_neighbors = vertex_neighbors
        self.axis_angle_R_mat_list = axis_angle_R_mat_list
        self.device = device
        self.semantic_grad_dir = semantic_grad_dir
        self.semantic_grad_weight = semantic_grad_weight
        self.semantic_labels = semantic_labels
        self.semantic_boundary_weight = semantic_boundary_weight
        self.semantic_intra_weight = semantic_intra_weight
        self.semantic_neighbor_weight = semantic_neighbor_weight
        self.semantic_cross_part_gamma = semantic_cross_part_gamma



    def forward(self, output_pred, mnfld_points, nonmnfld_points, mnfld_n_gt=None, near_points=None, batch_idx=0,
                logdir=None, filename=None, save_best=False, mnfld_pts_theta_output_pred=None, local_coord_u=None,
                local_coord_v=None):

        dims = mnfld_points.shape[-1]

        #########################################
        # Compute required terms
        #########################################

        non_manifold_pred = output_pred["nonmanifold_pnts_pred"]
        manifold_pred = output_pred["manifold_pnts_pred"]
        surface_normal = mnfld_n_gt.squeeze(0)

        morse_loss = torch.tensor([0.0], device=self.device)
        normal_term = torch.tensor([0.0], device=self.device)
        eikonal_term = torch.tensor([0.0], device=self.device)
        mnfld_hessian_term = torch.tensor([0.0], device=self.device)

        # compute gradients for div (divergence), curl and curv (curvature)
        if manifold_pred is not None:
            mnfld_grad = utils.gradient(mnfld_points, manifold_pred)
        else:
            mnfld_grad = None

        nonmnfld_grad = utils.gradient(nonmnfld_points, non_manifold_pred)

        morse_nonmnfld_grad = None
        if self.use_morse and near_points is not None:
            morse_nonmnfld_grad = utils.gradient(near_points, output_pred['near_points_pred'])
        elif self.use_morse and near_points is None:
            morse_nonmnfld_grad = nonmnfld_grad

        if self.use_morse:
            mnfld_dx = utils.gradient(mnfld_points, mnfld_grad[:, :, 0])
            mnfld_dy = utils.gradient(mnfld_points, mnfld_grad[:, :, 1])
            if dims == 3:
                mnfld_dz = utils.gradient(mnfld_points, mnfld_grad[:, :, 2])
                mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy, mnfld_dz), dim=-1)
            else:
                mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy), dim=-1)

            morse_mnfld = torch.tensor([0.0], device=self.device)
            if self.div_type == 'l1':
                mnfld_n_gt = mnfld_n_gt.permute(1, 0, 2)
                mnfld_hessian_term = mnfld_hessian_term.squeeze(0)

                morse_mnfld = torch.bmm(mnfld_n_gt, mnfld_hessian_term)
                morse_mnfld = morse_mnfld.abs().mean()

            morse_loss = 0.5 * morse_mnfld

        sdf_term = torch.abs(manifold_pred).mean()

        # eikonal term
        eikonal_term = eikonal_loss(morse_nonmnfld_grad, mnfld_grad=mnfld_grad, eikonal_type='abs')

        # inter term
        inter_term = torch.exp(-1e2 * torch.abs(non_manifold_pred)).mean()

        # theta_term
        local_coord_u = local_coord_u.squeeze(0)
        local_coord_v = local_coord_v.squeeze(0)

        mnfld_pts_theta_output_pred = mnfld_pts_theta_output_pred.squeeze(0)

        vector_alpha = local_coord_u * torch.cos(mnfld_pts_theta_output_pred) + local_coord_v * torch.sin(
            mnfld_pts_theta_output_pred)
        vector_alpha = vector_alpha / (vector_alpha.norm(dim=-1, keepdim=True) + 1e-12)
        vector_alpha = vector_alpha.unsqueeze(1)

        vector_beta = -local_coord_u * torch.sin(mnfld_pts_theta_output_pred) + local_coord_v * torch.cos(
            mnfld_pts_theta_output_pred)
        vector_beta = vector_beta / (vector_beta.norm(dim=-1, keepdim=True) + 1e-12)
        vector_beta = vector_beta.unsqueeze(1)

        theta_hessian_term = torch.tensor([0.0], device=self.device)
        theta_neighbors_term = torch.tensor([0.0], device=self.device)
        semantic_boundary_term = torch.tensor([0.0], device=self.device)
        semantic_intra_term = torch.tensor([0.0], device=self.device)
        semantic_neighbor_term = torch.tensor([0.0], device=self.device)
        semantic_loss = torch.tensor([0.0], device=self.device)
        semantic_neighbor_numerator = torch.tensor([0.0], device=self.device)
        semantic_neighbor_denominator = torch.tensor([0.0], device=self.device)
        semantic_intra_numerator = torch.tensor([0.0], device=self.device)
        semantic_intra_denominator = torch.tensor([0.0], device=self.device)

        if self.semantic_grad_dir is not None and self.semantic_grad_weight is not None:
            semantic_boundary_term = semantic_boundary_alignment_loss(
                vector_alpha,
                vector_beta,
                self.semantic_grad_dir,
                surface_normal,
                self.semantic_grad_weight
            )
        for i in range(len(self.vertex_neighbors_list)):
            idx = torch.tensor(self.vertex_neighbors_list[i]).to(self.device)

            # theta_hessian_term
            hessian_i = mnfld_hessian_term[idx].unsqueeze(1)  # n x 1 x 3 x 3
            vector_alpha_i = vector_alpha[idx].unsqueeze(1)  # n x 1 x 1 x 3
            vector_beta_i = vector_beta[idx].unsqueeze(1)  # n x 1 x 1 x 3

            vertex_h_term_alpha = torch.matmul(vector_alpha_i, hessian_i)
            vertex_h_term_alpha = torch.linalg.cross(vertex_h_term_alpha, vector_alpha_i)
            vertex_h_term_alpha = vertex_h_term_alpha.abs().mean()

            vertex_h_term_beta = torch.matmul(vector_beta_i, hessian_i)
            vertex_h_term_beta = torch.linalg.cross(vertex_h_term_beta, vector_beta_i)
            vertex_h_term_beta = vertex_h_term_beta.abs().mean()

            vertex_h_term = 0.5 * (vertex_h_term_alpha + vertex_h_term_beta)
            theta_hessian_term += vertex_h_term

            # theta_neighbors_term
            vertex_neighbors_i = [self.vertex_neighbors[z] for z in idx]
            vertex_neighbors_i = torch.tensor(vertex_neighbors_i, dtype=torch.long)  # n x neighbors_size
            vector_alpha_j = vector_alpha[vertex_neighbors_i]  # n x neighbors_size x 1 x 3
            vector_beta_j = vector_beta[vertex_neighbors_i]  # n x neighbors_size x 1 x 3

            if self.axis_angle_R_mat_list is not None:
                axis_angle_R_mat_i = torch.tensor(self.axis_angle_R_mat_list[i]).to(self.device)
                vector_alpha_j = utils.transform_vectors_only_rotation(vector_alpha_j, axis_angle_R_mat_i)
                vector_beta_j = utils.transform_vectors_only_rotation(vector_beta_j, axis_angle_R_mat_i)

            neighbors_term_alpha_alpha = torch.matmul(vector_alpha_i, vector_alpha_j.permute(0, 1, 3, 2)).abs()
            neighbors_term_alpha_beta = torch.matmul(vector_alpha_i, vector_beta_j.permute(0, 1, 3, 2)).abs()
            neighbors_term_beta_alpha = torch.matmul(vector_beta_i, vector_alpha_j.permute(0, 1, 3, 2)).abs()
            neighbors_term_beta_beta = torch.matmul(vector_beta_i, vector_beta_j.permute(0, 1, 3, 2)).abs()
            raw_neighbors_term = (
                neighbors_term_alpha_alpha +
                neighbors_term_alpha_beta +
                neighbors_term_beta_alpha +
                neighbors_term_beta_beta - 2
            ).squeeze(-1).squeeze(-1)
            neighbors_term = raw_neighbors_term.mean()  # the sum value is greater than 2

            theta_neighbors_term += neighbors_term

            if self.semantic_labels is not None:
                semantic_labels_i = self.semantic_labels[idx]
                semantic_labels_j = self.semantic_labels[vertex_neighbors_i.to(self.device)]

                neighbor_loss_sum, neighbor_weight_sum = semantic_neighbor_smoothness_loss(
                    raw_neighbors_term,
                    semantic_labels_i,
                    semantic_labels_j,
                    gamma=self.semantic_cross_part_gamma
                )
                semantic_neighbor_numerator += neighbor_loss_sum
                semantic_neighbor_denominator += neighbor_weight_sum

                intra_loss_sum, intra_weight_sum = semantic_intra_consistency_loss(
                    vector_alpha_i,
                    vector_alpha_j,
                    semantic_labels_i,
                    semantic_labels_j
                )
                semantic_intra_numerator += intra_loss_sum
                semantic_intra_denominator += intra_weight_sum

        num_vert_neigh = len(self.vertex_neighbors_list)
        theta_hessian_term = theta_hessian_term / num_vert_neigh
        theta_neighbors_term = theta_neighbors_term / num_vert_neigh

        if self.semantic_labels is not None:
            semantic_neighbor_term = semantic_neighbor_numerator / (semantic_neighbor_denominator + 1e-8)
            semantic_intra_term = semantic_intra_numerator / (semantic_intra_denominator + 1e-8)

        semantic_loss = (
            self.semantic_boundary_weight * semantic_boundary_term +
            self.semantic_intra_weight * semantic_intra_term +
            self.semantic_neighbor_weight * semantic_neighbor_term
        )

        # get the grad, curvature of the points
        if save_best:
            output_dir = os.path.join(logdir, 'save_crossField')
            utils.save_only_crossField(vector_alpha, vector_beta, batch_idx=batch_idx, output_dir=output_dir,
                                       shapename=filename)

        # losses used in the paper
        if self.loss_type == 'siren_wo_n_w_morse_w_theta':
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + self.weights[3] * eikonal_term + \
                   self.weights[5] * morse_loss + self.weights[2] * theta_hessian_term + self.weights[
                       4] * theta_neighbors_term + self.weights[6] * semantic_loss
        else:
            print(self.loss_type)
            raise Warning("unrecognized loss type")


        return {"loss": loss, 'sdf_term': sdf_term, 'inter_term': inter_term,
                'eikonal_term': eikonal_term, 'normals_loss': normal_term, 'morse_term': morse_loss,
                'theta_hessian_term': theta_hessian_term, 'theta_neighbors_term': theta_neighbors_term,
                'semantic_align_term': semantic_boundary_term, 'semantic_boundary_term': semantic_boundary_term,
                'semantic_intra_term': semantic_intra_term, 'semantic_neighbor_term': semantic_neighbor_term,
                'semantic_loss': semantic_loss}

    def update_morse_weight(self, current_iteration, n_iterations, params=None):
        # `params`` should be (start_weight, *optional middle, end_weight) where optional middle is of the form [percent, value]*
        # Thus (1e2, 0.5, 1e2 0.7 0.0, 0.0) means that the weight at [0, 0.5, 0.7, 1] of the training process, the weight should
        #   be [1e2,1e2,0.0,0.0]. Between these points, the weights change as per the div_decay parameter, e.g. linearly, quintic, step etc.
        #   Thus the weight stays at 1e2 from 0-0.5, decay from 1e2 to 0.0 from 0.5-0.75, and then stays at 0.0 from 0.75-1.

        if not hasattr(self, 'decay_params_list'):
            assert len(params) >= 2, params
            assert len(params[1:-1]) % 2 == 0
            self.decay_params_list = list(zip([params[0], *params[1:-1][1::2], params[-1]], [0, *params[1:-1][::2], 1]))

        curr = current_iteration / n_iterations
        we, e = min([tup for tup in self.decay_params_list if tup[1] >= curr], key=lambda tup: tup[1])
        w0, s = max([tup for tup in self.decay_params_list if tup[1] <= curr], key=lambda tup: tup[1])

        # Divergence term anealing functions
        if self.div_decay == 'linear':  # linearly decrease weight from iter s to iter e
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[5] = w0 + (we - w0) * (current_iteration / n_iterations - s) / (e - s)
            else:
                self.weights[5] = we
        elif self.div_decay == 'quintic':  # linearly decrease weight from iter s to iter e
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[5] = w0 + (we - w0) * (1 - (1 - (current_iteration / n_iterations - s) / (e - s)) ** 5)
            else:
                self.weights[5] = we
        elif self.div_decay == 'step':  # change weight at s
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            else:
                self.weights[5] = we
        elif self.div_decay == 'none':
            pass
        else:
            raise Warning("unsupported div decay value")
