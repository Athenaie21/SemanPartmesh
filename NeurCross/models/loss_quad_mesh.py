import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.utils as utils

def _normalize_vectors(vectors, eps=1e-8):
    return vectors / (vectors.norm(dim=-1, keepdim=True) + eps)


def semantic_boundary_alignment_loss(
        vector_alpha,
        vector_beta,
        semantic_grad_dir,
        surface_normal,
        weight,
        margin=0.0,
        eps=1e-8):
    """Alignment loss with optional margin and quadratic penalty.

    When margin > 0, faces whose alignment score exceeds (1 - margin) are
    not penalized, focusing gradient on the worst-aligned faces.
    """
    alpha = vector_alpha.squeeze(1)
    beta = vector_beta.squeeze(1)

    nb = _normalize_vectors(semantic_grad_dir, eps)
    ns = _normalize_vectors(surface_normal, eps)
    tb = torch.linalg.cross(ns, nb, dim=-1)
    tb = _normalize_vectors(tb, eps)

    score1 = (alpha * nb).sum(-1).abs() + (beta * tb).sum(-1).abs()
    score2 = (alpha * tb).sum(-1).abs() + (beta * nb).sum(-1).abs()
    score = 0.5 * torch.maximum(score1, score2)

    penalty = torch.relu(1.0 - score - margin) ** 2
    return (weight * penalty).sum() / (weight.sum() + eps)


def semantic_cross_part_diversity_loss(
        vector_alpha_i, vector_alpha_j,
        semantic_labels_i, semantic_labels_j,
        margin=0.3):
    """Encourage cross-field discontinuity across semantic boundaries.

    Penalizes cross-part neighbor pairs whose alpha vectors are too similar
    (above *margin*), forming a contrastive complement to the intra-part
    consistency loss.
    """
    alpha_i = vector_alpha_i.squeeze(1).squeeze(1)
    alpha_j = vector_alpha_j.squeeze(2)
    cross_part = (semantic_labels_i.unsqueeze(-1) != semantic_labels_j)
    similarity = (alpha_i.unsqueeze(1) * alpha_j).sum(-1).abs()
    violation = torch.relu(similarity - margin)
    cross_weight = cross_part.to(violation.dtype)
    return (cross_weight * violation).sum(), cross_weight.sum()


def semantic_soft_boundary_weight(
        features_i,
        features_j,
        temperature=0.1,
        chunk_size=0,
        eps=1e-8):
    """Soft boundary weight from feature cosine similarity.

    High similarity -> weight ~1 (enforce smoothness).
    Low similarity  -> weight ~0 (allow discontinuity).
    """
    if features_i.shape != features_j.shape:
        raise ValueError(
            f"Feature pair shapes must match, got {features_i.shape} vs {features_j.shape}")

    if chunk_size is None or int(chunk_size) <= 0:
        sim = F.cosine_similarity(features_i, features_j, dim=-1, eps=eps)
        return torch.sigmoid(sim / temperature)

    original_shape = features_i.shape[:-1]
    flat_i = features_i.reshape(-1, features_i.shape[-1])
    flat_j = features_j.reshape(-1, features_j.shape[-1])
    if flat_i.shape[0] == 0:
        return torch.empty(original_shape, dtype=features_i.dtype, device=features_i.device)

    sim_chunks = []
    chunk_size = int(chunk_size)
    for start in range(0, flat_i.shape[0], chunk_size):
        end = min(start + chunk_size, flat_i.shape[0])
        sim_chunk = F.cosine_similarity(
            flat_i[start:end],
            flat_j[start:end],
            dim=-1,
            eps=eps,
        )
        sim_chunks.append(sim_chunk)

    sim = torch.cat(sim_chunks, dim=0).reshape(original_shape)
    return torch.sigmoid(sim / temperature)


def semantic_neighbor_weight(labels_i, labels_j, gamma=0.2):
    same_part = labels_i.unsqueeze(-1) == labels_j
    return torch.where(
        same_part,
        torch.ones_like(labels_j, dtype=torch.float32),
        torch.full_like(labels_j, gamma, dtype=torch.float32)
    )


def semantic_boundary_mask(labels_i, labels_j):
    """Returns 1.0 for same-part pairs, 0.0 for cross-part pairs."""
    same_part = labels_i.unsqueeze(-1) == labels_j
    return same_part.to(torch.float32)


def semantic_neighbor_smoothness_loss(
        neighbors_term,
        semantic_labels_i,
        semantic_labels_j,
        gamma=0.2,
        boundary_reward=0.0):
    same_part = semantic_labels_i.unsqueeze(-1) == semantic_labels_j
    cross_part_weight = boundary_reward if boundary_reward != 0.0 else gamma
    weight = torch.where(
        same_part,
        torch.ones_like(semantic_labels_j, dtype=torch.float32),
        torch.full_like(semantic_labels_j, cross_part_weight, dtype=torch.float32)
    )
    return (weight * neighbors_term).sum(), weight.abs().sum()


def semantic_intra_consistency_loss(
        vector_alpha_i,
        vector_alpha_j,
        semantic_labels_i,
        semantic_labels_j,
        temperature=1.0,
        hard_ratio=0.0):
    alpha_i = vector_alpha_i.squeeze(1).squeeze(1)
    alpha_j = vector_alpha_j.squeeze(2)
    same_part = semantic_labels_i.unsqueeze(-1) == semantic_labels_j

    similarity = (alpha_i.unsqueeze(1) * alpha_j).sum(-1).abs()
    if temperature != 1.0 and temperature > 0:
        penalty = 1.0 - similarity.pow(temperature)
    else:
        penalty = 1.0 - similarity
    same_part_weight = same_part.to(penalty.dtype)
    weighted_penalty = same_part_weight * penalty

    if hard_ratio > 0:
        valid_mask = same_part_weight > 0
        valid_penalties = weighted_penalty[valid_mask]
        if valid_penalties.numel() > 0:
            k = max(1, int(valid_penalties.numel() * hard_ratio))
            topk_vals, _ = valid_penalties.topk(k)
            return topk_vals.sum(), torch.tensor(float(k), device=penalty.device)
    return weighted_penalty.sum(), same_part_weight.sum()


def instruction_instance_consistency_loss(
        vector_alpha_i,
        vector_alpha_j,
        instance_labels_i,
        instance_labels_j):
    """Intra-instance cross-field consistency: penalize 1 - |dot(alpha_i, alpha_j)|
    for same-instance neighbor pairs."""
    alpha_i = vector_alpha_i.squeeze(1).squeeze(1)
    alpha_j = vector_alpha_j.squeeze(2)
    same_instance = instance_labels_i.unsqueeze(-1) == instance_labels_j

    similarity = (alpha_i.unsqueeze(1) * alpha_j).sum(-1).abs()
    penalty = 1.0 - similarity
    same_instance_weight = same_instance.to(penalty.dtype)
    weighted_penalty = same_instance_weight * penalty
    return weighted_penalty.sum(), same_instance_weight.sum()


# ---------------------------------------------------------------------------
# Legacy instruction loss helpers (kept for backward compatibility / eval)
# ---------------------------------------------------------------------------

def instruction_type_face_weight(feature_type, location):
    weight = torch.ones_like(feature_type, dtype=torch.float32)
    weight = torch.where(feature_type == 1, torch.full_like(weight, 1.15), weight)
    weight = torch.where(feature_type == 4, torch.full_like(weight, 1.10), weight)
    weight = torch.where(feature_type == 2, torch.full_like(weight, 0.80), weight)
    weight = torch.where(feature_type == 3, torch.full_like(weight, 0.70), weight)
    end_mask = (location == 2) | (location == 3)
    weight = torch.where(end_mask, weight * 0.90, weight)
    return weight


def instruction_neighbor_relaxation_loss(
        neighbors_term, instance_labels_i, instance_labels_j, gamma=0.1):
    same_instance = instance_labels_i.unsqueeze(-1) == instance_labels_j
    weight = torch.where(
        same_instance,
        torch.ones_like(neighbors_term, dtype=torch.float32),
        torch.full_like(neighbors_term, gamma, dtype=torch.float32)
    )
    return (weight * neighbors_term).sum(), weight.sum()


def instruction_type_prior_loss(
        neighbors_term, instance_labels_i, instance_labels_j,
        feature_type_i, feature_type_j, location_i, location_j):
    same_instance = instance_labels_i.unsqueeze(-1) == instance_labels_j
    weight_i = instruction_type_face_weight(feature_type_i, location_i).unsqueeze(-1)
    weight_j = instruction_type_face_weight(feature_type_j, location_j)
    pair_weight = 0.5 * (weight_i + weight_j)
    pair_weight = pair_weight * same_instance.to(pair_weight.dtype)
    return (pair_weight * neighbors_term).sum(), pair_weight.sum()


def instruction_direction_alignment_loss(
        vector_alpha, vector_beta, target_dir, surface_normal,
        face_weight, eps=1e-8):
    """Align cross-field with a per-face target tangent direction."""
    alpha = vector_alpha.squeeze(1)
    beta = vector_beta.squeeze(1)
    d = _normalize_vectors(target_dir, eps)
    ns = _normalize_vectors(surface_normal, eps)
    d_perp = torch.linalg.cross(ns, d, dim=-1)
    d_perp = _normalize_vectors(d_perp, eps)
    score1 = (alpha * d).sum(-1).abs() + (beta * d_perp).sum(-1).abs()
    score2 = (alpha * d_perp).sum(-1).abs() + (beta * d).sum(-1).abs()
    score = 0.5 * torch.maximum(score1, score2)
    penalty = torch.relu(1.0 - score) ** 2
    w = face_weight
    denom = w.sum() + eps
    return (w * penalty).sum() / denom


def instruction_label_neighbor_loss(
        neighbors_term, labels_i, labels_j, gamma=0.1):
    same_label = labels_i.unsqueeze(-1) == labels_j
    weight = torch.where(
        same_label,
        torch.ones_like(neighbors_term, dtype=torch.float32),
        torch.full_like(neighbors_term, gamma, dtype=torch.float32)
    )
    return (weight * neighbors_term).sum(), weight.sum()


def instruction_region_consistency_loss(
        raw_neighbors_term, face_weight_i, face_weight_j):
    pair_weight = 0.5 * (face_weight_i.unsqueeze(-1) + face_weight_j)
    return (pair_weight * raw_neighbors_term).sum(), pair_weight.sum()


def instruction_singularity_sparsity_loss(
        theta_hessian_per_face, allowed_weight, eps=1e-8):
    penalty_weight = torch.clamp(1.0 - allowed_weight, min=0.0)
    return (penalty_weight * theta_hessian_per_face).sum(), penalty_weight.sum()


def instruction_prototype_loss(
        alpha_similarity, prototype_intra, prototype_boundary,
        same_instance_mask, eps=1e-8):
    intra_target = prototype_intra.unsqueeze(-1).expand_as(alpha_similarity)
    boundary_target = prototype_boundary.unsqueeze(-1).expand_as(alpha_similarity)
    target = torch.where(same_instance_mask, intra_target, boundary_target)
    error = (alpha_similarity - target) ** 2
    return error.sum(), torch.ones(1, device=error.device) * max(error.numel(), 1)


# ---------------------------------------------------------------------------
# Core losses
# ---------------------------------------------------------------------------

def eikonal_loss(nonmnfld_grad, mnfld_grad, eikonal_type='abs'):
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
    return eikonal_term

class MorseLoss_quad_mesh(nn.Module):
    def __init__(self, weights=None, loss_type='siren_wo_n_w_morse', div_decay='none',
                 div_type='l1', vertex_neighbors_list=None,
                 vertex_neighbors=None, axis_angle_R_mat_list=None, device=None,
                 # --- Feature mode params (untouched) ---
                 semantic_grad_dir=None, semantic_grad_weight=None, semantic_labels=None,
                 semantic_boundary_weight=1.0, semantic_intra_weight=1.0,
                 semantic_neighbor_weight=1.0, semantic_cross_part_gamma=0.2,
                 semantic_boundary_reward=0.0,
                 semantic_features=None,
                 semantic_diversity_weight=0.5,
                 semantic_diversity_margin=0.3,
                 semantic_soft_boundary_temp=0.1,
                 semantic_soft_boundary_chunk_size=1024,
                 alignment_margin=0.0,
                 guidance_warmup_fraction=0.0,
                 guidance_mode='none',
                 # --- Instruction mode params ---
                 instruction_instance_labels=None,
                 instruction_boundary_grad_dir=None,
                 instruction_boundary_grad_weight=None,
                 instruction_boundary_weight=1.5,
                 instruction_intra_weight=1.0,
                 instruction_cross_instance_gamma=0.4,
                 instruction_operation_align_weight=1.0,
                 instruction_anchor_weight=0.3,
                 instruction_extrude_dir=None,
                 instruction_extrude_face_weight=None,
                 instruction_revolve_dir=None,
                 instruction_revolve_face_weight=None,
                 instruction_chamfer_dir=None,
                 instruction_chamfer_face_weight=None,
                 instruction_fillet_dir=None,
                 instruction_fillet_face_weight=None,
                 instruction_anchor_dir=None,
                 instruction_anchor_face_weight=None,
                 # --- Safeguards ---
                 guidance_cap_ratio=0.5,
                 guidance_eikonal_guard=0.0,
                 guidance_warmup_type='linear',
                 # --- Feature mode extras (untouched) ---
                 intra_temperature=1.0,
                 intra_hard_ratio=0.0,
                 # --- Legacy kwargs (accepted but ignored for backward compat) ---
                 **kwargs):
        super().__init__()
        if weights is None:
            weights = [7e3, 6e2, 10, 5e1, 30, 3, 50]
        self.weights = weights
        self.loss_type = loss_type
        self.div_decay = div_decay
        self.div_type = div_type
        self.use_morse = True if 'morse' in self.loss_type else False
        self.vertex_neighbors_list = vertex_neighbors_list
        self.vertex_neighbors = vertex_neighbors
        self.axis_angle_R_mat_list = axis_angle_R_mat_list
        self.device = device

        # Feature mode state (untouched)
        self.semantic_grad_dir = semantic_grad_dir
        self.semantic_grad_weight = semantic_grad_weight
        self.semantic_labels = semantic_labels
        self.semantic_boundary_weight = semantic_boundary_weight
        self.semantic_intra_weight = semantic_intra_weight
        self.semantic_neighbor_weight = semantic_neighbor_weight
        self.semantic_cross_part_gamma = semantic_cross_part_gamma
        self.semantic_boundary_reward = semantic_boundary_reward
        self.semantic_features = semantic_features
        self.semantic_diversity_weight = semantic_diversity_weight
        self.semantic_diversity_margin = semantic_diversity_margin
        self.semantic_soft_boundary_temp = semantic_soft_boundary_temp
        self.semantic_soft_boundary_chunk_size = semantic_soft_boundary_chunk_size
        self.alignment_margin = alignment_margin
        self.intra_temperature = intra_temperature
        self.intra_hard_ratio = intra_hard_ratio

        # Guidance scheduling
        self.guidance_warmup_fraction = guidance_warmup_fraction
        self.guidance_target_weight = weights[6] if len(weights) > 6 else 50.0
        self.guidance_mode = guidance_mode
        self.guidance_cap_ratio = guidance_cap_ratio
        self.guidance_eikonal_guard = guidance_eikonal_guard
        self.guidance_warmup_type = guidance_warmup_type

        # Instruction mode state
        self.instruction_instance_labels = instruction_instance_labels
        self.instruction_boundary_grad_dir = instruction_boundary_grad_dir
        self.instruction_boundary_grad_weight = instruction_boundary_grad_weight
        self.instruction_boundary_weight = instruction_boundary_weight
        self.instruction_intra_weight = instruction_intra_weight
        self.instruction_cross_instance_gamma = instruction_cross_instance_gamma
        self.instruction_operation_align_weight = instruction_operation_align_weight
        self.instruction_anchor_weight = instruction_anchor_weight
        self.instruction_extrude_dir = instruction_extrude_dir
        self.instruction_extrude_face_weight = instruction_extrude_face_weight
        self.instruction_revolve_dir = instruction_revolve_dir
        self.instruction_revolve_face_weight = instruction_revolve_face_weight
        self.instruction_chamfer_dir = instruction_chamfer_dir
        self.instruction_chamfer_face_weight = instruction_chamfer_face_weight
        self.instruction_fillet_dir = instruction_fillet_dir
        self.instruction_fillet_face_weight = instruction_fillet_face_weight
        self.instruction_anchor_dir = instruction_anchor_dir
        self.instruction_anchor_face_weight = instruction_anchor_face_weight

    def forward(self, output_pred, mnfld_points, nonmnfld_points, mnfld_n_gt=None, near_points=None, batch_idx=0,
                logdir=None, filename=None, save_best=False, mnfld_pts_theta_output_pred=None, local_coord_u=None,
                local_coord_v=None):

        dims = mnfld_points.shape[-1]

        non_manifold_pred = output_pred["nonmanifold_pnts_pred"]
        manifold_pred = output_pred["manifold_pnts_pred"]
        surface_normal = mnfld_n_gt.squeeze(0)

        morse_loss = torch.tensor([0.0], device=self.device)
        normal_term = torch.tensor([0.0], device=self.device)
        eikonal_term = torch.tensor([0.0], device=self.device)
        mnfld_hessian_term = torch.tensor([0.0], device=self.device)

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

        eikonal_term = eikonal_loss(morse_nonmnfld_grad, mnfld_grad=mnfld_grad, eikonal_type='abs')

        inter_term = torch.exp(-1e2 * torch.abs(non_manifold_pred)).mean()

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

        _zero = torch.tensor([0.0], device=self.device)
        theta_hessian_term = _zero.clone()
        theta_neighbors_term = _zero.clone()

        # Feature mode accumulators (untouched)
        semantic_boundary_term = _zero.clone()
        semantic_intra_term = _zero.clone()
        semantic_neighbor_term = _zero.clone()
        semantic_loss = _zero.clone()
        semantic_neighbor_numerator = _zero.clone()
        semantic_neighbor_denominator = _zero.clone()
        semantic_intra_numerator = _zero.clone()
        semantic_intra_denominator = _zero.clone()
        semantic_diversity_term = _zero.clone()
        semantic_diversity_numerator = _zero.clone()
        semantic_diversity_denominator = _zero.clone()

        # Instruction mode accumulators
        instruction_boundary_align_term = _zero.clone()
        instruction_intra_term = _zero.clone()
        instruction_intra_num = _zero.clone()
        instruction_intra_den = _zero.clone()
        instruction_operation_align_term = _zero.clone()
        instruction_anchor_term = _zero.clone()
        instruction_loss = _zero.clone()

        # --- Boundary alignment (computed outside the neighbor loop) ---
        if self.guidance_mode == 'feature' and self.semantic_grad_dir is not None and self.semantic_grad_weight is not None:
            semantic_boundary_term = semantic_boundary_alignment_loss(
                vector_alpha,
                vector_beta,
                self.semantic_grad_dir,
                surface_normal,
                self.semantic_grad_weight,
                margin=self.alignment_margin,
            )

        if self.guidance_mode == 'instruction' and self.instruction_boundary_grad_dir is not None:
            instruction_boundary_align_term = semantic_boundary_alignment_loss(
                vector_alpha,
                vector_beta,
                self.instruction_boundary_grad_dir,
                surface_normal,
                self.instruction_boundary_grad_weight,
            )

        # --- Operation direction alignment (extrude/revolve/chamfer/fillet) ---
        if self.guidance_mode == 'instruction':
            op_align_sum = _zero.clone()
            op_align_count = 0
            for op_dir, op_w in [
                (self.instruction_extrude_dir, self.instruction_extrude_face_weight),
                (self.instruction_revolve_dir, self.instruction_revolve_face_weight),
                (self.instruction_chamfer_dir, self.instruction_chamfer_face_weight),
                (self.instruction_fillet_dir, self.instruction_fillet_face_weight),
            ]:
                if op_dir is not None and op_w is not None and op_w.sum() > 1e-8:
                    op_align_sum = op_align_sum + instruction_direction_alignment_loss(
                        vector_alpha, vector_beta, op_dir, surface_normal, op_w)
                    op_align_count += 1
            if op_align_count > 0:
                instruction_operation_align_term = op_align_sum / op_align_count

            if self.instruction_anchor_dir is not None and self.instruction_anchor_face_weight is not None:
                if self.instruction_anchor_face_weight.sum() > 1e-8:
                    instruction_anchor_term = instruction_direction_alignment_loss(
                        vector_alpha, vector_beta,
                        self.instruction_anchor_dir, surface_normal,
                        self.instruction_anchor_face_weight)

        # --- Neighbor loop: hessian + neighbors + intra-consistency ---
        valid_neighbor_groups = 0
        for i in range(len(self.vertex_neighbors_list)):
            idx = torch.tensor(self.vertex_neighbors_list[i]).to(self.device)
            if idx.numel() == 0:
                continue

            hessian_i = mnfld_hessian_term[idx].unsqueeze(1)
            vector_alpha_i = vector_alpha[idx].unsqueeze(1)
            vector_beta_i = vector_beta[idx].unsqueeze(1)

            vertex_h_term_alpha_raw = torch.matmul(vector_alpha_i, hessian_i)
            vertex_h_term_alpha_raw = torch.linalg.cross(vertex_h_term_alpha_raw, vector_alpha_i)
            vertex_h_term_alpha = vertex_h_term_alpha_raw.abs().squeeze(1).squeeze(1).mean(dim=-1).mean()

            vertex_h_term_beta_raw = torch.matmul(vector_beta_i, hessian_i)
            vertex_h_term_beta_raw = torch.linalg.cross(vertex_h_term_beta_raw, vector_beta_i)
            vertex_h_term_beta = vertex_h_term_beta_raw.abs().squeeze(1).squeeze(1).mean(dim=-1).mean()

            vertex_h_term = 0.5 * (vertex_h_term_alpha + vertex_h_term_beta)
            theta_hessian_term += vertex_h_term

            vertex_neighbors_i = [self.vertex_neighbors[z] for z in idx]
            vertex_neighbors_i = torch.tensor(vertex_neighbors_i, dtype=torch.long)
            vector_alpha_j = vector_alpha[vertex_neighbors_i]
            vector_beta_j = vector_beta[vertex_neighbors_i]

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
            raw_neighbors_term = torch.clamp(raw_neighbors_term, min=0.0)
            if raw_neighbors_term.numel() == 0:
                continue
            raw_neighbors_term = torch.where(
                torch.isfinite(raw_neighbors_term),
                raw_neighbors_term,
                torch.zeros_like(raw_neighbors_term)
            )
            valid_neighbor_groups += 1

            # ============================================================
            # Feature mode: neighbor weighting (UNTOUCHED)
            # ============================================================
            if self.guidance_mode == 'feature' and self.semantic_labels is not None:
                semantic_labels_i = self.semantic_labels[idx]
                semantic_labels_j = self.semantic_labels[vertex_neighbors_i.to(self.device)]

                mask = semantic_boundary_mask(semantic_labels_i, semantic_labels_j)
                masked_sum = (mask * raw_neighbors_term).sum()
                masked_count = mask.sum() + 1e-8
                theta_neighbors_term += masked_sum / masked_count

                neighbor_loss_sum, neighbor_weight_sum = semantic_neighbor_smoothness_loss(
                    raw_neighbors_term,
                    semantic_labels_i,
                    semantic_labels_j,
                    gamma=self.semantic_cross_part_gamma,
                    boundary_reward=self.semantic_boundary_reward,
                )
                semantic_neighbor_numerator += neighbor_loss_sum
                semantic_neighbor_denominator += neighbor_weight_sum

                intra_loss_sum, intra_weight_sum = semantic_intra_consistency_loss(
                    vector_alpha_i,
                    vector_alpha_j,
                    semantic_labels_i,
                    semantic_labels_j,
                    temperature=self.intra_temperature,
                    hard_ratio=self.intra_hard_ratio,
                )
                semantic_intra_numerator += intra_loss_sum
                semantic_intra_denominator += intra_weight_sum

                div_loss_sum, div_weight_sum = semantic_cross_part_diversity_loss(
                    vector_alpha_i,
                    vector_alpha_j,
                    semantic_labels_i,
                    semantic_labels_j,
                    margin=self.semantic_diversity_margin,
                )
                semantic_diversity_numerator += div_loss_sum
                semantic_diversity_denominator += div_weight_sum

                if self.semantic_features is not None:
                    feat_i = self.semantic_features[idx]
                    feat_j = self.semantic_features[vertex_neighbors_i.to(self.device)]
                    soft_w = semantic_soft_boundary_weight(
                        feat_i.unsqueeze(1).expand_as(feat_j),
                        feat_j,
                        temperature=self.semantic_soft_boundary_temp,
                        chunk_size=self.semantic_soft_boundary_chunk_size,
                    )
                    soft_neighbors = (soft_w * raw_neighbors_term).sum()
                    soft_count = soft_w.sum() + 1e-8
                    theta_neighbors_term += soft_neighbors / soft_count
                    theta_neighbors_term -= masked_sum / masked_count

            # ============================================================
            # Instruction mode: soft cross-instance smoothness + intra
            # ============================================================
            elif self.guidance_mode == 'instruction' and self.instruction_instance_labels is not None:
                instance_labels_i = self.instruction_instance_labels[idx]
                instance_labels_j = self.instruction_instance_labels[vertex_neighbors_i.to(self.device)]
                same_inst = (instance_labels_i.unsqueeze(-1) == instance_labels_j).float()

                # Soft-weighted theta_neighbors: same-instance=1, cross-instance=gamma
                gamma = self.instruction_cross_instance_gamma
                inst_mask = torch.where(
                    same_inst > 0,
                    torch.ones_like(same_inst),
                    torch.full_like(same_inst, gamma),
                )
                inst_masked_sum = (inst_mask * raw_neighbors_term).sum()
                inst_masked_count = inst_mask.sum() + 1e-8
                theta_neighbors_term += inst_masked_sum / inst_masked_count

                # Intra-instance consistency
                intra_loss_sum, intra_weight_sum = instruction_instance_consistency_loss(
                    vector_alpha_i,
                    vector_alpha_j,
                    instance_labels_i,
                    instance_labels_j,
                )
                instruction_intra_num += intra_loss_sum
                instruction_intra_den += intra_weight_sum

            # ============================================================
            # No guidance: uniform neighbor smoothness
            # ============================================================
            else:
                theta_neighbors_term += raw_neighbors_term.mean()

        num_vert_neigh = max(valid_neighbor_groups, 1)
        theta_hessian_term = theta_hessian_term / num_vert_neigh
        theta_neighbors_term = theta_neighbors_term / num_vert_neigh

        # --- Feature mode loss aggregation (UNTOUCHED) ---
        if self.guidance_mode == 'feature' and self.semantic_labels is not None:
            semantic_neighbor_term = semantic_neighbor_numerator / (semantic_neighbor_denominator + 1e-8)
            semantic_intra_term = semantic_intra_numerator / (semantic_intra_denominator + 1e-8)
            semantic_diversity_term = semantic_diversity_numerator / (semantic_diversity_denominator + 1e-8)

        semantic_loss = (
            self.semantic_boundary_weight * semantic_boundary_term +
            self.semantic_intra_weight * semantic_intra_term +
            self.semantic_neighbor_weight * semantic_neighbor_term +
            self.semantic_diversity_weight * semantic_diversity_term
        )

        # --- Instruction mode loss aggregation ---
        if self.guidance_mode == 'instruction' and self.instruction_instance_labels is not None:
            instruction_intra_term = instruction_intra_num / (instruction_intra_den + 1e-8)
            instruction_loss = (
                self.instruction_boundary_weight * instruction_boundary_align_term +
                self.instruction_intra_weight * instruction_intra_term +
                self.instruction_operation_align_weight * instruction_operation_align_term +
                self.instruction_anchor_weight * instruction_anchor_term
            )

        guidance_loss = semantic_loss if self.guidance_mode == 'feature' else instruction_loss

        if save_best:
            output_dir = os.path.join(logdir, 'save_crossField')
            utils.save_only_crossField(vector_alpha, vector_beta, batch_idx=batch_idx, output_dir=output_dir,
                                       shapename=filename)

        if self.loss_type == 'siren_wo_n_w_morse_w_theta':
            geo_loss = (self.weights[0] * sdf_term + self.weights[1] * inter_term +
                        self.weights[3] * eikonal_term + self.weights[5] * morse_loss +
                        self.weights[2] * theta_hessian_term + self.weights[4] * theta_neighbors_term)

            if len(self.weights) > 6 and self.guidance_mode != 'none':
                effective_guidance_weight = self.weights[6]

                if self.guidance_eikonal_guard > 0 and eikonal_term.item() > self.guidance_eikonal_guard:
                    ratio = self.guidance_eikonal_guard / max(eikonal_term.item(), 1e-8)
                    effective_guidance_weight = effective_guidance_weight * ratio

                weighted_guidance = effective_guidance_weight * guidance_loss
                if self.guidance_cap_ratio > 0:
                    geo_detached = geo_loss.detach().item()
                    cap = self.guidance_cap_ratio * max(geo_detached, 1.0)
                    wg = weighted_guidance.detach().item()
                    if wg > cap:
                        weighted_guidance = weighted_guidance * (cap / max(wg, 1e-8))

                loss = geo_loss + weighted_guidance
            else:
                loss = geo_loss
        else:
            print(self.loss_type)
            raise Warning("unrecognized loss type")

        return {"loss": loss, 'sdf_term': sdf_term, 'inter_term': inter_term,
                'eikonal_term': eikonal_term, 'normals_loss': normal_term, 'morse_term': morse_loss,
                'theta_hessian_term': theta_hessian_term, 'theta_neighbors_term': theta_neighbors_term,
                'semantic_align_term': semantic_boundary_term, 'semantic_boundary_term': semantic_boundary_term,
                'semantic_intra_term': semantic_intra_term, 'semantic_neighbor_term': semantic_neighbor_term,
                'semantic_diversity_term': semantic_diversity_term,
                'semantic_loss': semantic_loss, 'guidance_loss': guidance_loss,
                'instruction_boundary_align_term': instruction_boundary_align_term,
                'instruction_intra_term': instruction_intra_term,
                'instruction_operation_align_term': instruction_operation_align_term,
                'instruction_anchor_term': instruction_anchor_term,
                'instruction_loss': instruction_loss}

    def update_morse_weight(self, current_iteration, n_iterations, params=None):
        if not hasattr(self, 'decay_params_list'):
            assert len(params) >= 2, params
            assert len(params[1:-1]) % 2 == 0
            self.decay_params_list = list(zip([params[0], *params[1:-1][1::2], params[-1]], [0, *params[1:-1][::2], 1]))

        curr = current_iteration / n_iterations
        we, e = min([tup for tup in self.decay_params_list if tup[1] >= curr], key=lambda tup: tup[1])
        w0, s = max([tup for tup in self.decay_params_list if tup[1] <= curr], key=lambda tup: tup[1])

        if self.div_decay == 'linear':
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[5] = w0 + (we - w0) * (current_iteration / n_iterations - s) / (e - s)
            else:
                self.weights[5] = we
        elif self.div_decay == 'quintic':
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[5] = w0 + (we - w0) * (1 - (1 - (current_iteration / n_iterations - s) / (e - s)) ** 5)
            else:
                self.weights[5] = we
        elif self.div_decay == 'step':
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            else:
                self.weights[5] = we
        elif self.div_decay == 'none':
            pass
        else:
            raise Warning("unsupported div decay value")

    def update_guidance_weight(self, current_iteration, n_iterations):
        """Warmup for guidance loss weight (weights[6])."""
        if self.guidance_warmup_fraction <= 0 or len(self.weights) <= 6:
            return
        progress = current_iteration / max(n_iterations, 1)
        if progress < self.guidance_warmup_fraction:
            t = progress / self.guidance_warmup_fraction
            if self.guidance_warmup_type == 'cosine':
                scale = 0.5 * (1.0 - math.cos(math.pi * t))
            else:
                scale = t
            self.weights[6] = self.guidance_target_weight * scale
        else:
            self.weights[6] = self.guidance_target_weight
