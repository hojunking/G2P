"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES
from .lovasz import LovaszLoss

@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss



def _binary_cross_entropy(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target.float())

# Binary Dice loss used by the boundary branch.
def _binary_dice_loss(pred, target, smooth=1e-5, exponent=2):
    pred_prob = torch.sigmoid(pred) # logits -> probabilities
    target = target.float()
    
    intersection = (pred_prob * target).sum()
    union = (pred_prob.pow(exponent).sum() + target.pow(exponent).sum())
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return (1 - dice)


def _binary_dice_loss_prob(pred, target, smooth=1.0):
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def _multiclass_dice_loss(pred, target, ignore_index=-1, smooth=1.0):
    valid_mask = target != ignore_index
    if valid_mask.sum() == 0:
        return pred.new_tensor(0.0)
    pred = pred[valid_mask]
    target = target[valid_mask]
    prob = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target.long(), num_classes=pred.shape[1]).float()
    intersection = (prob * target_one_hot).sum(dim=0)
    union = prob.sum(dim=0) + target_one_hot.sum(dim=0)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()



@LOSSES.register_module()
class BoundarySemanticLoss(nn.Module):
    def __init__(self, semantic_loss_weight=1.0, boundary_loss_weight=1.0,
                 ignore_index=-1, num_semantic_classes=None, semantic_boundary_weight_factor=9.0):
        super().__init__()
        self.semantic_loss_weight = semantic_loss_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.ignore_index = ignore_index
        self.num_semantic_classes = num_semantic_classes if num_semantic_classes is not None else 20 
        self.semantic_boundary_weight_factor = semantic_boundary_weight_factor

        self.ce_loss = CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.dice_loss_semantic = DiceLoss(ignore_index=ignore_index, exponent=2)
        
        # Binary CE and binary Dice are module-level helpers; nothing to init here.
        # Swap in BinaryFocalLoss here if a focal boundary loss is wanted.

    def forward(self, initial_sem_logits, initial_bou_logits, final_sem_logits, final_bou_logits,
                                gt_semantic_label, gt_boundary_label):
        
        # Reshape gt_boundary_label to (N, 1).
        if gt_boundary_label.dim() == 1:
            gt_boundary_label_unsqueeze = gt_boundary_label.unsqueeze(-1)
        else:
            gt_boundary_label_unsqueeze = gt_boundary_label
        
        # Shared validity mask over the ignore index.
        valid_semantic_mask = (gt_semantic_label != self.ignore_index) # (N,) bool Tensor
        
        sem_weight = torch.ones_like(gt_semantic_label, dtype=torch.float32, device=gt_semantic_label.device)
        sem_weight += gt_boundary_label_unsqueeze.squeeze(-1).float() * self.semantic_boundary_weight_factor # (N,)
        
        # --- 1. Initial semantic loss (BFANet sem_score) ---
        # Per-point CE (reduction='none').
        initial_sem_loss_ce_per_point = self.ce_loss(initial_sem_logits, gt_semantic_label)
        # Apply the boundary-aware weight, averaging over valid points only.
        initial_sem_loss_ce = (initial_sem_loss_ce_per_point * sem_weight)[valid_semantic_mask].mean()
        
        # Dice Loss
        initial_sem_loss_dice = self.dice_loss_semantic(initial_sem_logits, gt_semantic_label)
        
        total_loss_initial_semantic = initial_sem_loss_ce + initial_sem_loss_dice 

        # --- 2. Initial boundary loss (BFANet margin_score) ---
        # BCE loss (the original uses weight=weight_mask).
        # Restrict the loss to valid points via valid_boundary_mask.
        
        valid_boundary_mask_initial = valid_semantic_mask & (
            gt_boundary_label_unsqueeze.squeeze(-1) != self.ignore_index
        )
        valid_initial_bou_logits = initial_bou_logits[valid_boundary_mask_initial]
        valid_gt_boundary_label_initial = gt_boundary_label_unsqueeze[valid_boundary_mask_initial]

        if valid_gt_boundary_label_initial.numel() == 0:
            loss_initial_bou_bce = torch.tensor(0.0, device=initial_bou_logits.device)
            loss_initial_bou_dice = torch.tensor(0.0, device=initial_bou_logits.device)
        else:
            loss_initial_bou_bce = _binary_cross_entropy(valid_initial_bou_logits, valid_gt_boundary_label_initial)
            loss_initial_bou_dice = _binary_dice_loss(valid_initial_bou_logits, valid_gt_boundary_label_initial)
        
        total_loss_initial_boundary = loss_initial_bou_bce + loss_initial_bou_dice

        # --- 3. Final semantic loss (BFANet sem_score_v2) ---
        final_sem_loss_ce_per_point = self.ce_loss(final_sem_logits, gt_semantic_label)
        final_sem_loss_ce = (final_sem_loss_ce_per_point * sem_weight)[valid_semantic_mask].mean()

        final_sem_loss_dice = self.dice_loss_semantic(final_sem_logits, gt_semantic_label)
        total_loss_final_semantic = final_sem_loss_ce + final_sem_loss_dice

        # --- 4. Final boundary loss (BFANet margin_score_v2) ---
        valid_boundary_mask_final = valid_semantic_mask & (
            gt_boundary_label_unsqueeze.squeeze(-1) != self.ignore_index
        )
        valid_final_bou_logits = final_bou_logits[valid_boundary_mask_final]
        valid_gt_boundary_label_final = gt_boundary_label_unsqueeze[valid_boundary_mask_final]

        if valid_gt_boundary_label_final.numel() == 0:
            loss_final_bou_bce = torch.tensor(0.0, device=final_bou_logits.device)
            loss_final_bou_dice = torch.tensor(0.0, device=final_bou_logits.device)
        else:
            loss_final_bou_bce = _binary_cross_entropy(valid_final_bou_logits, valid_gt_boundary_label_final)
            loss_final_bou_dice = _binary_dice_loss(valid_final_bou_logits, valid_gt_boundary_label_final)
        
        total_loss_final_boundary = loss_final_bou_bce + loss_final_bou_dice

        total_loss = (total_loss_initial_semantic * self.semantic_loss_weight +
                      total_loss_initial_boundary * self.boundary_loss_weight +
                      total_loss_final_semantic * self.semantic_loss_weight + # same weights for the final head
                      total_loss_final_boundary * self.boundary_loss_weight) # same weights for the final head
        
        # Report every component alongside the total.
        losses = {
            'loss_initial_semantic': total_loss_initial_semantic * self.semantic_loss_weight,
            'loss_initial_boundary': total_loss_initial_boundary * self.boundary_loss_weight,
            'loss_final_semantic': total_loss_final_semantic * self.semantic_loss_weight,
            'loss_final_boundary': total_loss_final_boundary * self.boundary_loss_weight,
            'loss': total_loss # total
        }
        
        return losses
    

@LOSSES.register_module()
class FeatureDistillationLoss(nn.Module):
    def __init__(self,
                 loss_weight=1.0,
                 loss_type="L2"):
        super().__init__()
        self.loss_weight = loss_weight
        assert loss_type in ["L2", "CosineSimilarity"], "loss_type must be L2 or CosineSimilarity"
        self.loss_type = loss_type

    def forward(self, student_feat, teacher_feat):
        if self.loss_type == "L2":
            loss = F.mse_loss(student_feat, teacher_feat)
        elif self.loss_type == "CosineSimilarity":
            # Cosine similarity lies in [-1, 1]; use (1 - similarity) as the loss.
            loss = 1 - F.cosine_similarity(student_feat, teacher_feat, dim=-1).mean()
        
        # Return the loss dict.
        return loss * self.loss_weight

@LOSSES.register_module()
class BSLossWithLovasz(nn.Module):
    """
    Boundary-Semantic Loss with Lovasz-Softmax.
    - Semantic Loss: CrossEntropy + Lovasz-Softmax
    - Boundary Loss: Binary CrossEntropy + Binary Dice
    """
    def __init__(self, 
                 semantic_loss_weight=1.0, 
                 boundary_loss_weight=1.0,
                 lovasz_loss_weight=1.0, # separate weight for the Lovasz term
                 ignore_index=-1, 
                 num_semantic_classes=20, 
                 semantic_boundary_weight_factor=9.0):
        super().__init__()
        self.semantic_loss_weight = semantic_loss_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.lovasz_loss_weight = lovasz_loss_weight 
        self.ignore_index = ignore_index
        self.semantic_boundary_weight_factor = semantic_boundary_weight_factor

        # --- Loss components ---
        # 1. Semantic.
        self.ce_loss = CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.lovasz_loss = LovaszLoss(mode="multiclass", ignore_index=ignore_index)
        
        # 2. Boundary losses are module-level helpers.

    def forward(self, 
                initial_sem_logits, initial_bou_logits, 
                final_sem_logits, final_bou_logits,
                gt_semantic_label, gt_boundary_label):
        
        # Guard against out-of-range class indices. Skip the check when the
        # batch is entirely ignore_index -- .max() on an empty tensor raises.
        valid_labels = gt_semantic_label[gt_semantic_label != self.ignore_index]
        if valid_labels.numel() > 0 and valid_labels.max() >= initial_sem_logits.shape[1]:
            raise ValueError(
                "Invalid class index in gt_semantic_label: "
                f"range [{gt_semantic_label.min().item()}, {gt_semantic_label.max().item()}] "
                f"but the model predicts {initial_sem_logits.shape[1]} classes."
            )

        # --- Preprocessing ---
        if gt_boundary_label.dim() == 1:
            gt_boundary_label_unsqueeze = gt_boundary_label.unsqueeze(-1)
        else:
            gt_boundary_label_unsqueeze = gt_boundary_label
        valid_semantic_mask = (gt_semantic_label != self.ignore_index)
        sem_weight = torch.ones_like(gt_semantic_label, dtype=torch.float32)
        sem_weight += gt_boundary_label_unsqueeze.squeeze(-1).float() * self.semantic_boundary_weight_factor

        # --- 1. Initial semantic loss (CE + Lovasz) ---
        # A batch that is entirely ignore_index contributes no semantic loss.
        # LovaszLoss returns an empty (0, C) tensor in that case, which would
        # make the total loss non-scalar and break backward().
        if valid_semantic_mask.any():
            initial_sem_loss_ce_per_point = self.ce_loss(initial_sem_logits, gt_semantic_label)
            initial_sem_loss_ce = (initial_sem_loss_ce_per_point * sem_weight)[valid_semantic_mask].mean()
            initial_sem_loss_lovasz = self.lovasz_loss(initial_sem_logits, gt_semantic_label)
            total_loss_initial_semantic = (
                initial_sem_loss_ce + initial_sem_loss_lovasz * self.lovasz_loss_weight
            )
        else:
            total_loss_initial_semantic = initial_sem_logits.sum() * 0.0

        # --- 2. Initial boundary loss (BCE + Dice) ---
        valid_boundary_mask_initial = valid_semantic_mask & (
            gt_boundary_label_unsqueeze.squeeze(-1) != self.ignore_index
        )
        valid_initial_bou_logits = initial_bou_logits[valid_boundary_mask_initial]
        valid_gt_boundary_label_initial = gt_boundary_label_unsqueeze[valid_boundary_mask_initial]
        if valid_gt_boundary_label_initial.numel() > 0:
            loss_initial_bou_bce = _binary_cross_entropy(valid_initial_bou_logits, valid_gt_boundary_label_initial)
            loss_initial_bou_dice = _binary_dice_loss(valid_initial_bou_logits, valid_gt_boundary_label_initial)
            total_loss_initial_boundary = loss_initial_bou_bce + loss_initial_bou_dice
        else:
            total_loss_initial_boundary = torch.tensor(0.0, device=initial_bou_logits.device)

        # --- 3. Final semantic loss (CE + Lovasz) ---
        if valid_semantic_mask.any():
            final_sem_loss_ce_per_point = self.ce_loss(final_sem_logits, gt_semantic_label)
            final_sem_loss_ce = (final_sem_loss_ce_per_point * sem_weight)[valid_semantic_mask].mean()
            final_sem_loss_lovasz = self.lovasz_loss(final_sem_logits, gt_semantic_label)
            total_loss_final_semantic = (
                final_sem_loss_ce + final_sem_loss_lovasz * self.lovasz_loss_weight
            )
        else:
            total_loss_final_semantic = final_sem_logits.sum() * 0.0

        # --- 4. Final boundary loss (BCE + Dice) ---
        valid_boundary_mask_final = valid_semantic_mask & (
            gt_boundary_label_unsqueeze.squeeze(-1) != self.ignore_index
        )
        valid_final_bou_logits = final_bou_logits[valid_boundary_mask_final]
        valid_gt_boundary_label_final = gt_boundary_label_unsqueeze[valid_boundary_mask_final]
        if valid_gt_boundary_label_final.numel() > 0:
            loss_final_bou_bce = _binary_cross_entropy(valid_final_bou_logits, valid_gt_boundary_label_final)
            loss_final_bou_dice = _binary_dice_loss(valid_final_bou_logits, valid_gt_boundary_label_final)
            total_loss_final_boundary = loss_final_bou_bce + loss_final_bou_dice
        else:
            total_loss_final_boundary = torch.tensor(0.0, device=final_bou_logits.device)

        # --- Total ---
        total_loss = (total_loss_initial_semantic * self.semantic_loss_weight +
                      total_loss_initial_boundary * self.boundary_loss_weight +
                      total_loss_final_semantic * self.semantic_loss_weight +
                      total_loss_final_boundary * self.boundary_loss_weight)
        
        # --- Return dict ---
        losses = {
            'loss_initial_semantic': total_loss_initial_semantic * self.semantic_loss_weight,
            'loss_initial_boundary': total_loss_initial_boundary * self.boundary_loss_weight,
            'loss_final_semantic': total_loss_final_semantic * self.semantic_loss_weight,
            'loss_final_boundary': total_loss_final_boundary * self.boundary_loss_weight,
            'loss': total_loss
        }
        
        return losses
