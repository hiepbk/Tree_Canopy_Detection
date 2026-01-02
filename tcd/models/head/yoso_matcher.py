"""
YOSO Hungarian Matcher - standard implementation from YOSO/DETR.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch.cuda.amp import autocast


def point_sample(input, point_coords, **kwargs):
    """Sample points from input tensor using bilinear interpolation.
    
    A wrapper around torch.nn.functional.grid_sample to support 3D point_coords tensors.
    Unlike grid_sample it assumes point_coords to lie inside [0, 1] x [0, 1] square.
    
    Args:
        input: [N, C, H, W] tensor
        point_coords: [N, P, 2] tensor with normalized coordinates in [0, 1]
        **kwargs: Additional arguments for grid_sample (e.g., align_corners)
    
    Returns:
        [N, C, P] tensor with sampled values
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    # Convert from [0, 1] to [-1, 1] for grid_sample
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """Compute the DICE loss between predictions and targets.
    
    Args:
        inputs: [num_queries, num_points] - predicted mask logits
        targets: [num_targets, num_points] - target mask values
    
    Returns:
        [num_queries, num_targets] cost matrix
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """Compute sigmoid cross-entropy loss between predictions and targets.
    
    Args:
        inputs: [num_queries, num_points] - predicted mask logits
        targets: [num_targets, num_points] - target mask values
    
    Returns:
        [num_queries, num_targets] cost matrix
    """
    hw = inputs.shape[1]
    
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )
    
    return loss / hw


class HungarianMatcher(nn.Module):
    """Hungarian Matcher for bipartite matching.
    
    This is the standard implementation from YOSO/DETR/Mask2Former.
    
    Args:
        cost_class: Weight for classification cost
        cost_mask: Weight for mask cost
        cost_dice: Weight for dice cost
        num_points: Number of points to sample for mask cost
    """
    
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        num_points: int = 6000,
    ):
        super(HungarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points
        
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"
    
    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Forward function - standard YOSO/DETR implementation.
        
        Args:
            outputs: Dict with 'pred_logits' and 'pred_masks'
            targets: List of dicts with 'labels' and 'masks'
            
        Returns:
            List of (pred_idx, target_idx) pairs
        """
        bs, num_queries = outputs['pred_masks'].shape[:2]
        
        indices = []
        
        # Iterate through batch size
        for b in range(bs):
            tgt_ids = targets[b]['labels']
            
            # Classification cost
            if outputs['pred_logits'] is None:
                cost_class = 0.0
            else:
                out_prob = outputs['pred_logits'][b].softmax(-1)  # [num_queries, num_classes]
                # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                # but approximate it in 1 - proba[target class].
                cost_class = -out_prob[:, tgt_ids]  # [num_queries, num_targets]
            
            # Mask cost
            out_mask = outputs['pred_masks'][b]  # [num_queries, H_pred, W_pred]
            tgt_mask = targets[b]['masks'].to(out_mask)  # [num_targets, H_gt, W_gt]
            
            # Prepare masks for point sampling: add channel dimension
            out_mask = out_mask[:, None]  # [num_queries, 1, H, W]
            tgt_mask = tgt_mask[:, None]  # [num_targets, 1, H, W]
            
            # All masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)  # [1, num_points, 2] in [0, 1]
            
            # Sample from target masks
            tgt_mask_sampled = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)  # [num_targets, num_points]
            
            # Sample from predicted masks
            out_mask_sampled = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)  # [num_queries, num_points]
            
            # Compute costs
            with autocast(enabled=False):
                out_mask_sampled = out_mask_sampled.float()
                tgt_mask_sampled = tgt_mask_sampled.float()
                
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss(out_mask_sampled, tgt_mask_sampled)  # [num_queries, num_targets]
                
                # Compute the dice loss between masks
                cost_dice = batch_dice_loss(out_mask_sampled, tgt_mask_sampled)  # [num_queries, num_targets]
            
            # Final cost matrix
            if isinstance(cost_class, float):
                C = self.cost_mask * cost_mask + self.cost_dice * cost_dice
            else:
                C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
                )
            
            C = C.reshape(num_queries, -1).cpu()
            
            # Hungarian algorithm
            C_np = C.numpy()
            row_ind, col_ind = linear_sum_assignment(C_np)
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64), 
                          torch.as_tensor(col_ind, dtype=torch.int64)))
        
        return indices

