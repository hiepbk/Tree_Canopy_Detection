"""
Mask2Former Head with loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from tcd.models import HEAD


class FocalLoss(nn.Module):
    """
    Focal Loss for classification.
    
    Args:
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        reduction: Reduction method
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Args:
        use_sigmoid: Whether to apply sigmoid before computing loss
        activate: Whether to apply sigmoid/softmax to predictions
        reduction: Reduction method
    """
    
    def __init__(self, use_sigmoid: bool = True, activate: bool = False, reduction: str = 'mean'):
        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.activate = activate
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        if self.activate:
            if self.use_sigmoid:
                pred = torch.sigmoid(pred)
            else:
                pred = F.softmax(pred, dim=1)
        
        # Flatten
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        
        # Compute dice
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        loss = 1 - dice
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for bipartite matching.
    
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
        num_points: int = 12544,
    ):
        super(HungarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points
    
    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Forward function.
        
        Args:
            outputs: Dict with 'pred_logits' and 'pred_masks'
            targets: List of dicts with 'labels' and 'masks'
            
        Returns:
            List of (pred_idx, target_idx) pairs
        """
        bs, num_queries = outputs['pred_logits'].shape[:2]
        
        # Compute cost for each batch
        indices = []
        for b in range(bs):
            out_prob = outputs['pred_logits'][b].softmax(-1)  # [num_queries, num_classes]
            out_mask = outputs['pred_masks'][b]  # [num_queries, H, W]
            
            tgt_ids = targets[b]['labels']  # [num_targets]
            tgt_mask = targets[b]['masks']  # [num_targets, H, W]
            
            # Classification cost
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            
            # Mask cost
            # out_mask: [num_queries, H, W]
            # tgt_mask: [num_targets, H, W]
            
            # Flatten masks
            out_mask_flat = out_mask.flatten(1)  # [num_queries, H*W]
            tgt_mask_flat = tgt_mask.flatten(1)  # [num_targets, H*W]
            
            # Sample points from target masks
            with torch.no_grad():
                points = torch.multinomial(tgt_mask_flat + 1e-8, self.num_points, replacement=True)  # [num_targets, num_points]
            
            # Sample from out_mask for each query-target pair
            # out_mask_flat: [num_queries, H*W]
            # points: [num_targets, num_points]
            # We need: [num_queries, num_targets, num_points]
            num_queries = out_mask_flat.shape[0]
            num_targets = points.shape[0]
            num_points = points.shape[1]
            
            # Expand out_mask_flat: [num_queries, 1, H*W] -> [num_queries, num_targets, H*W]
            out_mask_expanded = out_mask_flat.unsqueeze(1).expand(-1, num_targets, -1)  # [num_queries, num_targets, H*W]
            # Expand points: [1, num_targets, num_points] -> [num_queries, num_targets, num_points]
            points_expanded = points.unsqueeze(0).expand(num_queries, -1, -1)  # [num_queries, num_targets, num_points]
            
            # Gather sampled points: [num_queries, num_targets, num_points]
            out_mask_sampled = out_mask_expanded.gather(2, points_expanded)  # [num_queries, num_targets, num_points]
            
            # Sample from tgt_mask: [num_targets, num_points]
            tgt_mask_sampled = tgt_mask_flat.gather(1, points)  # [num_targets, num_points]
            # Expand to match: [1, num_targets, num_points] -> [num_queries, num_targets, num_points]
            tgt_mask_sampled = tgt_mask_sampled.unsqueeze(0).expand(num_queries, -1, -1)  # [num_queries, num_targets, num_points]
            
            out_mask_sampled = out_mask_sampled.sigmoid()
            tgt_mask_sampled = tgt_mask_sampled.float()
            
            # Dice cost: [num_queries, num_targets]
            cost_dice = self._dice_cost(out_mask_sampled, tgt_mask_sampled)  # [num_queries, num_targets]
            
            # Mask cost (binary cross entropy): [num_queries, num_targets]
            cost_mask = F.binary_cross_entropy_with_logits(
                out_mask_sampled,
                tgt_mask_sampled,
                reduction='none',
            ).mean(-1)  # [num_queries, num_targets]
            
            # Final cost matrix
            C = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            C = C.cpu()
            
            # Hungarian algorithm
            indices.append(self._linear_sum_assignment(C))
        
        return indices
    
    def _dice_cost(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute dice cost.
        
        Args:
            pred: [num_queries, num_targets, num_points] or [..., num_points]
            target: [num_queries, num_targets, num_points] or [..., num_points]
            
        Returns:
            Dice cost: [num_queries, num_targets] or [...]
        """
        # Sum over the last dimension (num_points)
        intersection = (pred * target).sum(-1)  # [num_queries, num_targets]
        union = pred.sum(-1) + target.sum(-1)  # [num_queries, num_targets]
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        return 1 - dice
    
    def _linear_sum_assignment(self, cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Linear sum assignment (Hungarian algorithm).
        
        Simplified version - in practice, use scipy.optimize.linear_sum_assignment
        """
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost.numpy())
            return (torch.from_numpy(row_ind), torch.from_numpy(col_ind))
        except ImportError:
            # Fallback: greedy matching
            row_ind = []
            col_ind = []
            cost_np = cost.numpy()
            used_cols = set()
            for i in range(cost_np.shape[0]):
                valid_cols = [j for j in range(cost_np.shape[1]) if j not in used_cols]
                if valid_cols:
                    j = valid_cols[np.argmin(cost_np[i, valid_cols])]
                    row_ind.append(i)
                    col_ind.append(j)
                    used_cols.add(j)
                else:
                    row_ind.append(i)
                    col_ind.append(0)
            return (torch.tensor(row_ind), torch.tensor(col_ind))


@HEAD.register_module()
class Mask2FormerHead(nn.Module):
    """
    Mask2Former Head for instance segmentation.
    
    Args:
        in_channels: List of input channel numbers
        feat_channels: Feature channel number
        out_channels: Output channel number
        num_things_classes: Number of thing classes
        num_stuff_classes: Number of stuff classes
        num_queries: Number of query embeddings
        num_transformer_feat_level: Number of transformer feature levels
        align_corners: Whether to align corners in interpolation
        pixel_decoder: Pixel decoder config
        transformer_decoder: Transformer decoder config
        loss_cls: Classification loss config
        loss_mask: Mask loss config
        loss_dice: Dice loss config
    """
    
    def __init__(
        self,
        in_channels: List[int],
        feat_channels: int = 256,
        out_channels: int = 256,
        num_things_classes: int = 2,
        num_stuff_classes: int = 0,
        num_queries: int = 100,
        num_transformer_feat_level: int = 3,
        align_corners: bool = False,
        pixel_decoder: Optional[Dict] = None,
        transformer_decoder: Optional[Dict] = None,
        loss_cls: Optional[Dict] = None,
        loss_mask: Optional[Dict] = None,
        loss_dice: Optional[Dict] = None,
    ):
        super(Mask2FormerHead, self).__init__()
        
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = num_things_classes + num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.align_corners = align_corners
        
        # Pixel decoder (simplified - in practice use MSDeformAttn)
        self.pixel_decoder = self._build_pixel_decoder(
            in_channels,
            feat_channels,
            out_channels,
            pixel_decoder or {},
        )
        
        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, out_channels)
        
        # Transformer decoder (simplified)
        self.transformer_decoder = self._build_transformer_decoder(
            out_channels,
            transformer_decoder or {},
        )
        
        # Classification head
        self.cls_embed = nn.Linear(out_channels, self.num_classes + 1)  # +1 for no-object
        
        # Mask head
        self.mask_embed = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, feat_channels),
        )
        
        # Loss functions
        self.loss_cls = self._build_loss(loss_cls or {})
        self.loss_mask = self._build_loss(loss_mask or {})
        self.loss_dice = self._build_loss(loss_dice or {})
    
    def _build_pixel_decoder(self, in_channels, feat_channels, out_channels, cfg):
        """Build pixel decoder."""
        # Simplified pixel decoder - just use 1x1 convs
        layers = []
        for i, in_ch in enumerate(in_channels[:self.num_transformer_feat_level]):
            layers.append(nn.Conv2d(in_ch, out_channels, 1))
        return nn.ModuleList(layers)
    
    def _build_transformer_decoder(self, embed_dims, cfg):
        """Build transformer decoder."""
        num_layers = cfg.get('num_layers', 6)
        layers = []
        for _ in range(num_layers):
            layers.append(nn.TransformerDecoderLayer(
                d_model=embed_dims,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation='relu',
            ))
        return nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dims,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation='relu',
            ),
            num_layers=num_layers,
        )
    
    def _build_loss(self, cfg):
        """Build loss function."""
        loss_type = cfg.get('type', 'FocalLoss')
        if loss_type == 'FocalLoss':
            return FocalLoss(
                alpha=cfg.get('alpha', 0.25),
                gamma=cfg.get('gamma', 2.0),
            )
        elif loss_type == 'DiceLoss':
            return DiceLoss(
                use_sigmoid=cfg.get('use_sigmoid', True),
                activate=cfg.get('activate', False),
            )
        else:
            raise ValueError(f'Unknown loss type: {loss_type}')
    
    def forward(
        self,
        feats: List[torch.Tensor],
        img_metas: List[Dict],
        gt_bboxes: Optional[List[torch.Tensor]] = None,
        gt_labels: Optional[List[torch.Tensor]] = None,
        gt_masks: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward function.
        
        Args:
            feats: List of feature maps from neck
            img_metas: List of image meta information
            gt_bboxes: Ground truth bounding boxes
            gt_labels: Ground truth labels
            gt_masks: Ground truth masks
            
        Returns:
            Dict with predictions and losses (if training)
        """
        # Decode features
        decoder_inputs = []
        for i, feat in enumerate(feats[:self.num_transformer_feat_level]):
            decoded = self.pixel_decoder[i](feat)
            decoder_inputs.append(decoded)
        
        # Flatten features for transformer
        # Simplified - in practice need proper positional encoding
        B, C, H, W = decoder_inputs[0].shape
        query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Transformer decoder
        # Simplified - in practice need proper cross-attention
        memory = decoder_inputs[0].flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        tgt = query_embed.permute(1, 0, 2)  # [num_queries, B, C]
        
        decoder_output = self.transformer_decoder(tgt, memory)
        decoder_output = decoder_output.permute(1, 0, 2)  # [B, num_queries, C]
        
        # Classification
        pred_logits = self.cls_embed(decoder_output)  # [B, num_queries, num_classes+1]
        
        # Mask prediction
        mask_features = self.mask_embed(decoder_output)  # [B, num_queries, feat_channels]
        pred_masks = torch.einsum('bqc,bchw->bqhw', mask_features, decoder_inputs[0])
        
        if self.training:
            # Compute losses
            losses = self.loss(
                pred_logits,
                pred_masks,
                gt_labels,
                gt_masks,
                img_metas,
            )
            return losses
        else:
            # Inference
            return {
                'pred_logits': pred_logits,
                'pred_masks': pred_masks,
            }
    
    def loss(
        self,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        gt_labels: List[torch.Tensor],
        gt_masks: List[torch.Tensor],
        img_metas: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """Compute losses.
        
        Args:
            pred_logits: Predicted classification logits [B, num_queries, num_classes+1]
            pred_masks: Predicted masks [B, num_queries, H, W]
            gt_labels: Ground truth labels [List of [num_targets]]
            gt_masks: Ground truth masks [List of [num_targets, H, W]]
            img_metas: Image meta information
            
        Returns:
            Dict of losses
        """
        # Prepare targets
        # Expected shapes:
        # gt_labels: List[Tensor] - each [num_instances]
        # gt_masks: List[Tensor] - each [num_instances, H_img, W_img]
        # pred_masks: [B, num_queries, H_pred, W_pred]
        
        targets = []
        pred_h, pred_w = pred_masks.shape[-2:]
        
        for labels, masks in zip(gt_labels, gt_masks):
            # masks shape: [num_instances, H_img, W_img]
            # Resize to match pred_masks: [num_instances, H_pred, W_pred]
            if masks.shape[-2:] != (pred_h, pred_w):
                # Resize: [num_instances, H_img, W_img] -> [num_instances, H_pred, W_pred]
                masks = F.interpolate(
                    masks.unsqueeze(1).float(),  # [num_instances, 1, H_img, W_img]
                    size=(pred_h, pred_w),
                    mode='bilinear',
                    align_corners=self.align_corners,
                ).squeeze(1)  # [num_instances, H_pred, W_pred]
            
            targets.append({
                'labels': labels,  # [num_instances]
                'masks': masks,    # [num_instances, H_pred, W_pred]
            })
        
        # Hungarian matching
        matcher = HungarianMatcher()
        indices = matcher(
            {'pred_logits': pred_logits, 'pred_masks': pred_masks},
            targets,
        )
        
        # Compute losses
        loss_cls_all = []
        loss_mask_all = []
        loss_dice_all = []
        
        for b, (pred_idx, target_idx) in enumerate(indices):
            # Classification loss
            pred_logits_b = pred_logits[b]  # [num_queries, num_classes+1]
            target_labels = torch.full(
                (self.num_queries,),
                self.num_classes,  # no-object class
                dtype=torch.long,
                device=pred_logits.device,
            )
            if len(target_idx) > 0:
                target_labels[pred_idx] = targets[b]['labels'][target_idx]
            
            loss_cls = self.loss_cls(pred_logits_b, target_labels)
            loss_cls_all.append(loss_cls)
            
            # Mask losses
            if len(target_idx) > 0:
                pred_masks_b = pred_masks[b][pred_idx]  # [num_matched, H, W]
                target_masks_b = targets[b]['masks'][target_idx]  # [num_matched, H, W]
                
                # Shapes: both [num_matched, H, W]
                # Loss functions expect [N, C, H, W] or [N, H, W] with proper dimensions
                # Add channel dimension for loss computation
                pred_masks_b = pred_masks_b.unsqueeze(1)  # [num_matched, 1, H, W]
                target_masks_b = target_masks_b.unsqueeze(1)  # [num_matched, 1, H, W]
                
                loss_mask = self.loss_mask(pred_masks_b, target_masks_b)
                loss_dice = self.loss_dice(pred_masks_b, target_masks_b)
                
                loss_mask_all.append(loss_mask)
                loss_dice_all.append(loss_dice)
            else:
                loss_mask_all.append(torch.tensor(0.0, device=pred_masks.device))
                loss_dice_all.append(torch.tensor(0.0, device=pred_masks.device))
        
        return {
            'loss_cls': torch.stack(loss_cls_all).mean(),
            'loss_mask': torch.stack(loss_mask_all).mean(),
            'loss_dice': torch.stack(loss_dice_all).mean(),
        }

