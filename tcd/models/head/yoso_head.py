"""
YOSO Head adapted for instance segmentation.
Based on YOSO architecture but adapted to work with our data format.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from timm.models.layers import trunc_normal_

from tcd.models import HEAD


class FFN(nn.Module):
    """Feed Forward Network."""
    def __init__(self, embed_dims=256, feedforward_channels=1024, num_fcs=2, add_identity=True):
        super(FFN, self).__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(nn.Linear(in_channels, feedforward_channels),
                              nn.ReLU(True),
                              nn.Dropout(0.0)
                              ))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(0.0))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity
        self.dropout_layer = nn.Dropout(0.0)

    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class DySepConvAtten(nn.Module):
    """Dynamic Separable Convolution Attention."""
    def __init__(self, hidden_dim, num_proposals, kernel_size, conv_kernel_size_2d=1):
        super(DySepConvAtten, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_proposals = num_proposals
        self.kernel_size = kernel_size
        self.conv_kernel_size_2d = conv_kernel_size_2d
        # With K=1, input_dim = hidden_dim * 1**2 = hidden_dim (matches original YOSO)
        # Original YOSO uses hidden_dim directly for the linear layer
        self.input_dim = hidden_dim * conv_kernel_size_2d**2

        # Original YOSO uses hidden_dim, not input_dim, but with K=1 they're the same
        self.weight_linear = nn.Linear(self.hidden_dim, self.num_proposals + self.kernel_size)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, query, value):
        if query.shape != value.shape:
            raise AssertionError(f"Shape mismatch: query={query.shape}, value={value.shape}")
        B, N, C = query.shape
        
        dy_conv_weight = self.weight_linear(query)
        dy_depth_conv_weight = dy_conv_weight[:, :, :self.kernel_size].view(B, self.num_proposals, 1, self.kernel_size)
        dy_point_conv_weight = dy_conv_weight[:, :, self.kernel_size:].view(B, self.num_proposals, self.num_proposals, 1)

        res = []
        value = value.unsqueeze(1)
        
        depth_padding = self.calculate_padding(self.kernel_size)
        point_padding = 0  # since kernel size is 1 in point convolution
        for i in range(B):
            out = F.relu(F.conv1d(input=value[i], weight=dy_depth_conv_weight[i], groups=N, padding=depth_padding))
            out = F.conv1d(input=out, weight=dy_point_conv_weight[i], padding=point_padding)
            res.append(out)
        point_out = torch.cat(res, dim=0)
        point_out = self.norm(point_out)
        return point_out
    
    def calculate_padding(self, kernel_size):
        return (kernel_size - 1) // 2


class CrossAttenHead(nn.Module):
    """Cross Attention Head for multi-stage refinement."""
    def __init__(
        self,
        hidden_dim: int,
        num_proposals: int,
        num_classes: int,
        conv_kernel_size_2d: int = 3,
        conv_kernel_size_1d: int = 3,
        num_cls_fcs: int = 2,
        num_mask_fcs: int = 2,
        feedforward_channels: int = 2048,
    ):
        super(CrossAttenHead, self).__init__()
        self.num_classes = num_classes
        self.conv_kernel_size_2d = conv_kernel_size_2d
        self.hidden_dim = hidden_dim
        self.num_proposals = num_proposals
        self.hard_mask_thr = 0.5

        # With K=1 (conv_kernel_size_2d=1), hidden_dim * K**2 = hidden_dim
        # So all norms expect hidden_dim, not hidden_dim * K**2
        self.f_atten = DySepConvAtten(hidden_dim, num_proposals, conv_kernel_size_1d, conv_kernel_size_2d)
        self.f_dropout = nn.Dropout(0.0)
        self.f_atten_norm = nn.LayerNorm(self.hidden_dim * self.conv_kernel_size_2d**2)

        self.k_atten = DySepConvAtten(hidden_dim, num_proposals, conv_kernel_size_1d, conv_kernel_size_2d)
        self.k_dropout = nn.Dropout(0.0)
        self.k_atten_norm = nn.LayerNorm(self.hidden_dim * self.conv_kernel_size_2d**2)
        
        self.s_atten = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * self.conv_kernel_size_2d**2,
            num_heads=8,
            dropout=0.0,
        )
        self.s_dropout = nn.Dropout(0.0)
        self.s_atten_norm = nn.LayerNorm(self.hidden_dim * self.conv_kernel_size_2d**2)

        self.ffn = FFN(self.hidden_dim, feedforward_channels=feedforward_channels, num_fcs=2)
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False))
            self.cls_fcs.append(nn.LayerNorm(self.hidden_dim))
            self.cls_fcs.append(nn.ReLU(True))
        self.fc_cls = nn.Linear(self.hidden_dim, self.num_classes + 1)

        self.mask_fcs = nn.ModuleList()
        for _ in range(num_mask_fcs):
            self.mask_fcs.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False))
            self.mask_fcs.append(nn.LayerNorm(self.hidden_dim))
            self.mask_fcs.append(nn.ReLU(True))
        self.fc_mask = nn.Linear(self.hidden_dim, self.hidden_dim)

        prior_prob = 0.01
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)

        self.apply(self._init_weights)
        nn.init.constant_(self.fc_cls.bias, self.bias_value)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, features, proposal_kernels, mask_preds, train_flag):
        B, C, H, W = features.shape

        soft_sigmoid_masks = mask_preds.sigmoid()
        nonzero_inds = soft_sigmoid_masks > self.hard_mask_thr
        hard_sigmoid_masks = nonzero_inds.float()
        
        if torch.isnan(mask_preds).any() or torch.isinf(mask_preds).any():
            raise ValueError("mask_preds contains NaN/Inf")
        
        if torch.isnan(hard_sigmoid_masks).any() or torch.isinf(hard_sigmoid_masks).any():
            raise ValueError("hard_sigmoid_masks contains NaN/Inf")
        
        if torch.isnan(features).any() or torch.isinf(features).any():
            raise ValueError("features contains NaN/Inf")

        # [B, N, C] - extract features using hard masks
        # Compute einsum (original YOSO uses float32 throughout, no mixed precision)
        # This is the weighted sum: sum_{h,w} (mask[b,n,h,w] * features[b,c,h,w])
        f = torch.einsum('bnhw,bchw->bnc', hard_sigmoid_masks, features)
        
        if torch.isnan(f).any() or torch.isinf(f).any():
            raise ValueError("f contains NaN/Inf after einsum")
        
        # [B, N, C, K, K] -> [B, N, C * K * K]
        # With K=1 (conv_kernel_size_2d=1), this is [B, N, C, 1, 1] -> [B, N, C]
        k = proposal_kernels.reshape(B, self.num_proposals, -1)
        
        expected_k_dim = self.hidden_dim * self.conv_kernel_size_2d**2
        
        # With K=1, both f and k should be [B, N, hidden_dim]
        # No expansion needed!
        if f.shape != k.shape:
            raise AssertionError(f"Shape mismatch: f={f.shape}, k={k.shape}")

        # Feature attention
        f_tmp = self.f_atten(k, f)
        f = f + self.f_dropout(f_tmp)
        f = self.f_atten_norm(f)

        # Kernel attention
        f_tmp = self.k_atten(k, f)
        f = f + self.k_dropout(f_tmp)
        k = self.k_atten_norm(f)

        # Self attention
        k = k.permute(1, 0, 2)  # [N, B, C]
        k_tmp = self.s_atten(query=k, key=k, value=k)[0]
        k = k + self.s_dropout(k_tmp)
        k = self.s_atten_norm(k.permute(1, 0, 2))

        # [B, N, C * K * K] -> [B, N, C, K * K] -> [B, N, K * K, C]
        obj_feat = k.reshape(B, self.num_proposals, self.hidden_dim, -1).permute(0, 1, 3, 2)
        obj_feat = self.ffn_norm(self.ffn(obj_feat))
        cls_feat = obj_feat.sum(-2)
        mask_feat = obj_feat
        
        if train_flag:
            for cls_layer in self.cls_fcs:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.fc_cls(cls_feat).view(B, self.num_proposals, -1)
        else:
            cls_score = None

        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)
        
        # [B, N, K * K, C] -> [B, N, C]
        mask_kernels = self.fc_mask(mask_feat).squeeze(2)
        
        new_mask_preds = torch.einsum("bqc,bchw->bqhw", mask_kernels, features)

        return cls_score, new_mask_preds, obj_feat.permute(0, 1, 3, 2).reshape(B, self.num_proposals, self.hidden_dim, self.conv_kernel_size_2d, self.conv_kernel_size_2d)


@HEAD.register_module()
class YOSOHead(nn.Module):
    """
    YOSO Head for instance segmentation.
    Adapted from YOSO architecture but for instance segmentation task.
    
    Args:
        in_channels: Input channel number (single feature map from neck)
        hidden_dim: Hidden dimension
        num_things_classes: Number of thing classes
        num_proposals: Number of proposal kernels
        num_stages: Number of refinement stages
        conv_kernel_size_2d: 2D convolution kernel size
        conv_kernel_size_1d: 1D convolution kernel size for attention
        temperature: Temperature for logits scaling
        num_cls_fcs: Number of classification FC layers
        num_mask_fcs: Number of mask FC layers
        feedforward_channels: Feedforward channel number
        loss_cls: Classification loss config
        loss_mask: Mask loss config
        loss_dice: Dice loss config
        num_points: Number of points for mask loss computation
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_things_classes: int = 2,
        num_proposals: int = 100,
        num_stages: int = 3,
        conv_kernel_size_2d: int = 3,
        conv_kernel_size_1d: int = 3,
        temperature: float = 0.1,
        num_cls_fcs: int = 2,
        num_mask_fcs: int = 2,
        feedforward_channels: int = 2048,
        loss_cls: Optional[Dict] = None,
        loss_mask: Optional[Dict] = None,
        loss_dice: Optional[Dict] = None,
        num_points: int = 6000,
    ):
        super(YOSOHead, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_things_classes = num_things_classes
        self.num_proposals = num_proposals
        self.num_stages = num_stages
        self.temperature = temperature
        
        # Initial kernel generator
        self.kernels = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_proposals,
            kernel_size=1
        )
        
        # Multi-stage refinement heads
        self.mask_heads = nn.ModuleList()
        for _ in range(self.num_stages):
            self.mask_heads.append(CrossAttenHead(
                hidden_dim=hidden_dim,
                num_proposals=num_proposals,
                num_classes=num_things_classes,
                conv_kernel_size_2d=conv_kernel_size_2d,
                conv_kernel_size_1d=conv_kernel_size_1d,
                num_cls_fcs=num_cls_fcs,
                num_mask_fcs=num_mask_fcs,
                feedforward_channels=feedforward_channels,
            ))
        
        # Loss functions
        from .mask2former_head import FocalLoss, DiceLoss
        from .yoso_matcher import HungarianMatcher
        
        # Remove 'type' key from loss configs before passing to constructors
        if loss_cls is None:
            self.loss_cls = FocalLoss()
        else:
            loss_cls_clean = {k: v for k, v in loss_cls.items() if k != 'type'}
            self.loss_cls = FocalLoss(**loss_cls_clean)
        
        # loss_mask should be BCE, loss_dice should be Dice
            # Use BCE for mask loss
            self.loss_mask = nn.BCEWithLogitsLoss(reduction='mean')
            # If config specifies DiceLoss, use it, otherwise BCE
            if loss_mask.get('type') == 'DiceLoss':
                loss_mask_clean = {k: v for k, v in loss_mask.items() if k != 'type'}
                self.loss_mask = DiceLoss(**loss_mask_clean)
            else:
                self.loss_mask = nn.BCEWithLogitsLoss(reduction='mean')
        
        if loss_dice is None:
            self.loss_dice = DiceLoss(use_sigmoid=True, activate=False)
        else:
            loss_dice_clean = {k: v for k, v in loss_dice.items() if k != 'type'}
            self.loss_dice = DiceLoss(**loss_dice_clean)
        self.num_points = num_points
        
        # Hungarian matcher
        self.matcher = HungarianMatcher(
            cost_class=2.0,
            cost_mask=5.0,
            cost_dice=5.0,
            num_points=num_points,
        )
    
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
            feats: List of feature maps from neck (should be single tensor for YOSO)
            img_metas: List of image meta information
            gt_bboxes: Ground truth bounding boxes (not used, kept for compatibility)
            gt_labels: Ground truth labels
            gt_masks: Ground truth masks
            
        Returns:
            Dict with predictions and losses (if training)
        """
        # YOSO expects single feature map
        if len(feats) != 1:
            raise ValueError(f"YOSOHead expects single feature map, got {len(feats)}")
        features = feats[0]  # [B, C, H, W]
        
        if torch.isnan(features).any() or torch.isinf(features).any():
            raise ValueError("Features from neck contain NaN/Inf - check neck implementation!")
        
        # Multi-stage refinement
        all_stage_loss = {}
        for stage in range(self.num_stages + 1):
            if stage == 0:
                # Initial prediction from kernels
                mask_preds = self.kernels(features)  # [B, N, H, W]
                
                if torch.isnan(mask_preds).any() or torch.isinf(mask_preds).any():
                    raise ValueError("mask_preds from stage 0 contains NaN/Inf")
                
                # Clamp mask_preds to prevent Inf in sigmoid
                mask_preds = torch.clamp(mask_preds, min=-10.0, max=10.0)
                
                cls_scores = None
                proposal_kernels = self.kernels.weight.clone()  # [N, C, 1, 1]
                object_kernels = proposal_kernels[None].expand(features.shape[0], *proposal_kernels.size())  # [B, N, C, 1, 1]
            elif stage == self.num_stages:
                # Final stage - always compute classification
                mask_head = self.mask_heads[stage - 1]
                cls_scores, mask_preds, proposal_kernels = mask_head(features, object_kernels, mask_preds, True)
            else:
                # Intermediate stages
                mask_head = self.mask_heads[stage - 1]
                cls_scores, mask_preds, proposal_kernels = mask_head(features, object_kernels, mask_preds, self.training)
                object_kernels = proposal_kernels
            
            # Scale logits by temperature
            if cls_scores is not None:
                cls_scores = cls_scores / self.temperature
            
            # Compute loss if training (only when cls_scores is available)
            if self.training and gt_labels is not None and gt_masks is not None and cls_scores is not None:
                single_stage_loss = self.loss(cls_scores, mask_preds, gt_labels, gt_masks, img_metas)
                for key, value in single_stage_loss.items():
                    all_stage_loss[f's{stage}_{key}'] = value
        
        if self.training:
            return all_stage_loss
        else:
            return {
                'pred_logits': cls_scores,
                'pred_masks': mask_preds,
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
            pred_logits: Predicted classification logits [B, num_proposals, num_classes+1]
            pred_masks: Predicted masks [B, num_proposals, H, W]
            gt_labels: Ground truth labels [List of [num_targets]]
            gt_masks: Ground truth masks [List of [num_targets, H, W]]
            img_metas: Image meta information
            
        Returns:
            Dict of losses
        """
        # Prepare targets in format expected by matcher
        targets = []
        pred_h, pred_w = pred_masks.shape[-2:]
        
        for labels, masks in zip(gt_labels, gt_masks):
            # Handle empty masks (no instances)
            if masks.shape[0] == 0:
                empty_masks = torch.zeros((0, pred_h, pred_w), dtype=torch.float32, device=masks.device)
                empty_labels = torch.zeros((0,), dtype=torch.long, device=labels.device)
                targets.append({
                    'labels': empty_labels,
                    'masks': empty_masks,
                })
                continue
            
            # Resize GT masks to match prediction size
            if masks.shape[-2:] != (pred_h, pred_w):
                masks_resized = F.interpolate(
                    masks.unsqueeze(0).float(),
                    size=(pred_h, pred_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            else:
                masks_resized = masks.float()
            
            targets.append({
                'labels': labels,
                'masks': masks_resized,
            })
        
        # Match predictions to targets
        outputs = {'pred_logits': pred_logits, 'pred_masks': pred_masks}
        indices = self.matcher(outputs, targets)
        
        # Compute classification loss
        loss_cls = self._loss_cls(pred_logits, targets, indices)
        
        # Compute mask losses
        loss_mask, loss_dice = self._loss_masks(pred_masks, targets, indices)
        
        return {
            'loss_cls': loss_cls,
            'loss_mask': loss_mask,
            'loss_dice': loss_dice,
        }
    
    def _loss_cls(self, pred_logits, targets, indices):
        """Compute classification loss."""
        if pred_logits is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Flatten predictions
        pred_logits_flat = pred_logits.flatten(0, 1)  # [B*N, num_classes+1]
        
        # Create target classes (background for all, then fill matched ones)
        target_classes = torch.full(
            pred_logits.shape[:2],
            self.num_things_classes,  # background class
            dtype=torch.long,
            device=pred_logits.device
        )
        
        # Fill matched predictions with target labels
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) > 0:
                target_classes[b, pred_idx] = targets[b]['labels'][tgt_idx]
        
        # Compute loss
        target_classes_flat = target_classes.flatten(0, 1)
        loss = self.loss_cls(pred_logits_flat, target_classes_flat)
        
        return loss
    
    def _loss_masks(self, pred_masks, targets, indices):
        """Compute mask losses (BCE and Dice)."""
        # Collect matched masks
        src_masks = []
        tgt_masks = []
        
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) > 0:
                src_masks.append(pred_masks[b, pred_idx])
                tgt_masks.append(targets[b]['masks'][tgt_idx])
        
        if len(src_masks) == 0:
            device = pred_masks.device
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        src_masks = torch.cat(src_masks, dim=0)  # [num_matched, H, W]
        tgt_masks = torch.cat(tgt_masks, dim=0)  # [num_matched, H, W]
        
        # Flatten for loss computation
        src_masks_flat = src_masks.flatten(1)  # [num_matched, H*W]
        tgt_masks_flat = tgt_masks.flatten(1)  # [num_matched, H*W]
        
        # Compute losses
        # For BCE loss, need to ensure proper format
        if isinstance(self.loss_mask, nn.BCEWithLogitsLoss):
            loss_mask = self.loss_mask(src_masks_flat, tgt_masks_flat)
        else:
            loss_mask = self.loss_mask(src_masks_flat, tgt_masks_flat)
        
        loss_dice = self.loss_dice(src_masks_flat, tgt_masks_flat)
        
        return loss_mask, loss_dice

