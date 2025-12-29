"""
Mask2Former model for instance segmentation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from tcd.models import MODEL
from .backbone import load_satlas_pretrained_weights


@MODEL.register_module()
class Mask2Former(nn.Module):
    """
    Mask2Former model for instance segmentation.
    
    Args:
        backbone: Backbone config
        neck: Neck config
        head: Head config
        train_cfg: Training config
        test_cfg: Testing config
    """
    
    def __init__(
        self,
        backbone: Dict,
        neck: Dict,
        head: Dict,
        train_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
    ):
        super(Mask2Former, self).__init__()
        
        from tcd.models import BACKBONE, NECK, HEAD
        from tcd.utils import build_from_cfg
        from .backbone import load_satlas_pretrained_weights
        
        # Build backbone using registry (config should already be dict)
        backbone_cfg = backbone.copy()
        pretrained_path = backbone_cfg.pop('pretrained', None)  # Remove pretrained from config
        frozen_stages = backbone_cfg.pop('frozen_stages', -1)  # Remove frozen_stages from config
        self.backbone = build_from_cfg(backbone_cfg, BACKBONE)
        
        # Freeze stages if specified
        if frozen_stages >= 0:
            self._freeze_backbone_stages(self.backbone, frozen_stages)
        
        # Load pretrained weights if specified
        if pretrained_path:
            self.backbone = load_satlas_pretrained_weights(
                self.backbone,
                pretrained_path,
            )
        
        # Build neck using registry
        neck_cfg = neck.copy()
        self.neck = build_from_cfg(neck_cfg, NECK)
        
        # Build head using registry
        head_cfg = head.copy()
        self.head = build_from_cfg(head_cfg, HEAD)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    
    def _freeze_backbone_stages(self, backbone, frozen_stages):
        """Freeze backbone stages."""
        if hasattr(backbone, 'backbone') and hasattr(backbone.backbone, 'features'):
            # Swin Transformer structure: features is a Sequential with stages
            # Freeze first (frozen_stages + 1) stages
            for i in range(min(frozen_stages + 1, len(backbone.backbone.features))):
                for param in backbone.backbone.features[i].parameters():
                    param.requires_grad = False
    
    def forward(
        self,
        img: torch.Tensor,
        img_metas: List[Dict],
        return_loss: bool = True,
        gt_bboxes: Optional[List[torch.Tensor]] = None,
        gt_labels: Optional[List[torch.Tensor]] = None,
        gt_masks: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward function.
        
        Args:
            img: Input images [B, C, H, W]
            img_metas: List of image meta information
            return_loss: Whether to return loss (training mode)
            gt_bboxes: Ground truth bounding boxes
            gt_labels: Ground truth labels
            gt_masks: Ground truth masks
            
        Returns:
            Dict with predictions or losses
        """
        # Backbone
        x = self.backbone(img)
        
        # Neck
        x = self.neck(x)
        
        # Head
        if self.training and return_loss:
            return self.head(
                x,
                img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_masks=gt_masks,
            )
        else:
            return self.head(
                x,
                img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_masks=gt_masks,
            )
    
    def train_step(self, data: Dict, optimizer: torch.optim.Optimizer) -> Dict[str, torch.Tensor]:
        """Training step."""
        losses = self(**data)
        loss = sum(losses.values())
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return losses
    
    def val_step(self, data: Dict) -> Dict[str, torch.Tensor]:
        """Validation step."""
        return self(**data, return_loss=False)

