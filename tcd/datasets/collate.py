"""
Collate functions for batching data.
"""

import torch
import numpy as np
from typing import List, Dict, Any


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching.
    
    Handles:
    - img: Stack into [B, C, H, W]
    - gt_bboxes, gt_labels, gt_masks: Keep as lists (one per image)
    - img_metas: Keep as list
    - ori_gt_masks: Keep as list of masks (at original size)
    
    Args:
        batch: List of data dictionaries from dataset
        
    Returns:
        Batched data dictionary
    """
    imgs = []
    gt_bboxes_list = []
    gt_labels_list = []
    gt_masks_list = []
    img_metas_list = []
    ori_gt_masks_list = []
    ori_imgs_list = []
    
    for item in batch:
        # Image: already tensor [C, H, W] from DefaultFormatBundle
        img = item['img']
        imgs.append(img)
        
        # Get image shape
        _, H, W = img.shape
        
        # Bboxes: already tensor from DefaultFormatBundle
        bboxes = item['gt_bboxes']
        gt_bboxes_list.append(bboxes)
        
        # Labels: tensor [num_instances] -> keep as list
        labels = item['gt_labels']
        gt_labels_list.append(labels)
        
        # Masks: already tensors from DefaultFormatBundle -> stack to [num_instances, H, W]
        masks = item['gt_masks']
        if len(masks) > 0:
            gt_masks_list.append(torch.stack(masks, dim=0))
        else:
            gt_masks_list.append(torch.zeros((0, H, W), dtype=torch.float32))
        
        # img_metas: keep as list
        img_metas_list.append(item['img_metas'])
        
        # Original masks: already masks (numpy arrays) at original size
        # Convert to list of tensors
        ori_masks = item['ori_gt_masks']
        ori_masks_tensors = [torch.from_numpy(m).float() for m in ori_masks]
        ori_gt_masks_list.append(ori_masks_tensors)
        
        # Original images: keep as numpy arrays (for visualization)
        ori_img = item['ori_img']  # numpy array [H, W, C] RGB
        ori_imgs_list.append(ori_img)
    
    # Stack images: [B, C, H, W]
    imgs = torch.stack(imgs, dim=0)
    
    return {
        'img': imgs,
        'gt_bboxes': gt_bboxes_list,
        'gt_labels': gt_labels_list,
        'gt_masks': gt_masks_list,
        'img_metas': img_metas_list,
        'ori_gt_masks': ori_gt_masks_list,
        'ori_img': ori_imgs_list,
    }

