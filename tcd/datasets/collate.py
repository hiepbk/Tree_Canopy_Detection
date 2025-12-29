"""
Collate functions for batching data.
"""

import torch
import numpy as np
import cv2
from typing import List, Dict, Any


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching.
    
    Handles:
    - img: Stack into [B, C, H, W]
    - gt_bboxes, gt_labels, gt_masks: Keep as lists (one per image)
    - img_metas: Keep as list
    
    Args:
        batch: List of data dictionaries from dataset
        
    Returns:
        Batched data dictionary
    """
    # Separate different data types
    imgs = []
    gt_bboxes_list = []
    gt_labels_list = []
    gt_masks_list = []
    img_metas_list = []
    
    for item in batch:
        # Image: should be [C, H, W] tensor
        img = item['img']
        if isinstance(img, np.ndarray):
            if img.ndim == 3:
                img = img.transpose(2, 0, 1)  # HWC -> CHW
            img = torch.from_numpy(img).float()
        imgs.append(img)
        
        # Get image shape for mask conversion (after converting to tensor)
        if isinstance(img, torch.Tensor):
            _, H, W = img.shape
        else:
            # Shouldn't happen, but handle it
            H, W = img.shape[-2:] if img.ndim >= 2 else (512, 512)
        
        # Bboxes: tensor [num_instances, 4] -> keep as list
        if 'gt_bboxes' in item:
            bboxes = item['gt_bboxes']
            if isinstance(bboxes, np.ndarray):
                bboxes = torch.from_numpy(bboxes).float()
            elif not isinstance(bboxes, torch.Tensor):
                bboxes = torch.tensor(bboxes, dtype=torch.float32)
            gt_bboxes_list.append(bboxes)
        else:
            gt_bboxes_list.append(torch.zeros((0, 4), dtype=torch.float32))
        
        # Labels: tensor [num_instances] -> keep as list
        if 'gt_labels' in item:
            labels = item['gt_labels']
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels).long()
            elif not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            gt_labels_list.append(labels)
        else:
            gt_labels_list.append(torch.zeros((0,), dtype=torch.long))
        
        # Masks: convert polygons to tensors [num_instances, H, W]
        if 'gt_masks' in item and len(item['gt_masks']) > 0:
            masks = item['gt_masks']
            
            mask_tensors = []
            for mask in masks:
                if isinstance(mask, list):
                    # Polygon format - convert to mask (H, W)
                    mask_array = np.zeros((H, W), dtype=np.uint8)
                    if len(mask) > 0:
                        if isinstance(mask[0], list):
                            for poly in mask:
                                coords = np.array(poly).reshape(-1, 2).astype(np.int32)
                                cv2.fillPoly(mask_array, [coords], 1)
                        else:
                            coords = np.array(mask).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(mask_array, [coords], 1)
                    mask_tensors.append(torch.from_numpy(mask_array).float())
                elif isinstance(mask, torch.Tensor):
                    # Ensure shape is (H, W)
                    if mask.dim() == 2:
                        mask_tensors.append(mask.float())
                    elif mask.dim() == 1 and mask.numel() == H * W:
                        mask_tensors.append(mask.view(H, W).float())
                    else:
                        mask_tensors.append(torch.zeros((H, W), dtype=torch.float32))
                else:
                    # Numpy array
                    mask_tensor = torch.from_numpy(mask).float()
                    if mask_tensor.dim() == 2:
                        mask_tensors.append(mask_tensor)
                    elif mask_tensor.dim() == 1 and mask_tensor.numel() == H * W:
                        mask_tensors.append(mask_tensor.view(H, W))
                    else:
                        mask_tensors.append(torch.zeros((H, W), dtype=torch.float32))
            
            # Stack masks for this image: [num_instances, H, W]
            if len(mask_tensors) > 0:
                gt_masks_list.append(torch.stack(mask_tensors, dim=0))
            else:
                gt_masks_list.append(torch.zeros((0, H, W), dtype=torch.float32))
        else:
            # Empty masks - use image shape already determined above
            gt_masks_list.append(torch.zeros((0, H, W), dtype=torch.float32))
        
        # img_metas: keep as list
        if 'img_metas' in item:
            img_metas_list.append(item['img_metas'])
        else:
            img_metas_list.append({})
    
    # Stack images: [B, C, H, W]
    imgs = torch.stack(imgs, dim=0)
    
    return {
        'img': imgs,
        'gt_bboxes': gt_bboxes_list,
        'gt_labels': gt_labels_list,
        'gt_masks': gt_masks_list,
        'img_metas': img_metas_list,
    }

