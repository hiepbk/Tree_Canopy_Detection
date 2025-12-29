"""
Weighted mAP Evaluator for Tree Canopy Detection.
"""

import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json
import re


class WeightedMAPEvaluator:
    """
    Weighted Mean Average Precision Evaluator.
    
    Weights are applied based on scene type and resolution.
    """
    
    # Scene type weights
    SCENE_WEIGHTS = {
        "agriculture_plantation": 2.00,
        "urban_area": 1.50,
        "rural_area": 1.00,
        "industrial_area": 1.25,
        "open_field": 1.00,
    }
    
    # Resolution weights (cm)
    RESOLUTION_WEIGHTS = {
        "10": 1.00,
        "20": 1.25,
        "40": 2.00,
        "60": 2.50,
        "80": 3.00,
    }
    
    # Class names
    CLASS_NAMES = ['individual_tree', 'group_of_trees']
    
    def __init__(self, iou_threshold: float = 0.75):
        """
        Args:
            iou_threshold: IoU threshold for positive detection
        """
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset evaluator state."""
        self.predictions = []  # List of predictions per image
        self.ground_truths = []  # List of ground truth per image
        self.image_weights = []  # List of weights per image
    
    def compute_mask_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Compute IoU between two binary masks.
        
        Args:
            pred_mask: Binary prediction mask [H, W]
            gt_mask: Binary ground truth mask [H, W]
            
        Returns:
            IoU score
        """
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            return 0.0
        
        return float(intersection) / float(union)
    
    def compute_ap(self, tp: List[bool], scores: List[float]) -> float:
        """Compute Average Precision from true positives and scores.
        
        Args:
            tp: List of true positive flags
            scores: List of confidence scores
            
        Returns:
            Average Precision
        """
        if len(tp) == 0:
            return 0.0
        
        # Sort by score (descending)
        indices = np.argsort(scores)[::-1]
        tp_sorted = [tp[i] for i in indices]
        
        # Compute cumulative precision
        tp_cumsum = np.cumsum(tp_sorted).astype(float)
        precision = tp_cumsum / np.arange(1, len(tp_sorted) + 1)
        
        # Average precision
        ap = np.sum(precision * tp_sorted) / max(len(tp_sorted), 1)
        
        return float(ap)
    
    def evaluate_image(
        self,
        pred_masks: List[np.ndarray],
        pred_labels: List[int],
        pred_scores: List[float],
        gt_masks: List[np.ndarray],
        gt_labels: List[int],
        scene_type: str = "rural_area",
        resolution: str = "40",
    ) -> Dict:
        """Evaluate predictions for a single image.
        
        Args:
            pred_masks: List of prediction masks [H, W] each
            pred_labels: List of prediction class labels
            pred_scores: List of prediction confidence scores
            gt_masks: List of ground truth masks [H, W] each
            gt_labels: List of ground truth class labels
            scene_type: Scene type for weighting
            resolution: Resolution in cm for weighting
            
        Returns:
            Dict with per-class AP and mAP
        """
        # Compute weight for this image
        scene_weight = self.SCENE_WEIGHTS.get(scene_type, 1.0)
        resolution_weight = self.RESOLUTION_WEIGHTS.get(str(resolution), 1.0)
        image_weight = scene_weight * resolution_weight
        
        num_classes = len(self.CLASS_NAMES)
        class_aps = []
        
        # Evaluate each class
        for class_id in range(1, num_classes + 1):  # 1-indexed classes
            # Get predictions and ground truth for this class
            pred_indices = [i for i, label in enumerate(pred_labels) if label == class_id]
            gt_indices = [i for i, label in enumerate(gt_labels) if label == class_id]
            
            if len(gt_indices) == 0:
                # No ground truth for this class
                if len(pred_indices) == 0:
                    ap = 1.0  # Perfect if no predictions and no GT
                else:
                    ap = 0.0  # All predictions are false positives
                class_aps.append(ap)
                continue
            
            if len(pred_indices) == 0:
                # No predictions for this class
                class_aps.append(0.0)
                continue
            
            # Match predictions to ground truth
            tp = [False] * len(pred_indices)
            used_gt = [False] * len(gt_indices)
            
            # Sort predictions by score
            pred_scores_class = [pred_scores[i] for i in pred_indices]
            sorted_pred_indices = sorted(
                range(len(pred_indices)),
                key=lambda i: pred_scores_class[i],
                reverse=True
            )
            
            # Greedy matching: assign each prediction to best unmatched GT
            for pred_idx in sorted_pred_indices:
                pred_mask = pred_masks[pred_indices[pred_idx]]
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx in range(len(gt_indices)):
                    if used_gt[gt_idx]:
                        continue
                    
                    gt_mask = gt_masks[gt_indices[gt_idx]]
                    iou = self.compute_mask_iou(pred_mask, gt_mask)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # If IoU > threshold, mark as true positive
                if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                    tp[pred_idx] = True
                    used_gt[best_gt_idx] = True
            
            # Compute AP for this class
            tp_list = [tp[i] for i in sorted_pred_indices]
            scores_list = [pred_scores_class[i] for i in sorted_pred_indices]
            ap = self.compute_ap(tp_list, scores_list)
            class_aps.append(ap)
        
        # Compute mAP (mean of class APs)
        map_score = np.mean(class_aps) if len(class_aps) > 0 else 0.0
        
        return {
            'map': map_score,
            'class_aps': class_aps,
            'weight': image_weight,
        }
    
    def add_prediction(
        self,
        pred_masks: List[np.ndarray],
        pred_labels: List[int],
        pred_scores: List[float],
        gt_masks: List[np.ndarray],
        gt_labels: List[int],
        scene_type: str = "rural_area",
        resolution: str = "40",
    ):
        """Add prediction and ground truth for evaluation.
        
        Args:
            pred_masks: List of prediction masks
            pred_labels: List of prediction class labels
            pred_scores: List of prediction confidence scores
            gt_masks: List of ground truth masks
            gt_labels: List of ground truth class labels
            scene_type: Scene type
            resolution: Resolution in cm
        """
        result = self.evaluate_image(
            pred_masks, pred_labels, pred_scores,
            gt_masks, gt_labels, scene_type, resolution
        )
        
        self.predictions.append({
            'masks': pred_masks,
            'labels': pred_labels,
            'scores': pred_scores,
        })
        self.ground_truths.append({
            'masks': gt_masks,
            'labels': gt_labels,
        })
        self.image_weights.append(result['weight'])
    
    def post_process_predictions(
        self,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        img_metas: List[Dict],
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> List[Dict]:
        """Post-process model predictions.
        
        Args:
            pred_logits: [B, num_queries, num_classes+1]
            pred_masks: [B, num_queries, H, W]
            img_metas: List of image metadata
            score_threshold: Confidence score threshold
            mask_threshold: Mask binarization threshold
            
        Returns:
            List of predictions per image
        """
        B, num_queries, num_classes_plus_one = pred_logits.shape
        num_classes = num_classes_plus_one - 1
        
        # Apply softmax to get probabilities
        pred_probs = F.softmax(pred_logits, dim=-1)  # [B, num_queries, num_classes+1]
        
        # Apply sigmoid to masks
        pred_masks_sigmoid = torch.sigmoid(pred_masks)  # [B, num_queries, H, W]
        
        results = []
        
        for b in range(B):
            img_meta = img_metas[b]
            ori_h, ori_w = img_meta.get('ori_shape', img_meta.get('img_shape', (1024, 1024)))
            
            # Get class probabilities (excluding background/no-object class)
            class_probs = pred_probs[b, :, :num_classes]  # [num_queries, num_classes]
            max_scores, pred_labels = torch.max(class_probs, dim=1)  # [num_queries]
            
            # Filter by score threshold
            valid_mask = max_scores > score_threshold
            valid_indices = torch.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                results.append({
                    'masks': [],
                    'labels': [],
                    'scores': [],
                })
                continue
            
            # Get valid predictions
            valid_scores = max_scores[valid_indices].cpu().numpy()
            valid_labels = (pred_labels[valid_indices] + 1).cpu().numpy()  # Convert to 1-indexed
            valid_masks = pred_masks_sigmoid[b, valid_indices]  # [num_valid, H, W]
            
            # Resize masks to original image size
            if valid_masks.shape[-2:] != (ori_h, ori_w):
                valid_masks = F.interpolate(
                    valid_masks.unsqueeze(1),
                    size=(ori_h, ori_w),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(1)
            
            # Binarize masks
            valid_masks_binary = (valid_masks > mask_threshold).cpu().numpy()
            
            # Convert to list of numpy arrays
            masks_list = [valid_masks_binary[i] for i in range(len(valid_indices))]
            labels_list = valid_labels.tolist()
            scores_list = valid_scores.tolist()
            
            results.append({
                'masks': masks_list,
                'labels': labels_list,
                'scores': scores_list,
            })
        
        return results
    
    def extract_metadata(self, img_meta: Dict) -> Tuple[str, str]:
        """Extract scene type and resolution from metadata.
        
        Args:
            img_meta: Image metadata dictionary
            
        Returns:
            Tuple of (scene_type, resolution)
        """
        # Get scene type
        scene_type = img_meta.get('scene_type', 'rural_area')
        if not scene_type or scene_type == 'rural_area':
            # Try to extract from filename or other metadata
            filename = img_meta.get('filename', '')
            # Default to rural_area if not found
            scene_type = 'rural_area'
        
        # Get resolution
        resolution = str(img_meta.get('cm_resolution', img_meta.get('resolution', 40)))
        if not resolution or resolution == '40':
            # Try to extract from filename (e.g., "60cm_train_122.tif")
            filename = img_meta.get('filename', '')
            if 'cm' in filename.lower():
                try:
                    # Extract number before "cm"
                    match = re.search(r'(\d+)cm', filename.lower())
                    if match:
                        resolution = match.group(1)
                except:
                    pass
            if not resolution:
                resolution = '40'
        
        return scene_type, resolution
    
    def evaluate_batch(
        self,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        gt_labels: List[torch.Tensor],
        gt_masks: List[torch.Tensor],
        img_metas: List[Dict],
        img_prefix: str = '',
        device: str = 'cuda',
        return_images: bool = False,
    ):
        """Evaluate a batch of predictions.
        
        Args:
            pred_logits: [B, num_queries, num_classes+1]
            pred_masks: [B, num_queries, H, W]
            gt_labels: List of ground truth labels [num_instances] per image
            gt_masks: List of ground truth masks [num_instances, H, W] per image
            img_metas: List of image metadata
            img_prefix: Path prefix for loading original images
            device: Device to run evaluation on
            return_images: Whether to return visualization data
            
        Returns:
            List of visualization dicts if return_images=True
        """
        import cv2
        import os
        
        # Post-process predictions
        pred_results = self.post_process_predictions(
            pred_logits,
            pred_masks,
            img_metas,
        )
        
        val_images = []
        
        # Process each image in batch
        for i, pred_result in enumerate(pred_results):
            # Get ground truth for this image
            gt_labels_i = gt_labels[i].cpu().numpy()
            gt_masks_i = gt_masks[i].cpu().numpy()  # [num_instances, H, W]
            
            # Resize GT masks to original size if needed
            img_meta = img_metas[i]
            ori_h, ori_w = img_meta.get('ori_shape', img_meta.get('img_shape', (1024, 1024)))
            
            if gt_masks_i.shape[0] > 0 and gt_masks_i.shape[-2:] != (ori_h, ori_w):
                gt_masks_i = F.interpolate(
                    torch.from_numpy(gt_masks_i).unsqueeze(1).float(),
                    size=(ori_h, ori_w),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(1).numpy()
            
            # Convert to list
            gt_masks_list = [gt_masks_i[j] for j in range(len(gt_labels_i))]
            gt_labels_list = (gt_labels_i + 1).tolist()  # Convert to 1-indexed
            
            # Extract scene type and resolution
            scene_type, resolution = self.extract_metadata(img_meta)
            
            # Add to evaluator
            self.add_prediction(
                pred_masks=pred_result['masks'],
                pred_labels=pred_result['labels'],
                pred_scores=pred_result['scores'],
                gt_masks=gt_masks_list,
                gt_labels=gt_labels_list,
                scene_type=scene_type,
                resolution=resolution,
            )
            
            # Prepare visualization data if requested
            if return_images:
                # Load original image from disk
                filename = img_meta.get('filename', '')
                if filename and img_prefix:
                    img_path = os.path.join(img_prefix, filename)
                    if os.path.exists(img_path):
                        # Load original image
                        original_img = cv2.imread(img_path)
                        if original_img is not None:
                            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                            # Resize to original shape if needed
                            if original_img.shape[:2] != (ori_h, ori_w):
                                original_img = cv2.resize(original_img, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
                            
                            val_images.append({
                                'image': original_img,
                                'pred_masks': pred_result['masks'],
                                'gt_masks': gt_masks_list,
                                'pred_labels': pred_result['labels'],
                                'gt_labels': gt_labels_list,
                            })
        
        if return_images:
            return val_images
        return None
    
    def compute_weighted_map(self) -> Dict[str, float]:
        """Compute weighted mAP across all images.
        
        Returns:
            Dict with 'weighted_map' and per-class information
        """
        if len(self.image_weights) == 0:
            return {
                'weighted_map': 0.0,
                'mean_map': 0.0,
                'class_aps': [0.0, 0.0],
            }
        
        # Re-evaluate all images to get current mAPs
        maps = []
        all_class_aps = [[] for _ in range(len(self.CLASS_NAMES))]
        
        for i in range(len(self.predictions)):
            pred = self.predictions[i]
            gt = self.ground_truths[i]
            
            # Get scene type and resolution from metadata if available
            scene_type = "rural_area"
            resolution = "40"
            
            result = self.evaluate_image(
                pred['masks'], pred['labels'], pred['scores'],
                gt['masks'], gt['labels'],
                scene_type, resolution
            )
            
            maps.append(result['map'])
            for j, ap in enumerate(result['class_aps']):
                all_class_aps[j].append(ap)
        
        # Compute weighted mAP
        weights = np.array(self.image_weights)
        maps = np.array(maps)
        
        weighted_map = np.sum(weights * maps) / np.sum(weights) if np.sum(weights) > 0 else 0.0
        mean_map = np.mean(maps)
        
        # Per-class AP (mean across images)
        class_aps = [np.mean(aps) if len(aps) > 0 else 0.0 for aps in all_class_aps]
        
        return {
            'weighted_map': float(weighted_map),
            'mean_map': float(mean_map),
            'class_aps': [float(ap) for ap in class_aps],
            'class_names': self.CLASS_NAMES,
        }

