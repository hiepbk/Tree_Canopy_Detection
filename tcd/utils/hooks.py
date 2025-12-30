"""
Training hooks for logging and monitoring.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import os
import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import cv2
import io


class Hook:
    """Base class for hooks."""
    
    def __init__(self, priority: int = 0):
        self.priority = priority
    
    def before_train_epoch(self, runner):
        """Called before each training epoch."""
        pass
    
    def after_train_epoch(self, runner):
        """Called after each training epoch."""
        pass
    
    def before_train_iter(self, runner, batch_idx=None, data=None):
        """Called before each training iteration."""
        pass
    
    def after_train_iter(self, runner, batch_idx=None, data=None, outputs=None):
        """Called after each training iteration."""
        pass
    
    def before_val_epoch(self, runner):
        """Called before validation epoch."""
        pass
    
    def after_val_epoch(self, runner, metrics=None):
        """Called after validation epoch."""
        pass


class TextLoggerHook(Hook):
    """Text logger hook for console output."""
    
    def __init__(self, interval: int = 10, ignore_last: bool = False):
        super().__init__(priority=50)
        self.interval = interval
        self.ignore_last = ignore_last
    
    def before_train_epoch(self, runner):
        pass
    
    def after_train_epoch(self, runner):
        if runner.rank == 0:
            avg_loss = runner.train_losses[-1] if runner.train_losses else 0.0
            print(f'Epoch {runner.epoch} completed. Average loss: {avg_loss:.4f}')
    
    def before_train_iter(self, runner, batch_idx, data):
        pass
    
    def after_train_iter(self, runner, batch_idx, data, outputs):
        if runner.rank == 0 and batch_idx % self.interval == 0:
            if isinstance(outputs, dict):
                loss_str = ', '.join([f'{k}: {v.item():.4f}' for k, v in outputs.items()])
                total_loss = sum(outputs.values()).item()
                print(f'Epoch {runner.epoch}, Batch {batch_idx}, Total Loss: {total_loss:.4f} ({loss_str})')
            else:
                print(f'Epoch {runner.epoch}, Batch {batch_idx}, Loss: {outputs.item():.4f}')
    
    def before_val_epoch(self, runner):
        if runner.rank == 0:
            print(f'Evaluating on validation set...')
    
    def after_val_epoch(self, runner, metrics):
        if runner.rank == 0:
            print(f'Evaluation Results:')
            print(f'  Weighted mAP: {metrics.get("weighted_map", 0.0):.4f}')
            print(f'  Mean mAP: {metrics.get("mean_map", 0.0):.4f}')
            class_aps = metrics.get('class_aps', [])
            class_names = metrics.get('class_names', [])
            for name, ap in zip(class_names, class_aps):
                print(f'  {name} AP: {ap:.4f}')


class WandbHook(Hook):
    """Weights & Biases logging hook."""
    
    def __init__(self, 
                 init_kwargs: Optional[Dict] = None,
                 interval: int = 10,
                 log_checkpoint: bool = False,
                 log_checkpoint_metadata: bool = False,
                 num_eval_images: int = 5,
                 work_dir: str = ''):
        super().__init__(priority=100)
        self.interval = interval
        self.log_checkpoint = log_checkpoint
        self.log_checkpoint_metadata = log_checkpoint_metadata
        self.num_eval_images = num_eval_images
        self.work_dir = work_dir
        self._wandb = None
        self._init_kwargs = init_kwargs or {}
    
    @property
    def wandb(self):
        if self._wandb is None:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                raise ImportError('wandb is not installed. Please install it with: pip install wandb')
        return self._wandb
    
    def before_train_epoch(self, runner):
        if runner.rank == 0 and runner.iter == 0:
            # Initialize wandb on first iteration
            if not self.wandb.run:
                # Set wandb directory to experiment folder
                wandb_dir = os.path.join(self.work_dir, 'wandb')
                os.makedirs(wandb_dir, exist_ok=True)
                
                init_kwargs = self._init_kwargs.copy()
                init_kwargs['dir'] = wandb_dir
                
                self.wandb.init(**init_kwargs)
    
    def after_train_epoch(self, runner):
        if runner.rank == 0:
            # Log epoch-level metrics
            avg_loss = runner.train_losses[-1] if runner.train_losses else 0.0
            self.wandb.log({
                'epoch': runner.epoch,
                'train/loss': avg_loss,
            }, step=runner.iter)
    
    def before_train_iter(self, runner, batch_idx, data):
        pass
    
    def after_train_iter(self, runner, batch_idx, data, outputs):
        if runner.rank == 0 and batch_idx % self.interval == 0:
            # Log iteration-level metrics
            log_dict = {
                'train_iter/total_loss': sum(outputs.values()).item() if isinstance(outputs, dict) else outputs.item(),
            }
            
            if isinstance(outputs, dict):
                for k, v in outputs.items():
                    log_dict[f'train_iter/{k}'] = v.item()
            
            self.wandb.log(log_dict, step=runner.iter)
    
    def before_val_epoch(self, runner):
        pass
    
    def after_val_epoch(self, runner, metrics):
        if runner.rank == 0:
            # Log validation metrics
            val_log_dict = {
                'val/weighted_map': metrics.get('weighted_map', 0.0),
                'val/mean_map': metrics.get('mean_map', 0.0),
            }
            
            class_aps = metrics.get('class_aps', [])
            class_names = metrics.get('class_names', [])
            for name, ap in zip(class_names, class_aps):
                val_log_dict[f'val/{name}_ap'] = ap
            
            self.wandb.log(val_log_dict, step=runner.iter)
            
            # Log validation visualizations
            if 'val_images' in metrics and metrics['val_images']:
                self._log_val_images(metrics['val_images'], runner.epoch, runner)
    
    def _log_val_images(self, val_images: List[Dict], epoch: int, runner):
        """Log validation images with overlayed masks to wandb.
        
        Args:
            val_images: List of dicts with 'image', 'pred_masks', 'gt_masks', 'pred_labels', 'gt_labels'
            epoch: Current epoch number
        """
        import cv2
        
        # Color mapping (BGR for OpenCV)
        class_colors = {
            1: (0, 255, 0),      # Green for individual_tree
            2: (0, 0, 255),      # Red for group_of_trees
        }
        
        for i, img_data in enumerate(val_images[:self.num_eval_images]):
            image = img_data['image']  # [H, W, 3] numpy array (RGB) at original size
            pred_masks = img_data['pred_masks']  # List of numpy arrays [H, W] from evaluator
            gt_masks = img_data['gt_masks']  # List of numpy arrays [H, W] at original size from ori_gt_masks
            pred_labels = img_data['pred_labels']  # List of labels
            gt_labels = img_data['gt_labels']  # List of labels
            
            # Verify formats - no fallbacks
            if not isinstance(gt_masks, list):
                raise TypeError(f"Expected gt_masks to be a list, but got {type(gt_masks)}")
            if not isinstance(pred_masks, list):
                raise TypeError(f"Expected pred_masks to be a list, but got {type(pred_masks)}")
            
            # Verify mask formats
            for j, mask in enumerate(gt_masks):
                if not isinstance(mask, np.ndarray):
                    raise TypeError(f"Expected gt_masks[{j}] to be numpy array, but got {type(mask)}")
                if mask.ndim != 2:
                    raise ValueError(f"Expected gt_masks[{j}] to be 2D [H, W], but got shape {mask.shape}")
            
            for j, mask in enumerate(pred_masks):
                if not isinstance(mask, np.ndarray):
                    raise TypeError(f"Expected pred_masks[{j}] to be numpy array, but got {type(mask)}")
                if mask.ndim != 2:
                    raise ValueError(f"Expected pred_masks[{j}] to be 2D [H, W], but got shape {mask.shape}")
            
            # Verify labels match masks length
            if len(gt_labels) != len(gt_masks):
                raise ValueError(f"gt_labels length ({len(gt_labels)}) != gt_masks length ({len(gt_masks)})")
            if len(pred_labels) != len(pred_masks):
                raise ValueError(f"pred_labels length ({len(pred_labels)}) != pred_masks length ({len(pred_masks)})")
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img_h, img_w = image_bgr.shape[:2]
            
            # Verify image and masks are at same size (all at original size)
            for j, mask in enumerate(gt_masks):
                if mask.shape != (img_h, img_w):
                    raise ValueError(
                        f"GT mask {j} shape {mask.shape} doesn't match image shape {(img_h, img_w)}. "
                        f"Masks should be at original size from ori_gt_masks."
                    )
            
            # Overlay GT masks on image (masks are already at original size from ori_gt_masks)
            gt_overlay = image_bgr.copy().astype(np.float32)
            gt_mask_count = 0
            for mask, label in zip(gt_masks, gt_labels):
                # Mask is already numpy array [H, W] at original size (float32 from evaluator)
                # Masks from LoadAnnotations are generated with cv2.fillPoly(..., 1.0), so they're 0.0 or 1.0
                # But they might be in [0, 1] range, so check and ensure binary
                if mask.max() > 1.0:
                    mask_binary = (mask / 255.0 > 0.5).astype(np.float32)
                else:
                    # Already in [0, 1] range - threshold to get binary
                    # Use lower threshold to catch masks that might be slightly less than 1.0
                    mask_binary = (mask > 0.1).astype(np.float32)
                
                # Skip if mask is empty
                if mask_binary.sum() == 0:
                    continue
                
                gt_mask_count += 1
                
                color = class_colors.get(int(label), (0, 255, 255))
                # Create colored overlay [H, W, 3]
                mask_3d = np.stack([mask_binary] * 3, axis=2)
                color_overlay = np.zeros((img_h, img_w, 3), dtype=np.float32)
                color_overlay[:, :] = color
                
                # Transparent blending: result = image * (1 - alpha*mask) + color * alpha*mask
                # Use lower alpha for more transparency
                alpha = 0.3  # Transparency factor
                gt_overlay = gt_overlay * (1 - alpha * mask_3d) + color_overlay * alpha * mask_3d
            
            # Debug: log if no GT masks were found
            if gt_mask_count == 0 and len(gt_masks) > 0:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f'No GT masks were overlayed (found {len(gt_masks)} masks, but all were empty after thresholding)')
                # Log mask statistics for debugging
                for idx, mask in enumerate(gt_masks[:3]):  # Log first 3 masks
                    logger.warning(f'GT mask {idx}: shape={mask.shape}, dtype={mask.dtype}, min={mask.min()}, max={mask.max()}, sum={mask.sum()}')
            
            # Overlay predicted masks on image (resize to original size if needed)
            pred_overlay = image_bgr.copy().astype(np.float32)
            for mask, label in zip(pred_masks, pred_labels):
                # Mask is numpy array [H, W] - resize to original size if needed
                if mask.shape != (img_h, img_w):
                    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                
                # Mask is already binary (0 or 1) from post_process_predictions (boolean array)
                # Convert to float32 [0, 1]
                if mask.dtype == bool:
                    mask_binary = mask.astype(np.float32)
                elif mask.max() > 1.0:
                    mask_binary = (mask / 255.0 > 0.5).astype(np.float32)
                else:
                    mask_binary = (mask > 0.5).astype(np.float32)
                
                # Skip if mask is empty
                if mask_binary.sum() == 0:
                    continue
                
                color = class_colors.get(int(label), (0, 255, 255))
                # Create colored overlay [H, W, 3]
                mask_3d = np.stack([mask_binary] * 3, axis=2)
                color_overlay = np.zeros((img_h, img_w, 3), dtype=np.float32)
                color_overlay[:, :] = color
                
                # Transparent blending: result = image * (1 - alpha*mask) + color * alpha*mask
                # Use lower alpha for more transparency
                alpha = 0.3  # Transparency factor
                pred_overlay = pred_overlay * (1 - alpha * mask_3d) + color_overlay * alpha * mask_3d
            
            # Convert back to uint8
            gt_overlay = gt_overlay.astype(np.uint8)
            pred_overlay = pred_overlay.astype(np.uint8)
            
            # Add text labels to BGR images before merging
            cv2.putText(gt_overlay, 'Ground Truth', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(pred_overlay, 'Predictions', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Merge GT and predictions side by side (still in BGR, at original size)
            merged_image_bgr = np.hstack([gt_overlay, pred_overlay])
            
            # Resize final merged image if too large (for wandb display)
            # Keep aspect ratio, max dimension 2048
            max_dim = 2048
            h, w = merged_image_bgr.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                merged_image_bgr = cv2.resize(merged_image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Convert to RGB for wandb
            merged_image = cv2.cvtColor(merged_image_bgr, cv2.COLOR_BGR2RGB)
            
            # Use consistent key across steps for slider feature (wandb will create slider automatically)
            # Format: val_visualization/img{i+1} - same key for all epochs
            # Wandb will show a slider when you log the same key across different steps
            self.wandb.log({
                f'val_visualization/img{i+1}': self.wandb.Image(merged_image, caption=f'Epoch {epoch} - Image {i+1}'),
            }, step=runner.iter)
    
    def _get_mask_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Get all contour points from binary mask (same style as vis_data.py).
        
        Args:
            mask: Binary mask array [H, W]
            
        Returns:
            List of contour points, each as numpy array [N, 2]
        """
        if mask is None or mask.size == 0:
            return []
        
        # Ensure mask is uint8
        if mask.dtype != np.uint8:
            if mask.max() <= 1.0:
                mask = (mask > 0.5).astype(np.uint8) * 255
            else:
                mask = mask.astype(np.uint8)
        
        # Find all contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return []
        
        # Convert all contours to coordinate lists
        contour_list = []
        for contour in contours:
            if len(contour) >= 3:  # Need at least 3 points
                # Reshape to [N, 2]
                if contour.shape[1] == 1:
                    contour_reshaped = contour.reshape(-1, 2)
                else:
                    contour_reshaped = contour
                contour_list.append(contour_reshaped)
        
        return contour_list
    
    def _denormalize_image(self, img: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
        """Denormalize image for visualization.
        
        Args:
            img: Normalized image [C, H, W] or [H, W, C]
            mean: Mean values
            std: Std values
            
        Returns:
            Denormalized image [H, W, 3] in uint8
        """
        if img.ndim == 3 and img.shape[0] == 3:
            # CHW -> HWC
            img = img.transpose(1, 2, 0)
        
        mean = np.array(mean)
        std = np.array(std)
        
        img = img * std + mean
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

