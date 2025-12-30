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
            image = img_data['image']  # [H, W, 3] numpy array (RGB)
            pred_masks = img_data.get('pred_masks', [])
            gt_masks = img_data.get('gt_masks', [])
            pred_labels = img_data.get('pred_labels', [])
            gt_labels = img_data.get('gt_labels', [])
            
            # Debug: ensure masks are lists and handle different formats
            if not isinstance(gt_masks, list):
                if isinstance(gt_masks, np.ndarray):
                    if gt_masks.ndim == 3:
                        gt_masks = [gt_masks[j] for j in range(gt_masks.shape[0])]
                    elif gt_masks.ndim == 2:
                        gt_masks = [gt_masks]
                    else:
                        gt_masks = []
                elif isinstance(gt_masks, torch.Tensor):
                    gt_masks_np = gt_masks.cpu().numpy()
                    if gt_masks_np.ndim == 3:
                        gt_masks = [gt_masks_np[j] for j in range(gt_masks_np.shape[0])]
                    elif gt_masks_np.ndim == 2:
                        gt_masks = [gt_masks_np]
                    else:
                        gt_masks = []
                else:
                    gt_masks = []
            
            if not isinstance(pred_masks, list):
                if isinstance(pred_masks, np.ndarray):
                    if pred_masks.ndim == 3:
                        pred_masks = [pred_masks[j] for j in range(pred_masks.shape[0])]
                    elif pred_masks.ndim == 2:
                        pred_masks = [pred_masks]
                    else:
                        pred_masks = []
                elif isinstance(pred_masks, torch.Tensor):
                    pred_masks_np = pred_masks.cpu().numpy()
                    if pred_masks_np.ndim == 3:
                        pred_masks = [pred_masks_np[j] for j in range(pred_masks_np.shape[0])]
                    elif pred_masks_np.ndim == 2:
                        pred_masks = [pred_masks_np]
                    else:
                        pred_masks = []
                else:
                    pred_masks = []
            
            # Ensure labels match masks length
            if len(gt_labels) != len(gt_masks):
                gt_labels = gt_labels[:len(gt_masks)] if len(gt_labels) > len(gt_masks) else gt_labels + [1] * (len(gt_masks) - len(gt_labels))
            if len(pred_labels) != len(pred_masks):
                pred_labels = pred_labels[:len(pred_masks)] if len(pred_labels) > len(pred_masks) else pred_labels + [1] * (len(pred_masks) - len(pred_labels))
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Overlay GT masks on image using OpenCV
            gt_overlay = image_bgr.copy()
            for mask, label in zip(gt_masks, gt_labels):
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                
                # Ensure mask is 2D [H, W]
                if mask.ndim > 2:
                    mask = mask.squeeze()
                
                # Ensure mask matches image dimensions
                if mask.shape[:2] != gt_overlay.shape[:2]:
                    mask = cv2.resize(mask, (gt_overlay.shape[1], gt_overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Convert mask to binary [H, W] float32 for blending
                if mask.max() <= 1.0:
                    mask_binary = (mask > 0.5).astype(np.float32)
                else:
                    mask_binary = (mask > 127).astype(np.float32)
                
                # Skip if mask is empty
                if mask_binary.sum() == 0:
                    continue
                
                color = class_colors.get(int(label), (0, 255, 255))
                # Create colored overlay
                color_overlay = np.zeros_like(gt_overlay, dtype=np.float32)
                color_overlay[:, :] = color
                
                # Blend: result = image * (1 - alpha*mask) + color * alpha*mask
                alpha = 0.25  # More transparent
                mask_3d = np.stack([mask_binary] * 3, axis=2)
                gt_overlay = (gt_overlay.astype(np.float32) * (1 - alpha * mask_3d) + 
                             color_overlay * alpha * mask_3d).astype(np.uint8)
            
            # Overlay predicted masks on image using OpenCV
            pred_overlay = image_bgr.copy()
            for mask, label in zip(pred_masks, pred_labels):
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                
                # Ensure mask is 2D [H, W]
                if mask.ndim > 2:
                    mask = mask.squeeze()
                
                # Ensure mask matches image dimensions
                if mask.shape[:2] != pred_overlay.shape[:2]:
                    mask = cv2.resize(mask, (pred_overlay.shape[1], pred_overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Convert mask to binary [H, W] float32 for blending
                if mask.max() <= 1.0:
                    mask_binary = (mask > 0.5).astype(np.float32)
                else:
                    mask_binary = (mask > 127).astype(np.float32)
                
                # Skip if mask is empty
                if mask_binary.sum() == 0:
                    continue
                
                color = class_colors.get(int(label), (0, 255, 255))
                # Create colored overlay
                color_overlay = np.zeros_like(pred_overlay, dtype=np.float32)
                color_overlay[:, :] = color
                
                # Blend: result = image * (1 - alpha*mask) + color * alpha*mask
                alpha = 0.25  # More transparent
                mask_3d = np.stack([mask_binary] * 3, axis=2)
                pred_overlay = (pred_overlay.astype(np.float32) * (1 - alpha * mask_3d) + 
                               color_overlay * alpha * mask_3d).astype(np.uint8)
            
            # Add text labels to BGR images before merging
            cv2.putText(gt_overlay, 'Ground Truth', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(pred_overlay, 'Predictions', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Merge GT and predictions side by side (still in BGR)
            merged_image_bgr = np.hstack([gt_overlay, pred_overlay])
            
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

