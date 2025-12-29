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
        # Color mapping
        class_colors = {
            1: np.array([0, 255, 0]),      # Green for individual_tree
            2: np.array([255, 0, 0]),      # Red for group_of_trees
        }
        
        for i, img_data in enumerate(val_images[:self.num_eval_images]):
            image = img_data['image']  # [H, W, 3] numpy array
            pred_masks = img_data.get('pred_masks', [])
            gt_masks = img_data.get('gt_masks', [])
            pred_labels = img_data.get('pred_labels', [])
            gt_labels = img_data.get('gt_labels', [])
            
            # Overlay GT masks on image
            gt_overlay = image.copy()
            for mask, label in zip(gt_masks, gt_labels):
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                # Convert mask to binary [H, W]
                if mask.max() <= 1.0:
                    mask_binary = (mask > 0.5).astype(np.float32)
                else:
                    mask_binary = (mask > 127).astype(np.float32)
                
                color = class_colors.get(int(label), np.array([255, 255, 0]))
                # Overlay mask: image = image * (1 - alpha) + color * alpha where mask is 1
                alpha = 0.4
                for c in range(3):
                    gt_overlay[:, :, c] = gt_overlay[:, :, c] * (1 - mask_binary * alpha) + color[c] * mask_binary * alpha
            
            # Overlay predicted masks on image
            pred_overlay = image.copy()
            for mask, label in zip(pred_masks, pred_labels):
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                # Convert mask to binary [H, W]
                if mask.max() <= 1.0:
                    mask_binary = (mask > 0.5).astype(np.float32)
                else:
                    mask_binary = (mask > 127).astype(np.float32)
                
                color = class_colors.get(int(label), np.array([255, 255, 0]))
                # Overlay mask: image = image * (1 - alpha) + color * alpha where mask is 1
                alpha = 0.4
                for c in range(3):
                    pred_overlay[:, :, c] = pred_overlay[:, :, c] * (1 - mask_binary * alpha) + color[c] * mask_binary * alpha
            
            # Convert to uint8
            gt_overlay = np.clip(gt_overlay, 0, 255).astype(np.uint8)
            pred_overlay = np.clip(pred_overlay, 0, 255).astype(np.uint8)
            
            # Log each image separately to val_visualization group (creates separate tab)
            # Each image will be displayed as equal size like charts
            self.wandb.log({
                f'val_visualization/gt_epoch{epoch}_img{i+1}': self.wandb.Image(gt_overlay, caption=f'Ground Truth - Epoch {epoch}, Image {i+1}'),
                f'val_visualization/pred_epoch{epoch}_img{i+1}': self.wandb.Image(pred_overlay, caption=f'Predictions - Epoch {epoch}, Image {i+1}'),
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

