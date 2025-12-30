"""
Data transforms for Tree Canopy Detection dataset.
"""

import numpy as np
import cv2
import torch
from typing import Dict, List, Tuple, Optional, Any
import random
from PIL import Image
from tcd.utils import Registry

# Create transform registry
TRANSFORM = Registry('transform')


@TRANSFORM.register_module()
class LoadImageFromFile:
    """Load image from file."""
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load image."""
        if 'img' not in data:
            # Image should already be loaded in dataset
            pass
        return data


@TRANSFORM.register_module()
class LoadAnnotations:
    """Load annotations (bboxes, labels, masks).
    
    Generates original masks from polygons and stores as ori_gt_masks.
    Keeps gt_masks as polygons for transforms.
    """
    
    def __init__(self, with_bbox: bool = True, with_mask: bool = True):
        self.with_bbox = with_bbox
        self.with_mask = with_mask
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load annotations and generate original masks from polygons."""
        if self.with_mask and 'gt_masks' in data and len(data['gt_masks']) > 0:
            # Get original image size
            ori_h, ori_w = data.get('ori_shape', data.get('img_shape', (1024, 1024)))
            
            # Generate masks from polygons at original size
            ori_gt_masks = []
            for mask in data['gt_masks']:
                if isinstance(mask, list) and len(mask) > 0:
                    # Polygon format - convert to mask at original size
                    mask_array = np.zeros((ori_h, ori_w), dtype=np.float32)
                    if isinstance(mask[0], list):
                        # Multiple polygons per mask
                        for poly in mask:
                            coords = np.array(poly).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(mask_array, [coords], 1.0)
                    else:
                        # Single polygon: [x1, y1, x2, y2, ...]
                        coords = np.array(mask).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask_array, [coords], 1.0)
                    ori_gt_masks.append(mask_array)
                else:
                    # Empty polygon
                    ori_gt_masks.append(np.zeros((ori_h, ori_w), dtype=np.float32))
            
            # Store original masks (at original size)
            data['ori_gt_masks'] = ori_gt_masks
        else:
            # No masks or empty
            data['ori_gt_masks'] = []
        
        # Keep gt_masks as polygons (for transforms)
        return data


@TRANSFORM.register_module()
class Resize:
    """Resize image and masks."""
    
    def __init__(
        self,
        img_scale: Optional[Tuple[int, int]] = None,
        multiscale_mode: str = 'value',
        keep_ratio: bool = True,
        scale: Optional[Tuple[int, int]] = None,
    ):
        if scale is not None:
            img_scale = scale
        if isinstance(img_scale, list):
            self.img_scale = img_scale
        else:
            self.img_scale = [img_scale] if img_scale else [(1024, 1024)]
        self.multiscale_mode = multiscale_mode
        self.keep_ratio = keep_ratio
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Resize image and annotations."""
        img = data['img']
        h, w = img.shape[:2]
        
        # Select scale
        if self.multiscale_mode == 'range':
            scale_idx = random.randint(0, len(self.img_scale) - 1)
        else:
            scale_idx = 0
        
        new_h, new_w = self.img_scale[scale_idx]
        
        if self.keep_ratio:
            scale = min(new_h / h, new_w / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
        
        # Resize image
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        data['img'] = img
        data['img_shape'] = (new_h, new_w)
        data['scale_factor'] = (new_h / h, new_w / w)
        
        # Resize masks (polygons) if present - scale coordinates to match resized image
        # Calculate scale factors for polygon coordinate scaling
        scale_w = new_w / w if w > 0 else 1.0
        scale_h = new_h / h if h > 0 else 1.0
        
        if 'gt_masks' in data:
            # Scale polygons (keep as polygons, don't generate masks yet)
            resized_masks = []
            for mask in data['gt_masks']:
                if isinstance(mask, list):
                    # Polygon format - scale coordinates to match resized image
                    if len(mask) > 0:
                        if isinstance(mask[0], list):
                            # Multiple polygons per mask
                            scaled_polys = []
                            for poly in mask:
                                # Scale polygon coordinates: [x1, y1, x2, y2, ...]
                                scaled_poly = []
                                for i in range(0, len(poly), 2):
                                    if i + 1 < len(poly):
                                        scaled_poly.append(poly[i] * scale_w)      # x coordinate
                                        scaled_poly.append(poly[i + 1] * scale_h)  # y coordinate
                                scaled_polys.append(scaled_poly)
                            resized_masks.append(scaled_polys)
                        else:
                            # Single polygon: [x1, y1, x2, y2, ...]
                            scaled_poly = []
                            for i in range(0, len(mask), 2):
                                if i + 1 < len(mask):
                                    scaled_poly.append(mask[i] * scale_w)      # x coordinate
                                    scaled_poly.append(mask[i + 1] * scale_h)  # y coordinate
                            resized_masks.append(scaled_poly)
                    else:
                        # Empty polygon
                        resized_masks.append([])
                else:
                    raise TypeError(f"Expected polygon (list), but got {type(mask)}. "
                                  f"gt_masks should be polygons before Polygon2Mask transform.")
            data['gt_masks'] = resized_masks
        
        # Keep original polygons unchanged (for evaluation/visualization)
        # ori_gt_masks is preserved as-is through all transforms
        
        return data


@TRANSFORM.register_module()
class RandomFlip:
    """Randomly flip image and annotations."""
    
    def __init__(self, prob: float = 0.5, flip_ratio: float = None, direction: str = 'horizontal'):
        # Support both 'prob' and 'flip_ratio' for compatibility
        self.prob = flip_ratio if flip_ratio is not None else prob
        self.direction = direction
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Flip image and annotations."""
        # ori_gt_masks (original masks) are preserved as-is through all transforms
        if random.random() < self.prob:
            img = data['img']
            h, w = img.shape[:2]
            
            if self.direction == 'horizontal':
                img = cv2.flip(img, 1)
                data['img'] = img
                
                # Flip bboxes
                if 'gt_bboxes' in data and len(data['gt_bboxes']) > 0:
                    bboxes = data['gt_bboxes'].copy()
                    bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
                    data['gt_bboxes'] = bboxes
                
                # Flip masks (polygons) - keep as polygons, don't generate masks yet
                if 'gt_masks' in data:
                    flipped_masks = []
                    for mask in data['gt_masks']:
                        if isinstance(mask, list):
                            # Polygon: flip x coordinates
                            if len(mask) > 0:
                                if isinstance(mask[0], list):
                                    # Multiple polygons per mask
                                    flipped_polys = []
                                    for poly in mask:
                                        # Flip x coordinates: [x1, y1, x2, y2, ...] -> [w-x1, y1, w-x2, y2, ...]
                                        flipped_poly = []
                                        for i in range(0, len(poly), 2):
                                            if i + 1 < len(poly):
                                                flipped_poly.append(w - poly[i])  # flipped x
                                                flipped_poly.append(poly[i + 1])  # y unchanged
                                        flipped_polys.append(flipped_poly)
                                    flipped_masks.append(flipped_polys)
                                else:
                                    # Single polygon: [x1, y1, x2, y2, ...]
                                    flipped_poly = []
                                    for i in range(0, len(mask), 2):
                                        if i + 1 < len(mask):
                                            flipped_poly.append(w - mask[i])  # flipped x
                                            flipped_poly.append(mask[i + 1])  # y unchanged
                                    flipped_masks.append(flipped_poly)
                            else:
                                # Empty polygon
                                flipped_masks.append([])
                        else:
                            raise TypeError(f"Expected polygon (list), but got {type(mask)}. "
                                          f"gt_masks should be polygons before Polygon2Mask transform.")
                    data['gt_masks'] = flipped_masks
            
            elif self.direction == 'vertical':
                img = cv2.flip(img, 0)
                data['img'] = img
                
                # Flip bboxes
                if 'gt_bboxes' in data and len(data['gt_bboxes']) > 0:
                    bboxes = data['gt_bboxes'].copy()
                    bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]]
                    data['gt_bboxes'] = bboxes
        
        return data


@TRANSFORM.register_module()
class ColorJitter:
    """Apply color jitter augmentation."""
    
    def __init__(self, brightness: float = 0.0, contrast: float = 0.0, saturation: float = 0.0, hue: float = 0.0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply color jitter."""
        img = data['img'].astype(np.float32)
        
        # Brightness
        if self.brightness > 0:
            alpha = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = img * alpha
            img = np.clip(img, 0, 255)
        
        # Contrast
        if self.contrast > 0:
            alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = img.mean(axis=(0, 1), keepdims=True)
            img = (img - mean) * alpha + mean
            img = np.clip(img, 0, 255)
        
        # Saturation
        if self.saturation > 0:
            img_hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            alpha = 1.0 + random.uniform(-self.saturation, self.saturation)
            img_hsv[:, :, 1] = img_hsv[:, :, 1] * alpha
            img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
            img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Hue
        if self.hue > 0:
            img_hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            alpha = random.uniform(-self.hue, self.hue) * 180
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + alpha) % 180
            img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        data['img'] = img.astype(np.uint8)
        return data


@TRANSFORM.register_module()
class Normalize:
    """Normalize image."""
    
    def __init__(self, mean: List[float], std: List[float], to_rgb: bool = True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize image."""
        img = data['img'].astype(np.float32)
        
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = (img - self.mean) / self.std
        data['img'] = img
        return data


@TRANSFORM.register_module()
class Pad:
    """Pad image to be divisible by size_divisor."""
    
    def __init__(self, size_divisor: int = 32, pad_val: float = 0.0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Pad image."""
        img = data['img']
        h, w = img.shape[:2]
        
        pad_h = (self.size_divisor - h % self.size_divisor) % self.size_divisor
        pad_w = (self.size_divisor - w % self.size_divisor) % self.size_divisor
        
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(
                img,
                0, pad_h, 0, pad_w,
                cv2.BORDER_CONSTANT,
                value=self.pad_val,
            )
            data['img'] = img
            data['pad_shape'] = (h + pad_h, w + pad_w)
        
        return data


@TRANSFORM.register_module()
class Polygon2Mask:
    """Convert polygons to masks.
    
    This transform should be placed right before DefaultFormatBundle.
    It converts gt_masks from polygon format to mask format (numpy arrays).
    """
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert polygons to masks."""
        if 'gt_masks' in data and len(data['gt_masks']) > 0:
            # Get current image size
            img = data['img']
            h, w = img.shape[:2]
            
            # Convert polygons to masks
            mask_arrays = []
            for mask in data['gt_masks']:
                if isinstance(mask, list):
                    # Polygon format - convert to mask
                    mask_array = np.zeros((h, w), dtype=np.float32)
                    if len(mask) > 0:
                        if isinstance(mask[0], list):
                            # Multiple polygons per mask
                            for poly in mask:
                                coords = np.array(poly).reshape(-1, 2).astype(np.int32)
                                cv2.fillPoly(mask_array, [coords], 1.0)
                        else:
                            # Single polygon: [x1, y1, x2, y2, ...]
                            coords = np.array(mask).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(mask_array, [coords], 1.0)
                    mask_arrays.append(mask_array)
                else:
                    raise TypeError(f"Expected polygon (list), but got {type(mask)}. "
                                  f"gt_masks should be polygons before Polygon2Mask transform.")
            data['gt_masks'] = mask_arrays
        else:
            data['gt_masks'] = []
        
        return data


@TRANSFORM.register_module()
class DefaultFormatBundle:
    """Format data to default format (tensors)."""
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data."""
        # Convert image to tensor
        img = data['img']
        if isinstance(img, np.ndarray):
            # HWC to CHW
            if img.ndim == 3:
                img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()
        data['img'] = img
        
        # Convert bboxes to tensor
        if 'gt_bboxes' in data and len(data['gt_bboxes']) > 0:
            data['gt_bboxes'] = torch.from_numpy(data['gt_bboxes']).float()
        else:
            data['gt_bboxes'] = torch.zeros((0, 4), dtype=torch.float32)
        
        # Convert labels to tensor
        if 'gt_labels' in data and len(data['gt_labels']) > 0:
            data['gt_labels'] = torch.from_numpy(data['gt_labels']).long()
        else:
            data['gt_labels'] = torch.zeros((0,), dtype=torch.long)
        
        # Convert masks to tensors (masks are already numpy arrays from Polygon2Mask)
        if 'gt_masks' in data and len(data['gt_masks']) > 0:
            # Masks should be numpy arrays [H, W] from Polygon2Mask transform
            mask_tensors = []
            for mask in data['gt_masks']:
                if isinstance(mask, np.ndarray):
                    mask_tensors.append(torch.from_numpy(mask).float())
                else:
                    raise TypeError(f"Expected mask (numpy array), but got {type(mask)}. "
                                  f"gt_masks should be masks (numpy arrays) from Polygon2Mask transform.")
            data['gt_masks'] = mask_tensors
        else:
            data['gt_masks'] = []
        
        return data


@TRANSFORM.register_module()
class Collect:
    """Collect data from the loader."""
    
    def __init__(self, keys: List[str]):
        self.keys = keys
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect specified keys."""
        result = {}
        for k in self.keys:
            if k in data:
                result[k] = data[k]
        # Always include img_metas if available
        if 'img_metas' in data:
            result['img_metas'] = data['img_metas']
        return result


@TRANSFORM.register_module()
class ImageToTensor:
    """Convert image to tensor."""
    
    def __init__(self, keys: List[str] = ['img']):
        self.keys = keys
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert image to tensor."""
        for key in self.keys:
            if key in data:
                img = data[key]
                if isinstance(img, np.ndarray):
                    if img.ndim == 3:
                        img = img.transpose(2, 0, 1)
                    img = torch.from_numpy(img).float()
                    data[key] = img
        return data


@TRANSFORM.register_module()
class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: List[Dict[str, Any]]):
        """Initialize Compose.
        
        Args:
            transforms: List of transform configs, each with 'type' key
        """
        from tcd.utils import build_from_cfg
        
        self.transforms = []
        for transform_cfg in transforms:
            if isinstance(transform_cfg, dict):
                transform = build_from_cfg(transform_cfg, TRANSFORM)
                self.transforms.append(transform)
            else:
                # Direct transform object
                self.transforms.append(transform_cfg)
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transforms sequentially."""
        for transform in self.transforms:
            data = transform(data)
        return data
