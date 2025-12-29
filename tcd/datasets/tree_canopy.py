"""
Tree Canopy Dataset for instance segmentation.
Supports COCO format annotations.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2


class TreeCanopyDataset(Dataset):
    """
    Tree Canopy Dataset for instance segmentation.
    
    Args:
        ann_file: Path to COCO format annotation file
        img_prefix: Path to image directory
        pipeline: List of data augmentation transforms
        test_mode: If True, dataset is in test mode (no annotations)
        filter_empty_gt: If True, filter images without annotations
    """
    
    CLASSES = ('individual_tree', 'group_of_trees')
    
    def __init__(
        self,
        ann_file: str,
        img_prefix: str,
        pipeline: Optional[List[Dict[str, Any]]] = None,
        test_mode: bool = False,
        filter_empty_gt: bool = True,
    ):
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.pipeline = pipeline if pipeline is not None else []
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        
        # Load annotations
        self.data_infos = self._load_annotations()
        
        # Filter empty GT if needed
        if filter_empty_gt and not test_mode:
            self.data_infos = self._filter_empty_gt()
    
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load annotations from COCO format file."""
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build image info list
        images = coco_data.get('images', [])
        annotations = coco_data.get('annotations', [])
        categories = coco_data.get('categories', [])
        
        # Create category mapping
        self.cat_ids = [cat['id'] for cat in categories]
        self.cat2label = {cat['id']: idx + 1 for idx, cat in enumerate(categories)}
        self.label2cat = {idx + 1: cat['id'] for idx, cat in enumerate(categories)}
        
        # Group annotations by image_id
        img_to_anns = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # Build data infos
        data_infos = []
        for img_info in images:
            img_id = img_info['id']
            data_info = {
                'id': img_id,
                'filename': img_info['file_name'],
                'width': img_info['width'],
                'height': img_info['height'],
                'annotations': img_to_anns.get(img_id, []),
            }
            
            # Add metadata if available
            if 'scene_type' in img_info:
                data_info['scene_type'] = img_info['scene_type']
            if 'cm_resolution' in img_info:
                data_info['cm_resolution'] = img_info['cm_resolution']
            
            data_infos.append(data_info)
        
        return data_infos
    
    def _filter_empty_gt(self) -> List[Dict[str, Any]]:
        """Filter images without annotations."""
        return [info for info in self.data_infos if len(info['annotations']) > 0]
    
    def __len__(self) -> int:
        return len(self.data_infos)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get data item by index."""
        data_info = self.data_infos[idx]
        
        # Load image
        img_path = os.path.join(self.img_prefix, data_info['filename'])
        img = self._load_image(img_path)
        
        # Prepare data dict
        data = {
            'img': img,
            'img_shape': img.shape[:2],
            'ori_shape': img.shape[:2],
        }
        
        # Load annotations if not in test mode
        if not self.test_mode:
            ann_data = self._load_annotations_for_image(data_info)
            data.update(ann_data)
        else:
            data['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
            data['gt_labels'] = np.zeros((0,), dtype=np.int64)
            data['gt_masks'] = []
            data['img_metas'] = {
                'img_id': data_info['id'],
                'filename': data_info['filename'],
                'ori_shape': img.shape[:2],
            }
        
        # Apply pipeline (data augmentation)
        data = self._apply_pipeline(data)
        
        return data
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """Load image from file."""
        # Try OpenCV first (faster)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        
        # Fallback to PIL
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img = np.array(img)
        return img
    
    def _load_annotations_for_image(self, data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Load annotations for a single image."""
        annotations = data_info['annotations']
        
        if len(annotations) == 0:
            return {
                'gt_bboxes': np.zeros((0, 4), dtype=np.float32),
                'gt_labels': np.zeros((0,), dtype=np.int64),
                'gt_masks': [],
                'gt_areas': np.zeros((0,), dtype=np.float32),
            }
        
        gt_bboxes = []
        gt_labels = []
        gt_masks = []
        gt_areas = []
        
        for ann in annotations:
            # Get category label
            cat_id = ann['category_id']
            label = self.cat2label.get(cat_id, 0)
            if label == 0:
                continue
            
            # Get bounding box
            bbox = ann['bbox']  # [x, y, w, h]
            gt_bboxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            
            # Get label
            gt_labels.append(label)
            
            # Get segmentation mask
            segmentation = ann.get('segmentation', [])
            if segmentation:
                gt_masks.append(segmentation)
            else:
                gt_masks.append([])
            
            # Get area
            area = ann.get('area', 0.0)
            gt_areas.append(area)
        
        return {
            'gt_bboxes': np.array(gt_bboxes, dtype=np.float32),
            'gt_labels': np.array(gt_labels, dtype=np.int64),
            'gt_masks': gt_masks,
            'gt_areas': np.array(gt_areas, dtype=np.float32),
        }
    
    def _apply_pipeline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data augmentation pipeline."""
        from .transforms import Compose
        
        # Build compose from pipeline config
        compose = Compose(self.pipeline)
        return compose(data)
    
    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category IDs for an image."""
        data_info = self.data_infos[idx]
        annotations = data_info['annotations']
        return [ann['category_id'] for ann in annotations]

