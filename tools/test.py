"""
Testing/Evaluation script for Tree Canopy Detection.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
import json
import numpy as np
from tqdm import tqdm

from tcd.models import Mask2Former
from tcd.datasets import TreeCanopyDataset
from tcd.utils import Config
from tcd.configs import tcd_config


def parse_args():
    parser = argparse.ArgumentParser(description='Test Tree Canopy Detection model')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--out', help='Output result file')
    parser.add_argument('--eval', type=str, nargs='+', choices=['segm'], default=['segm'], help='Evaluation metrics')
    parser.add_argument('--show-dir', help='Directory to save visualization results')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    return parser.parse_args()


def build_model(cfg, checkpoint_path):
    """Build model and load checkpoint."""
    model = Mask2Former(
        backbone=cfg.model.backbone,
        neck=cfg.model.neck,
        head=cfg.model.head,
        train_cfg=cfg.model.get('train_cfg'),
        test_cfg=cfg.model.get('test_cfg'),
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda()
    model.eval()
    return model


def build_dataset(cfg, mode='test'):
    """Build test dataset."""
    if mode == 'test':
        dataset_cfg = cfg.data.test
    elif mode == 'val':
        dataset_cfg = cfg.data.val
    else:
        raise ValueError(f'Unknown mode: {mode}')
    
    dataset = TreeCanopyDataset(
        ann_file=dataset_cfg.ann_file,
        img_prefix=dataset_cfg.img_prefix,
        pipeline=dataset_cfg.pipeline,
        test_mode=True,
    )
    return dataset


def evaluate(model, dataloader, cfg):
    """Evaluate model on dataset."""
    results = []
    
    with torch.no_grad():
        for data in tqdm(dataloader):
            img = data['img'].cuda()
            img_metas = data['img_metas']
            
            # Forward
            outputs = model(
                img=img,
                img_metas=img_metas,
                return_loss=False,
            )
            
            # Process outputs
            pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes+1]
            pred_masks = outputs['pred_masks']  # [B, num_queries, H, W]
            
            # Get predictions
            pred_scores = torch.softmax(pred_logits, dim=-1)[..., :-1]  # Remove no-object class
            pred_labels = torch.argmax(pred_scores, dim=-1)  # [B, num_queries]
            pred_scores = torch.max(pred_scores, dim=-1)[0]  # [B, num_queries]
            
            # Apply sigmoid to masks
            pred_masks = torch.sigmoid(pred_masks)
            
            # Filter by score threshold
            score_thr = cfg.model.test_cfg.get('score_thr', 0.3)
            for b in range(pred_logits.shape[0]):
                valid = pred_scores[b] > score_thr
                if valid.sum() == 0:
                    results.append({
                        'image_id': img_metas[b]['img_id'],
                        'predictions': [],
                    })
                    continue
                
                # Get valid predictions
                valid_labels = pred_labels[b][valid].cpu().numpy()
                valid_scores = pred_scores[b][valid].cpu().numpy()
                valid_masks = pred_masks[b][valid].cpu().numpy()
                
                # Convert masks to polygons (simplified)
                predictions = []
                for label, score, mask in zip(valid_labels, valid_scores, valid_masks):
                    # Threshold mask
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    
                    # Find contours
                    import cv2
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if len(contours) > 0:
                        # Get largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(largest_contour) > 10:  # Filter small areas
                            # Convert to polygon
                            polygon = largest_contour.reshape(-1, 2).flatten().tolist()
                            
                            # Map label to class name
                            class_name = 'individual_tree' if label == 1 else 'group_of_trees'
                            
                            predictions.append({
                                'class': class_name,
                                'confidence_score': float(score),
                                'segmentation': polygon,
                            })
                
                results.append({
                    'image_id': img_metas[b]['img_id'],
                    'predictions': predictions,
                })
    
    return results


def compute_metrics(results, dataset, cfg):
    """Compute evaluation metrics."""
    # Simplified metric computation
    # In practice, use proper COCO evaluation API
    
    total_images = len(results)
    total_predictions = sum(len(r['predictions']) for r in results)
    
    print(f'Total images: {total_images}')
    print(f'Total predictions: {total_predictions}')
    print(f'Average predictions per image: {total_predictions / total_images:.2f}')
    
    # TODO: Implement proper mAP computation
    return {
        'total_images': total_images,
        'total_predictions': total_predictions,
        'avg_predictions_per_image': total_predictions / total_images,
    }


def main():
    args = parse_args()
    
    # Load config
    if args.config.endswith('.py'):
        cfg = Config.fromfile(args.config)
    else:
        cfg = Config(tcd_config.__dict__)
    
    # Set device
    torch.cuda.set_device(args.gpu_id)
    
    # Build model
    model = build_model(cfg, args.checkpoint)
    
    # Build dataset
    dataset = build_dataset(cfg, mode='test')
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    # Evaluate
    results = evaluate(model, dataloader, cfg)
    
    # Compute metrics if validation dataset
    if hasattr(cfg.data, 'val') and cfg.data.val.ann_file:
        metrics = compute_metrics(results, dataset, cfg)
        print(f'Metrics: {metrics}')
    
    # Save results
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Results saved to {args.out}')
    
    print('Evaluation completed!')


if __name__ == '__main__':
    main()

