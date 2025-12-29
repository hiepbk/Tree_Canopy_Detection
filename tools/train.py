"""
Training script for Tree Canopy Detection.
"""

import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml
import numpy as np
import cv2

from tcd.models import MODEL
from tcd.datasets import TreeCanopyDataset, collate_fn
from tcd.utils import Config
from tcd.configs import tcd_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train Tree Canopy Detection model')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--work-dir', help='Working directory', default='work_dirs')
    parser.add_argument('--resume-from', help='Resume from checkpoint')
    parser.add_argument('--load-from', help='Load pretrained model')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic algorithms')
    return parser.parse_args()


def set_random_seed(seed, deterministic=False):
    """Set random seed."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(cfg):
    """Build model from config (cfg is already a dict)."""
    from tcd.models import MODEL
    from tcd.utils import build_from_cfg
    
    # Config is already a dict, just get model config
    model_cfg = cfg['model']
    
    # Build model using registry
    model = build_from_cfg(model_cfg, MODEL)
    return model


def build_dataset(cfg, mode='train'):
    """Build dataset from config (cfg is already a dict)."""
    if mode == 'train':
        dataset_cfg = cfg['data']['train']
    elif mode == 'val':
        dataset_cfg = cfg['data']['val']
    else:
        raise ValueError(f'Unknown mode: {mode}')
    
    dataset = TreeCanopyDataset(
        ann_file=dataset_cfg['ann_file'],
        img_prefix=dataset_cfg['img_prefix'],
        pipeline=dataset_cfg['pipeline'],
        test_mode=(mode != 'train'),
        filter_empty_gt=dataset_cfg.get('filter_empty_gt', True),
    )
    return dataset


def build_dataloader(dataset, cfg, mode='train', distributed=False):
    """Build dataloader (cfg is already a dict)."""
    if mode == 'train':
        samples_per_gpu = cfg['data']['samples_per_gpu']
        workers_per_gpu = cfg['data']['workers_per_gpu']
    else:
        samples_per_gpu = 1
        workers_per_gpu = 2
    
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=(mode == 'train'),
        )
        shuffle = False
    else:
        sampler = None
        shuffle = (mode == 'train')
    
    dataloader = DataLoader(
        dataset,
        batch_size=samples_per_gpu,
        shuffle=shuffle,
        collate_fn=collate_fn,
        sampler=sampler,
        num_workers=workers_per_gpu,
        pin_memory=True,
        drop_last=(mode == 'train'),
    )
    return dataloader


def build_optimizer(model, cfg):
    """Build optimizer (cfg is already a dict)."""
    optimizer_cfg = cfg['optimizer'].copy()
    optimizer_type = optimizer_cfg.pop('type')
    
    # Get parameters
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Apply custom learning rate multipliers
        lr_mult = 1.0
        if 'paramwise_cfg' in optimizer_cfg:
            custom_keys = optimizer_cfg['paramwise_cfg'].get('custom_keys', {})
            for key, value in custom_keys.items():
                if key in name:
                    lr_mult = value.get('lr_mult', 1.0)
                    break
        
        params.append({
            'params': [param],
            'lr': optimizer_cfg['lr'] * lr_mult,
        })
    
    if optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(
            params,
            weight_decay=optimizer_cfg.get('weight_decay', 0.0),
        )
    else:
        raise ValueError(f'Unknown optimizer type: {optimizer_type}')
    
    return optimizer


def train_epoch(model, dataloader, optimizer, epoch, cfg, distributed=False):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, data in enumerate(dataloader):
        # Data is already in correct format from collate_fn:
        # img: [B, C, H, W] tensor
        # gt_bboxes: List[Tensor] - one per image
        # gt_labels: List[Tensor] - one per image
        # gt_masks: List[Tensor] - one per image, each [num_instances, H, W]
        # img_metas: List[Dict] - one per image
        
        # Move to GPU
        img = data['img'].cuda()
        gt_bboxes = [bbox.cuda() for bbox in data['gt_bboxes']]
        gt_labels = [label.cuda() for label in data['gt_labels']]
        gt_masks = [mask.cuda() for mask in data['gt_masks']]
        img_metas = data['img_metas']
        
        # Forward
        losses = model(
            img=img,
            img_metas=img_metas,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            gt_masks=gt_masks,
        )
        
        # Compute total loss
        if isinstance(losses, dict):
            loss = sum(losses.values())
            # Print individual losses
            if batch_idx % 10 == 0:
                loss_str = ', '.join([f'{k}: {v.item():.4f}' for k, v in losses.items()])
                print(f'Epoch {epoch}, Batch {batch_idx}, Total Loss: {loss.item():.4f} ({loss_str})')
        else:
            loss = losses
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if 'optimizer_config' in cfg and 'grad_clip' in cfg['optimizer_config']:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                **cfg['optimizer_config']['grad_clip'],
            )
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    args = parse_args()
    
    # Load config (returns plain dict)
    if args.config.endswith('.py'):
        cfg = Config.fromfile(args.config, return_dict=True)
    else:
        # Use default config (convert to dict)
        cfg = Config(tcd_config.__dict__).to_dict()
    
    # Set random seed
    set_random_seed(cfg.get('seed', 42), cfg.get('deterministic', False))
    
    # Setup distributed training
    distributed = args.gpus > 1
    if distributed:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
    else:
        rank = 0
    
    # Create work directory
    work_dir = args.work_dir
    os.makedirs(work_dir, exist_ok=True)
    
    # Build model
    model = build_model(cfg)
    model = model.cuda()
    
    if distributed:
        model = DDP(model, device_ids=[rank])
    
    # Build datasets
    train_dataset = build_dataset(cfg, mode='train')
    val_dataset = build_dataset(cfg, mode='val')
    
    # Build dataloaders
    train_loader = build_dataloader(train_dataset, cfg, mode='train', distributed=distributed)
    val_loader = build_dataloader(val_dataset, cfg, mode='val', distributed=distributed)
    
    # Build optimizer
    optimizer = build_optimizer(model.module if distributed else model, cfg)
    
    # Training loop
    max_epochs = cfg['runner']['max_epochs']
    for epoch in range(max_epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(
            model.module if distributed else model,
            train_loader,
            optimizer,
            epoch,
            cfg,
            distributed=distributed,
        )
        
        if rank == 0:
            print(f'Epoch {epoch} completed. Average loss: {train_loss:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % cfg['checkpoint_config']['interval'] == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model': model.module.state_dict() if distributed else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                checkpoint_path = os.path.join(work_dir, f'epoch_{epoch+1}.pth')
                torch.save(checkpoint, checkpoint_path)
                print(f'Checkpoint saved to {checkpoint_path}')
    
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

