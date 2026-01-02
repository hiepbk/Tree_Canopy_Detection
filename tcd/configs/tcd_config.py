"""
TCD Configuration for Tree Canopy Detection
Contains training, testing, data loader, and model configurations.
"""

# Model configuration
model = dict(
    type='Mask2Former',
    backbone=dict(
        type='SwinBackbone',
        num_channels=3,
        arch='swinb',  # 'swinb' or 'swint'
        pretrained='pretrained/aerial_swinb_si.pth',
        frozen_stages=2,
    ),
    neck=dict(
        type='YOSONeck',
        in_channels=[128, 256, 512, 1024],  # For Swin-B: [stage2, stage3, stage4, stage5]
        # For Swin-T use: [96, 192, 384, 768]
        agg_dim=128,  # Aggregated feature dimension
        hidden_dim=256,  # Hidden dimension for output
        norm='BN',  # 'BN' for single GPU, 'SyncBN' for multi-GPU
    ),
    head=dict(
        type='YOSOHead',
        in_channels=256,  # YOSO neck outputs single feature map with hidden_dim channels
        hidden_dim=256,
        num_things_classes=2,  # individual_tree, group_of_trees
        num_proposals=100,  # Number of proposal kernels (equivalent to num_queries)
        num_stages=3,  # Number of refinement stages
        conv_kernel_size_2d=1,  # 2D convolution kernel size for proposal kernels (must be 1 to match original YOSO)
        conv_kernel_size_1d=3,  # 1D convolution kernel size for attention
        temperature=0.1,  # Temperature for logits scaling
        num_cls_fcs=2,  # Number of classification FC layers
        num_mask_fcs=2,  # Number of mask FC layers
        feedforward_channels=2048,  # FFN hidden dimension
        loss_cls=dict(
            type='FocalLoss',
            alpha=0.25,
            gamma=2.0,
        ),
        loss_mask=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=False,
        ),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=False,
        ),
        num_points=6000,  # Number of points for mask loss computation
    ),
    train_cfg=dict(
        # num_points=12544,  # Original - causes OOM
        # num_points=3136,  # Too low - poor matching quality
        num_points=6000,  # Balanced: ~2x of 3136, better matching while avoiding OOM
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='HungarianMatcher',
            cost_class=2.0,
            cost_mask=5.0,
            cost_dice=5.0,
        ),
    ),
    test_cfg=dict(
        panoptic_on=False,
        semantic_on=False,
        instance_on=True,
        max_per_image=100,
        filter_low_score=False,
        object_mask_thr=0.0,
    ),
)

# Dataset configuration
data_root = 'data/Tree_Canopy_Data/'

# Training dataset
train_dataset = dict(
    type='TreeCanopyDataset',
    ann_file=data_root + 'annotations/instances_train.json',
    img_prefix=data_root + 'raw_data/trainval_images/',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(
            type='Resize',
            img_scale=[(512, 512), (768, 768), (1024, 1024), (1280, 1280)],
            multiscale_mode='value',  # Changed from 'value' to 'range' for random multi-scale training
            keep_ratio=True,
        ),
        dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
        dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
        dict(
            type='ColorJitter',
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        ),
        dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        dict(type='Pad', size_divisor=32),
        dict(type='Polygon2Mask'),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'img_metas', 'ori_gt_masks', 'ori_img']),
    ],
    filter_empty_gt=True,
)

# Validation dataset
val_dataset = dict(
    type='TreeCanopyDataset',
    ann_file=data_root + 'annotations/instances_val.json',
    img_prefix=data_root + 'raw_data/trainval_images/',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(
            type='Resize',
            img_scale=(1024, 1024),
            keep_ratio=True,
        ),
        dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        dict(type='Pad', size_divisor=32),
        dict(type='Polygon2Mask'),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'img_metas', 'ori_gt_masks', 'ori_img']),
    ],
    test_mode=False,
    filter_empty_gt=False,
)

# Test dataset
test_dataset = dict(
    type='TreeCanopyDataset',
    ann_file=None,  # Will be set during testing
    img_prefix=None,  # Will be set during testing
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='Resize',
            img_scale=(1024, 1024),
            keep_ratio=True,
        ),
        dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        dict(type='Pad', size_divisor=32),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img']),
    ],
    test_mode=True,
)

# Data loader configuration
data = dict(
    samples_per_gpu=2,  # Batch size per GPU
    workers_per_gpu=4,  # Number of data loading workers per GPU
    pin_memory=False,  # Disable pin_memory to save VRAM
    train=train_dataset,
    val=val_dataset,
    test=test_dataset,
)

# Optimizer configuration
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
        },
    ),
)

optimizer_config = dict(
    grad_clip=dict(max_norm=0.1, norm_type=2),
    # No mixed precision - use FP32 like original YOSO
)

# Learning rate schedule
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr=0.00001,
)

# Training configuration
runner = dict(
    type='EpochBasedRunner',
    max_epochs=50,
    # Gradient accumulation - process multiple batches before updating
    # Effective batch size = samples_per_gpu * gradient_accumulation_steps
    gradient_accumulation_steps=1,  # Increase to 2-4 if still OOM
)

# Checkpoint configuration
checkpoint_config = dict(interval=5, save_last=True, max_keep_ckpts=3)

# Logging configuration
log_config = dict(
    interval=2,
    hooks=[
        'TextLoggerHook',
        'WandbHook',
    ],
    wandb=dict(
        init_kwargs=dict(
            project='tree_canopy_detection',
            name='tcd_exp1',
        ),
        num_eval_images=5,
    ),
)

# Evaluation configuration
evaluation = dict(
    interval=5,
    metric=['segm'],
    save_best='segm_mAP',
    rule='greater',
)

# Workflow
workflow = [('train', 1), ('val', 1)]

# Seed
seed = 42
deterministic = True

# Device
gpu_ids = [0]
