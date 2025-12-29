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
        type='FPN',
        in_channels=[128, 256, 512, 1024],  # For Swin-B
        out_channels=256,
        num_outs=4,
        start_level=0,
        add_extra_convs='on_input',
    ),
    head=dict(
        type='Mask2FormerHead',
        in_channels=[256, 256, 256, 256],  # FPN output channels
        feat_channels=256,
        out_channels=256,
        num_things_classes=2,  # individual_tree, group_of_trees
        num_stuff_classes=0,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='PixelDecoder',
            in_channels=[256, 256, 256, 256],
            feat_channels=256,
            out_channels=256,
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
        ),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=6,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1,
                ),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'),
            ),
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_mask=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            loss_weight=5.0,
        ),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=False,
            loss_weight=5.0,
        ),
    ),
    train_cfg=dict(
        num_points=12544,
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
            multiscale_mode='value',
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
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'img_metas']),
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
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'img_metas']),
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

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

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
)

# Checkpoint configuration
checkpoint_config = dict(interval=5, save_last=True, max_keep_ckpts=3)

# Logging configuration
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
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
