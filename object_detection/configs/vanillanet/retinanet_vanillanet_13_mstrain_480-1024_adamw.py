#Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.


_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# you can download ckpt from:
# https://github.com/huawei-noah/VanillaNet/releases/download/ckpt/vanillanet_13_act_num_4_kd_pretrain.pth
checkpoint_file = '/your_path_to/vanillanet_13_act_num_4_kd_pretrain.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='Vanillanet',
        act_num=4,  # enlarge act_num for better downstream performance
        dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
        out_indices=[1, 8, 10],
        strides=[1,2,2,1,1,1,1,1,1,2,1],
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(in_channels=[256*4, 512*4, 1024*4], start_level=0, num_outs=5))

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 512), (1333, 544), (1333, 576), (1333, 608), (1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768), (1333, 800), (1333, 832), (1333, 864), (1333, 896), (1333, 928), (1333, 960), (1333, 992), (1333, 1024)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

optimizer = dict(
    _delete_=True,
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=8e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.6,
        'decay_type': 'layer_wise',
        'num_layers': 6
    })

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
