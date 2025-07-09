auto_scale_lr = dict(base_batch_size=64, enable=False)
checkpoint_config = dict(interval=10)
custom_hooks = [
    dict(num_last_epochs=15, priority=48, type='YOLOXModeSwitchHook'),
    dict(interval=10, num_last_epochs=15, priority=48, type='SyncNormHook'),
    dict(
        momentum=0.0001,
        priority=49,
        resume_from=None,
        type='ExpMomentumEMAHook'),
]
data = dict(
    persistent_workers=True,
    samples_per_gpu=8,
    test=dict(
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                img_scale=(
                    416,
                    416,
                ),
                transforms=[
                    dict(keep_ratio=True, type='Resize'),
                    dict(type='RandomFlip'),
                    dict(
                        pad_to_square=True,
                        pad_val=dict(img=(
                            114.0,
                            114.0,
                            114.0,
                        )),
                        type='Pad'),
                    dict(type='DefaultFormatBundle'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        type='CocoDataset'),
    train=dict(
        dataset=dict(
            ann_file='data/coco/annotations/instances_train2017.json',
            filter_empty_gt=False,
            img_prefix='data/coco/train2017/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            type='CocoDataset'),
        pipeline=[
            dict(img_scale=(
                640,
                640,
            ), pad_val=114.0, type='Mosaic'),
            dict(
                border=(
                    -320,
                    -320,
                ),
                scaling_ratio_range=(
                    0.5,
                    1.5,
                ),
                type='RandomAffine'),
            dict(type='YOLOXHSVRandomAug'),
            dict(flip_ratio=0.5, type='RandomFlip'),
            dict(img_scale=(
                640,
                640,
            ), keep_ratio=True, type='Resize'),
            dict(
                pad_to_square=True,
                pad_val=dict(img=(
                    114.0,
                    114.0,
                    114.0,
                )),
                type='Pad'),
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    1,
                    1,
                ),
                type='FilterAnnotations'),
            dict(type='DefaultFormatBundle'),
            dict(keys=[
                'img',
                'gt_bboxes',
                'gt_labels',
            ], type='Collect'),
        ],
        type='MultiImageMixDataset'),
    val=dict(
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                img_scale=(
                    416,
                    416,
                ),
                transforms=[
                    dict(keep_ratio=True, type='Resize'),
                    dict(type='RandomFlip'),
                    dict(
                        pad_to_square=True,
                        pad_val=dict(img=(
                            114.0,
                            114.0,
                            114.0,
                        )),
                        type='Pad'),
                    dict(type='DefaultFormatBundle'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        type='CocoDataset'),
    workers_per_gpu=4)
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
dist_params = dict(backend='nccl')
evaluation = dict(
    dynamic_intervals=[
        (
            285,
            1,
        ),
    ],
    interval=10,
    metric='bbox',
    save_best='auto')
img_scale = (
    640,
    640,
)
interval = 10
load_from = None
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
    ], interval=50)
log_level = 'INFO'
lr_config = dict(
    by_epoch=False,
    min_lr_ratio=0.05,
    num_last_epochs=15,
    policy='YOLOX',
    warmup='exp',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=1)
max_epochs = 300
model = dict(
    backbone=dict(deepen_factor=0.33, type='CSPDarknet', widen_factor=0.375),
    bbox_head=dict(
        feat_channels=96, in_channels=96, num_classes=80, type='YOLOXHead'),
    input_size=(
        640,
        640,
    ),
    neck=dict(
        in_channels=[
            96,
            192,
            384,
        ],
        num_csp_blocks=1,
        out_channels=96,
        type='YOLOXPAFPN'),
    random_size_interval=10,
    random_size_range=(
        10,
        20,
    ),
    test_cfg=dict(nms=dict(iou_threshold=0.65, type='nms'), score_thr=0.01),
    train_cfg=dict(assigner=dict(center_radius=2.5, type='SimOTAAssigner')),
    type='YOLOX')
mp_start_method = 'fork'
num_last_epochs = 15
opencv_num_threads = 0
optimizer = dict(
    lr=0.01,
    momentum=0.9,
    nesterov=True,
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0),
    type='SGD',
    weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
resume_from = None
runner = dict(max_epochs=300, type='EpochBasedRunner')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        flip=False,
        img_scale=(
            416,
            416,
        ),
        transforms=[
            dict(keep_ratio=True, type='Resize'),
            dict(type='RandomFlip'),
            dict(
                pad_to_square=True,
                pad_val=dict(img=(
                    114.0,
                    114.0,
                    114.0,
                )),
                type='Pad'),
            dict(type='DefaultFormatBundle'),
            dict(keys=[
                'img',
            ], type='Collect'),
        ],
        type='MultiScaleFlipAug'),
]
train_dataset = dict(
    dataset=dict(
        ann_file='data/coco/annotations/instances_train2017.json',
        filter_empty_gt=False,
        img_prefix='data/coco/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='CocoDataset'),
    pipeline=[
        dict(img_scale=(
            640,
            640,
        ), pad_val=114.0, type='Mosaic'),
        dict(
            border=(
                -320,
                -320,
            ),
            scaling_ratio_range=(
                0.5,
                1.5,
            ),
            type='RandomAffine'),
        dict(type='YOLOXHSVRandomAug'),
        dict(flip_ratio=0.5, type='RandomFlip'),
        dict(img_scale=(
            640,
            640,
        ), keep_ratio=True, type='Resize'),
        dict(
            pad_to_square=True,
            pad_val=dict(img=(
                114.0,
                114.0,
                114.0,
            )),
            type='Pad'),
        dict(
            keep_empty=False,
            min_gt_bbox_wh=(
                1,
                1,
            ),
            type='FilterAnnotations'),
        dict(type='DefaultFormatBundle'),
        dict(keys=[
            'img',
            'gt_bboxes',
            'gt_labels',
        ], type='Collect'),
    ],
    type='MultiImageMixDataset')
train_pipeline = [
    dict(img_scale=(
        640,
        640,
    ), pad_val=114.0, type='Mosaic'),
    dict(
        border=(
            -320,
            -320,
        ),
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='RandomAffine'),
    dict(type='YOLOXHSVRandomAug'),
    dict(flip_ratio=0.5, type='RandomFlip'),
    dict(img_scale=(
        640,
        640,
    ), keep_ratio=True, type='Resize'),
    dict(
        pad_to_square=True,
        pad_val=dict(img=(
            114.0,
            114.0,
            114.0,
        )),
        type='Pad'),
    dict(keep_empty=False, min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(type='DefaultFormatBundle'),
    dict(keys=[
        'img',
        'gt_bboxes',
        'gt_labels',
    ], type='Collect'),
]
workflow = [
    (
        'train',
        1,
    ),
]
