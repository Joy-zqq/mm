'''
Description: Knowledge Distillation Single Stage Detector
Autor: Joy
Email: JoyZheng@human-horizons.com
LastEditors: Joy
Date: 2022-09-22 11:09:30
LastEditTime: 2022-10-14 12:06:32
'''

# python tools/train.py configs/_distiller/ld_r18_gflv1_r101_fpn_coco_1x.py  --work-dir 1_work_dir/ --gpu-id 7

_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# 修改数据集配置
dataset_type = 'CocoDataset'
data_root = '0_data/'
classes = ('traffic light', 'traffic sign', 'car', 'pedestrian', 
        'bus', 'truck', 'rider', 'bicycle', 'motorcycle', 
        'train', 'other vehicle', 'other person', 'trailer')

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1280, 720), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1280, 720),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]

data = dict(
    train=dict(
        img_prefix=data_root+'BDD100K/train_images/',
        classes=classes,
        # pipeline=train_pipeline,
        ann_file=data_root+'BDD100K/annotations/bdd100k_2d_train.json'),
        # ann_file=data_root+'OwnBDD/bdd100k_2d_train_own.json'),
    val=dict(
        img_prefix=data_root+'BDD100K/valid_images/',
        classes=classes,
        # pipeline=test_pipeline,
        ann_file=data_root+'BDD100K/annotations/bdd100k_2d_valid.json'),
        # ann_file=data_root+'OwnBDD/bdd100k_2d_valid_own.json'),
    test=dict(
        img_prefix=data_root+'BDD100K/valid_images/',
        classes=classes,
        # pipeline=test_pipeline,
        ann_file=data_root+'BDD100K/annotations/bdd100k_2d_valid.json'),
        # ann_file=data_root+'OwnBDD/bdd100k_2d_valid_own.json'),
)

# 修改模型架构配置
teacher_ckpt = '0_checkpoints/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'
model = dict(
    type='KnowledgeDistillationSingleStageDetector',
    teacher_config='configs/gfl/gfl_r101_fpn_mstrain_2x_coco.py',
    teacher_ckpt=teacher_ckpt,
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        # 特征图金字塔网络
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        # Localization Distillation
        type='LDHead',
        num_classes=len(classes),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        loss_ld=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# 修改训练计划配置
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# lr_config = dict(  # 学习率调整配置，用于注册 LrUpdater hook。
#     policy='step',  # 调度流程(scheduler)的策略，也支持 CosineAnnealing, Cyclic, 等。请从 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9 参考 LrUpdater 的细节。
#     warmup='exp',  # 预热(warmup)策略，也支持 `exp` 和 `constant`。
#     warmup_iters=200,  # 预热的迭代次数
#     warmup_ratio= 0.001,  # 用于热身的起始学习率的比率
#     step=[8, 11])  # 衰减学习率的起止回合数
runner = dict(max_epochs=8)         # 最大epoch数

# 修改运行信息配置
# evaluation = dict(interval=1, metric='mAP')

work_dir = '2_work_dir'