_base_ = [
    '../../_base_/datasets/mmcls/imagenet_bs64_swin_224.py',
    '../../_base_/schedules/mmcls/imagenet_bs1024_adamw_swin.py',
    '../../_base_/mmcls_runtime.py'
]

# student settings
student = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='DistilledVisionTransformer',
        arch='deit-base',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1),
    neck=None,
    head=dict(
        type='DeiTClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original', loss_weight=0.5),
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
    ]))

# teacher settings
teacher = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='TIMMBackbone',
        model_name='regnety_160',
        checkpoint_path='/mnt/lustre/caoweihan/checkpoint/cls/regnety_160-a5fe301d.pth'
        ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=3024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.),
        topk=(1, 5),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/mnt/lustre/caoweihan/checkpoint/cls/regnety_160-a5fe301d.pth',
            prefix='head.'),
    ))

# algorithm setting
algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMClsArchitecture',
        model=student,
    ),
    with_student_loss=True,
    with_teacher_loss=False,
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        teacher_norm_eval=True,
        components=[
            dict(
                student_module='head.layers.head_dist',
                teacher_module='head.fc',
                losses=[
                    dict(
                        type='CrossEntropyLoss',
                        name='loss_deit',
                        loss_weight=0.5)
                ])
        ]),
)

# data settings
data = dict(samples_per_gpu=64, workers_per_gpu=4)

paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    })
optimizer = dict(paramwise_cfg=paramwise_cfg)

custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

checkpoint_config = dict(interval=1, max_keep_ckpts=3, out_dir='s3://caoweihan/deit')
evaluation = dict(interval=1, metric='accuracy')

find_unused_parameters = True
