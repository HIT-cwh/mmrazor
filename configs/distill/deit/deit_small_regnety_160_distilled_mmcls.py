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
        arch='deit-small',
        img_size=224,
        patch_size=16,),
    neck=None,
    head=dict(
        type='DeiTClsHead',
        num_classes=1000,
        in_channels=384,
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
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original',
            loss_weight=0.5),
        topk=(1, 5),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/mnt/lustre/caoweihan/checkpoint/cls/regnety_160-a5fe301d.pth',
            prefix='head.'),
    ),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
    ]))

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
        type='SingleTeacherDistillerV2',
        teacher=teacher,
        teacher_trainable=False,
        teacher_norm_eval=True,
        student_recorders=[
            dict(
                type='ModuleOutputs',
                sources=['head.layers.head_dist'])
        ],
        teacher_recorders=[
            dict(
                type='ModuleOutputs',
                sources=['head.fc'])
        ],
        components=[
            dict(
                student_items=[
                    dict(record_type='ModuleOutputs',
                         source='head.layers.head_dist')
                ],
                teacher_items=[
                    dict(record_type='ModuleOutputs', source='head.fc')
                ],
                loss=dict(
                    type='CrossEntropyLoss',
                    loss_weight=0.5
                )),
        ],
        distill_deliveries=(
            dict(type='MethodOutputs', max_keep_data=1, source='teacher',
                 target='student', method='Augments.__call__',
                 import_module='mmcls.models.utils'),)
    ),
)

# data settings
data = dict(samples_per_gpu=256, workers_per_gpu=4)

paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    })
optimizer = dict(paramwise_cfg=paramwise_cfg)

checkpoint_config = dict(interval=1, max_keep_ckpts=3, out_dir='s3://caoweihan/deit')
evaluation = dict(interval=1, metric='accuracy')

find_unused_parameters = True
