_base_ = ['mmcls::resnet/resnet18_8xb16_cifar10.py']

resnet = _base_.model
pretrained_ckpt = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth'  # noqa: E501

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GeneralQuant',
    data_preprocessor=dict(
        type='mmcls.ClsDataPreprocessor',
        num_classes=10,
        # RGB format normalization parameters
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        # loaded images are already RGB format
        to_rgb=False),
    architecture=resnet,
    pretrained_ckpt=pretrained_ckpt,
    quantizer=dict(
        type='TensorRTQuantizer',
        skipped_methods=[
            'mmcls.models.heads.ClsHead._get_loss',
            'mmcls.models.heads.ClsHead._get_predictions'
        ],
        qconfig=dict(
            qtype='affine',
            w_observer=dict(type='mmrazor.LSQObserver'),
            a_observer=dict(type='mmrazor.LSQObserver'),
            w_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
            a_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
            w_qscheme=dict(
                bit=8,
                is_symmetry=False,
                is_per_channel=False,
                is_pot_scale=False,
            ),
            a_qscheme=dict(
                bit=8,
                is_symmetry=False,
                is_per_channel=False,
                is_pot_scale=False),
        )))

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = dict(
    _delete_=True,
    type='CosineAnnealingLR',
    T_max=100,
    by_epoch=True,
    begin=0,
    end=100)

model_wrapper_cfg = dict(
    type='mmrazor.GeneralQuantDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

# train, val, test setting
train_cfg = dict(
    _delete_=True,
    type='mmrazor.QATEpochBasedLoop',
    by_epoch=True,
    max_epochs=100,
    val_interval=1)
val_cfg = dict(_delete_=True, type='mmrazor.QATValLoop')
test_cfg = val_cfg
