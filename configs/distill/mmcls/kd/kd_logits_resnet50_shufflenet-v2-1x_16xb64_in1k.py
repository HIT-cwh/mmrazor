_base_ = ['mmcls::shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py']

student = _base_.model

teacher_ckpt = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    _delete_=True,
    type='SingleTeacherDistill',
    data_preprocessor=dict(
        type='ImgDataPreprocessor',
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        bgr_to_rgb=True),
    architecture=student,
    teacher=dict(
        cfg_path='mmcls::resnet/resnet50_8xb32_in1k.py', pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        teacher_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        distill_losses=dict(
            loss_kl=dict(type='KLDivergence', tau=1, loss_weight=5)),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
