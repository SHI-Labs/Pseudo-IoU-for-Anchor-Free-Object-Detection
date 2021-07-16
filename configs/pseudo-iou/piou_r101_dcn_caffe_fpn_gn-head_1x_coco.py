_base_ = './piou_r101_caffe_fpn_gn-head_1x_coco.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=4, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
# optimizer
optimizer = dict(
    lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
