# model settings
type = 'Retinanet'
model = dict(
    type='NASRetinaNet',
    pretrained=dict(
        use_load=True,
        load_path='./seed_mbv2.pt',        
        seed_num_layers=[1, 1, 2, 3, 4, 3, 3, 1, 1] # mbv2
        ),
    backbone=dict(
        type='RetinaNetBackbone',
        search_params=dict(
            sample_policy='prob', # prob uniform
            weight_sample_num=1,
            affine=False,
            track=False,
            net_scale=dict(
                chs = [32, 16, 24, 32, 64, 96, 160, 320],
                num_layers = [4, 4, 6, 6, 4, 1],
                strides = [2, 1, 2, 2, 2, 1, 2, 1, 1],
            ),
            primitives_normal=['k3_e3',
                                'k3_e6',
                                'k5_e3',
                                'k5_e6',
                                'k7_e3',
                                'k7_e6',
                                'skip',],
            primitives_reduce=['k3_e3',
                                'k3_e6',
                                'k5_e3',
                                'k5_e6',
                                'k7_e3',
                                'k7_e6',],
        ),
        output_indices=[2, 3, 5, 7],

    ),
    neck=None,
    bbox_head=dict(
        type='NewcrfsDecoder',
        dataset='colon',
        ))

# training and testing settings
train_cfg = None
test_cfg = None

# optimizer
optimizer = dict(
    weight_optim = dict(
        optimizer = dict(type='Adam', lr=2e-4, weight_decay=2e-4, betas=(0.9, 0.999)),
        optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
    ),
    arch_optim = dict(
        optimizer = dict(type='Adam', lr=0),
        optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
    )
)
optimizer_search = dict(
    weight_optim = dict(
        optimizer = dict(type='Adam', lr=1e-3, weight_decay=4e-5, betas=(0.9, 0.999)),
        optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
    ),
    arch_optim = dict(
        optimizer = dict(type='Adam', lr=0),
        optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
    )
)
# learning policy
lr_config = dict(
    policy='Cosine',
    target_lr=1e-5,
    by_epoch=False,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 10
)
lr_config_search = dict(
    policy='Cosine',
    target_lr=1e-5,
    by_epoch=False,
)
total_epochs = 100

checkpoint_config = dict(interval=100, filename_tmpl='supernet.pth', save_optimizer=False)  # interval must equal total_epochs!
# yapf:disable
log_config = dict(
    interval=20, 
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# configs for sub_obj optimizing
sub_obj=dict(
    if_sub_obj=True,
    type='flops',
    log_base=10.,
    sub_loss_factor=0.02
)
# yapf:enable
# runtime settings

single_epochs = 5
n_iter = 30
population_size = 40

use_syncbn = False
min_op=True

arch_update_epoch = 999
alter_type = 'step' # step / epoch
train_data_ratio = 0.5
image_size_madds = (352, 1120)
model_info_interval = 1000
device_ids = range(8)
dist_params = dict(backend='nccl')
# log_level = 'DEBUG'
log_level = 'INFO'
work_dir = './work_dirs/'
load_from = None
resume_from = None
workflow = [('train', 1)]
