# model settings
model = dict(
    type='NASRetinaNetTrain',
    pretrained=dict(
        use_load=True,
        load_path='./seed_mbv2.pt',
        seed_num_layers=[1, 1, 2, 3, 4, 3, 3, 1, 1] # mbv2
        ),
    backbone=dict(
        type='FNA_Retinanet',
        net_config="""[[32, 16], ['k3_e1'], 1]|
[[16, 24], ['k3_e6', 'k3_e6'], 2]|
[[24, 32], ['k3_e6', 'k3_e6', 'k3_e6'], 2]|
[[32, 64], ['k3_e6', 'k3_e6', 'k3_e6', 'k3_e6'], 2]|
[[64, 96], ['k3_e6', 'k3_e6', 'k3_e6'], 1]|
[[96, 160], ['k3_e6', 'k3_e6', 'k3_e6'], 2]|
[[160, 320], ['k3_e6'], 1]""",
        output_indices=[2, 3, 5, 7]
        ),
    neck=None,
    bbox_head=dict(
        type='NewcrfsDecoder',
        dataset='nyu',
        max_depth=10.0,
        # fapn_att='FSM'
        ))
# training and testing settings
train_cfg = None
test_cfg = None
# dataset settings

# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.00005) # lr 0.05
optimizer = dict(type='Adam', lr=6e-4, weight_decay=0.0, betas=(0.9, 0.999))
# 梯度裁剪(max_norm表示梯度的最大范数，norm_type表示L2范数)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))  
# learning policy
lr_config = dict(
    policy='Cosine',
    target_lr=1e-6,
    by_epoch=False,  # 按步骤更新而不是按轮更新
    warmup='linear',  # 线性预热
    warmup_iters=500,  # 预热的迭代次数
    warmup_ratio=0.1  # 预热的初始学习率比例
)
checkpoint_config = dict(interval=-1)  # 废弃，已改为CustomDistEvalHook保存最佳架构
# yapf:disable
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 40
use_syncbn = True
image_size_madds = (480, 640)
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/yzh_test'
load_from = None
resume_from = None
workflow = [('train', 1)]
