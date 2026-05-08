_base_ = ['./basicvsr-pp_c64n7_8xb1-600k_reds4.py']

experiment_name = 'basicvsr-pp_c64n7_fc_finetune'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = work_dir

load_from = '/home/fc/Coding/CV/part2_1/checkpoints/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth'

model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRPlusPlusNet',
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        spynet_pretrained='/home/fc/Coding/CV/part2_1/checkpoints/spynet_20210409-c6c1bd09.pth',
    ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(fix_iter=5000),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ),
)

train_dataloader = dict(
    _delete_=True,
    num_workers=4,
    batch_size=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='fc_train', task_name='vsr'),
        data_root='/home/fc/Coding/CV/data/train',
        data_prefix=dict(
            img='train_sharp_bicubic/X4',
            gt='train_sharp',
        ),
        ann_file='/home/fc/Coding/CV/part2_1/data/meta_info_train.txt',
        depth=1,
        num_input_frames=30,
        pipeline=[
            dict(type='GenerateSegmentIndices', interval_list=[1]),
            dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
            dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
            dict(type='SetValues', dictionary=dict(scale=4)),
            dict(type='PairedRandomCrop', gt_patch_size=256),
            dict(type='Flip', keys=['img', 'gt'], direction='horizontal'),
            dict(type='Flip', keys=['img', 'gt'], direction='vertical'),
            dict(type='RandomTransposeHW', keys=['img', 'gt']),
            dict(type='PackInputs'),
        ],
    ),
)

val_dataloader = dict(
    _delete_=True,
    num_workers=2,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='fc_val', task_name='vsr'),
        data_root='/home/fc/Coding/CV/data/val',
        data_prefix=dict(
            img='val_sharp_bicubic/X4',
            gt='val_sharp',
        ),
        ann_file='/home/fc/Coding/CV/part2_1/data/meta_info_val.txt',
        depth=1,
        num_input_frames=30,
        pipeline=[
            dict(type='GenerateSegmentIndices', interval_list=[1]),
            dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
            dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
            dict(type='PackInputs'),
        ],
    ),
)

test_dataloader = dict(
    _delete_=True,
    num_workers=2,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='fc_val', task_name='vsr'),
        data_root='/home/fc/Coding/CV/data/val',
        data_prefix=dict(
            img='val_sharp_bicubic/X4',
            gt='val_sharp',
        ),
        ann_file='/home/fc/Coding/CV/part2_1/data/meta_info_val.txt',
        depth=1,
        num_input_frames=30,
        pipeline=[
            dict(type='GenerateSegmentIndices', interval_list=[1]),
            dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
            dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
            dict(type='PackInputs'),
        ],
    ),
)

val_evaluator = [
    dict(type='PSNR', crop_border=0, input_order='CHW'),
    dict(type='SSIM', crop_border=0, input_order='CHW'),
]
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=20000,
    val_interval=2000,
)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')

optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)),
    paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)}),
)

param_scheduler = dict(
    type='CosineRestartLR',
    by_epoch=False,
    periods=[20000],
    restart_weights=[1],
    eta_min=1e-7,
)

default_hooks = dict(
    _delete_=True,
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=1,   # 🔥 每个 iteration 打一次
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        by_epoch=False,
        save_best='PSNR',
        rule='greater',
        out_dir='./work_dirs/basicvsr-pp_c64n7_fc_finetune',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

vis_backends = [dict(type='LocalVisBackend')]

visualizer = dict(
    _delete_=True,
    type='Visualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')

resume = False