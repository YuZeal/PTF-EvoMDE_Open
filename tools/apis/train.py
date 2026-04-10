from __future__ import division

from collections import OrderedDict

import torch
import mmcv
from mmcv.parallel import scatter, collate

import time

import torch.distributed as dist
from mmcv.runner import Runner, DistSamplerSeedHook
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.core import (DistOptimizerHook, DistEvalHook)
from mmdet.datasets import build_dataloader
from mmdet.apis.env import get_root_logger
from tools.apis.evalHook import CustomDistEvalHook
from tools.apis.evalHook_seg import CustomDistEvalHook_seg

def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
    
    # dist.barrier()
    # dist.all_reduce(loss)
    # loss /= dist.get_world_size()

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['image'].data))
    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   args,
                   distributed=False,
                   validate=False,
                   logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, dataset, cfg, args, validate=validate, logger=logger)
    else:
        _non_dist_train(model, dataset, cfg, args, validate=validate, logger=logger)


def _dist_train(model, dataset, cfg, args, validate=False, logger=None):
    # prepare data loaders
    data_loaders = [dataset[0].data]  # 统一格式：列表

    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())

    # build runner 
    # Runner 是训练和验证的核心对象，负责管理整个训练过程。
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    
    if getattr(cfg, 'param_groups', None):  # head和backbone使用不同的学习率
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:  # bbox_head
                head_params.append(param)

        # 构建 param_groups，手动设置学习率倍率
        param_groups = [
            {'params': backbone_params, 'lr': cfg.optimizer['lr'] * cfg.param_groups['lr_mult']['backbone']},
            {'params': head_params, 'lr': cfg.optimizer['lr'] * cfg.param_groups['lr_mult']['head']}
        ]

        # 使用手动定义的 param_groups 构建优化器
        optimizer = torch.optim.AdamW(param_groups, lr=cfg.optimizer['lr'], betas=cfg.optimizer['betas'], weight_decay=cfg.optimizer['weight_decay'])
        runner.optimizer = optimizer
        logger.info(f"==> backbone_params lr is {cfg.optimizer['lr'] * cfg.param_groups['lr_mult']['backbone']:.1e}")
        logger.info(f"==> head_params lr is {cfg.optimizer['lr'] * cfg.param_groups['lr_mult']['head']:.1e}")




    # register hooks
    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)  # 用于分布式训练中的优化器配置
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())  # 用于在分布式训练中设置采样器的种子
    # register eval hooks
    if validate:
        dataloader_eval = dataset[1].data
        if args.dataset in ['kitti', 'nyu', 'smalldata', 'blender', 'colon']:
            custom_eval_hook = CustomDistEvalHook(dataloader_eval, args, logger, cfg.work_dir, interval=1)
        elif args.dataset == 'cityscapes':
            custom_eval_hook = CustomDistEvalHook_seg(dataloader_eval, args, logger, cfg.work_dir, interval=1)
        else:
            assert False,'args.dataset is invalid name'
        runner.register_hook(custom_eval_hook)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)  # 调用 runner 的 run 方法，开始训练和验证过程


def _non_dist_train(model, dataset, cfg, validate=False, logger=None):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False)
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
