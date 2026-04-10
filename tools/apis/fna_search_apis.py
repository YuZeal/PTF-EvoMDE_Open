from __future__ import division

from collections import OrderedDict

import torch
import torch.distributed as dist

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook
from mmdet.apis.env import get_root_logger
from mmdet.core import (CocoDistEvalRecallHook, CocoDistEvalmAPHook,
                        DistEvalmAPHook, DistOptimizerHook)
from mmdet.datasets import build_dataloader

from models.dropped_model import Dropped_Network
from tools.apis.fna_search_runner import NASRunner
from tools.hooks.optimizer import ArchDistOptimizerHook
from tools.apis.evalHook import CustomDistEvalHook
from tools.apis.evalHook_seg import CustomDistEvalHook_seg
from tools.apis.evalHook_class import CustomDistEvalHook_class


from .code2net import decode_arch, uniform_sample_code

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


def batch_processor(model, data, train_mode=True, mode='train', search_stage=0, net_type='', alpha_index=None):
    if train_mode:
        if mode == 'train':
            sample_num = 1
        elif mode == 'arch':
            sample_num = -1
    else:
        sample_num = -1

    DroppedBackBone = Dropped_Network


    # if sample_num is not None:
    if alpha_index is None:
        # _ = model.module.backbone.sample_branch(sample_num, search_stage=search_stage)  # 采样更新self.alpha_index
        alpha_index = decode_arch(uniform_sample_code(num=1)[0])

    # 更新self.alpha_index
    if hasattr(model, 'module'):
        model.module.backbone.alpha_index = alpha_index  
    else:
        model.backbone.alpha_index = alpha_index
      # 更换backbone
    if hasattr(model, 'module'):
        model.module.backbone = DroppedBackBone(model.module.backbone)  # model.module.backbone的alpha_index会被DroppedBackBone初始化时继承，然后搭建架构
    else:
        model.backbone = DroppedBackBone(model.backbone)
    
    
    losses, sub_obj = model(**data)
    ### deubg
    # Total_params = 0
    # Trainable_params = 0
    # NonTrainable_params = 0
    # import numpy as np
    # for param in model.module.backbone.parameters():
    #     mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
    #     Total_params += mulValue  # 总参数量
    #     if param.requires_grad:
    #         Trainable_params += mulValue  # 可训练参数量
    #     else:
    #         NonTrainable_params += mulValue  # 非可训练参数量
    # print('Total params: {}M'.format(Total_params / 1e6))
    # print('Trainable params: {}M'.format(Trainable_params/ 1e6))
    # print('Non-trainable params: {}M'.format(NonTrainable_params/ 1e6))
    # assert False

    # Total params: 12.527257M
    # Trainable params: 1.915416M
    # Non-trainable params: 10.611841M

    sub_obj = torch.mean(sub_obj)
    loss, log_vars = parse_losses(losses)
    log_vars['sub_obj'] = sub_obj.item()
    outputs = dict(
        loss=loss, sub_obj=sub_obj, log_vars=log_vars, num_samples=len(data['image'].data))

    return outputs


def search_detector(model,
                   datasets,
                   eval_dataset,
                   cfg,
                   args,
                   distributed=False,
                   validate=False,
                   logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, datasets, eval_dataset, cfg, args, validate=validate, logger=logger)
    else:
        _non_dist_train(model, datasets, cfg, args, validate=validate, logger=logger)


def _dist_train(model, datasets, eval_dataset, cfg, args, validate=False, logger=None):
    # prepare data loaders
    data_loaders = [datasets.train_data, datasets.arch_data]
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())
    # build runner
    runner = NASRunner(model, batch_processor, None, cfg.work_dir, cfg.log_level, cfg=cfg, logger=logger, data_loaders=data_loaders, args=args)

    # register hooks
    weight_optim_config = DistOptimizerHook(**cfg.optimizer.weight_optim.optimizer_config)
    arch_optim_config = ArchDistOptimizerHook(**cfg.optimizer.arch_optim.optimizer_config)
    runner.register_training_hooks(cfg.lr_config, weight_optim_config, arch_optim_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        dataloader_eval = eval_dataset.data
        if args.dataset in ['kitti','nyu', 'smalldata', 'blender', 'colon']:
            custom_eval_hook = CustomDistEvalHook(dataloader_eval, args, logger, interval=-1, isSearch=False)
        elif args.dataset == 'cityscapes':
            custom_eval_hook = CustomDistEvalHook_seg(dataloader_eval, args, logger, interval=-1, isSearch=False) 
        elif args.dataset == 'imagenet':
            custom_eval_hook = CustomDistEvalHook_class(dataloader_eval, args, logger, interval=10, isSearch=True) 
        runner.register_hook(custom_eval_hook)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    if args.load_supernet_path == None:  # 不加载超网络权重，则从头训练超网
        runner.run(cfg.workflow, cfg.total_epochs, cfg.arch_update_epoch)

    ### Add EC
    if cfg.n_iter > 0:
        if args.DetNAS:
            runner.DetNAS(custom_eval_hook, n_iter=cfg.n_iter ,population_size=cfg.population_size)
            # runner.DetNAS(custom_eval_hook)
        else:
            runner.EC(cfg.workflow, custom_eval_hook, cfg.single_epochs, cfg.n_iter, cfg.population_size, cfg.min_op, args.crossover_type)
    


def _non_dist_train(model, datasets, cfg, args, validate=False, logger=None):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False) for dataset in datasets
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    # build runner
    runner = NASRunner(model, batch_processor, None, cfg.work_dir, cfg.log_level, cfg=cfg, logger=logger)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer.weight_optim.optimizer_config,
                                    cfg.optimizer.arch_optim.optimizer_config, 
                                    cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs, cfg.arch_update_epoch)

