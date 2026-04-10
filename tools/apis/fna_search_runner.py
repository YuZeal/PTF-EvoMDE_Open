import copy
import os
import os.path as osp
import time

import torch
import torch.distributed as dist

import mmcv
from mmcv.runner.checkpoint import load_checkpoint
from mmcv.runner import Runner, hooks
from mmcv.runner.hooks import (CheckpointHook, Hook, IterTimerHook,
                               LrUpdaterHook, OptimizerHook, lr_updater)
from mmcv.runner.priority import get_priority
from mmcv.runner.utils import (get_dist_info, get_host_info, get_time_str,
                               obj_from_dict)
from models.derive_arch import ArchGenerate_FNA
from models.derived_retinanet_backbone import FNA_Retinanet
from tools.hooks.fna_search_hooks import (DropProcessHook, ModelInfoHook,
                                          NASTextLoggerHook)
from tools.hooks.optimizer import ArchOptimizerHook
from .code2net import encode_arch, decode_arch, uniform_sample_code, evolution, sample_candidate, balanced_flops_friendly_weights
from calflops import calculate_flops

from models.dropped_model import Dropped_Network
import numpy as np
choice=lambda x:x[np.random.randint(len(x))] if isinstance(x,tuple) else choice(tuple(x))

class NASRunner(Runner):
    def __init__(self, *args, **kwargs):
        assert 'cfg' in kwargs.keys()
        self.cfg = kwargs.pop('cfg')
        self.data_loaders = kwargs.pop('data_loaders')
        self.args = kwargs.pop('args')
        super(NASRunner, self).__init__(*args, **kwargs)

        self.supernet_weights = None  # 初始化为None，用于缓存权重
        self.sub_obj_cfg = self.cfg.sub_obj
        self.type = self.cfg.type
        super_backbone = self.model.module.backbone if hasattr(self.model, 'module') \
                                                        else self.model.backbone
        self.arch_gener = ArchGenerate_FNA(super_backbone)
        self.der_Net = lambda net_config: FNA_Retinanet(net_config)


        nas_optimizers = self.cfg.optimizer
        self.optimizer, self.arch_optimizer = self.init_nas_optimizer(nas_optimizers)
        self._arch_hooks = []

    
    def run(self, workflow, max_epochs, arch_update_epoch, **kwargs):
        if self.cfg.alter_type=='epoch':  # only for search
            self.run_epoch_alter(workflow, max_epochs, arch_update_epoch, **kwargs)
        elif self.cfg.alter_type=='step':  # only for supernet
            self.run_step_alter(workflow, max_epochs, arch_update_epoch, **kwargs)
        else:
            raise TypeError('The alternation type of optimization must be epoch or step')
        
    def reset_search_optim(self):
        nas_optimizers = self.cfg.optimizer_search
        self.optimizer, self.arch_optimizer = self.init_nas_optimizer(nas_optimizers)

    # 定义内存映射加载函数
    def load_weights_mmap(self):
        if self.supernet_weights is None:  # 仅在第一次调用时加载权重
            filepath = self.cfg.work_dir + '/supernet.pth' if self.args.load_supernet_path == None else self.args.load_supernet_path
            self.supernet_weights = torch.load(filepath, map_location='cpu')['state_dict']
        return self.supernet_weights
    
    def predictor(self, workflow, custom_eval_hook, single_epochs, alpha_index, **kwargs):  # 训练+验证
        # 继承超网
        supernet_weights = self.load_weights_mmap()
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(supernet_weights, strict=True)
        else:
            self.model.load_state_dict(supernet_weights, strict=True)
        # missing_keys, unexpected_keys  = self.model.module.load_state_dict(supernet_weights, strict=False)
        # if missing_keys:
        #     print(f"Missing keys: {missing_keys}")
        # if unexpected_keys:
        #     print(f"Unexpected keys: {unexpected_keys}")
        # assert False,'debug'
        self.logger.info('finish loading supernet weight')
        # 训练single_epochs
        self.run_epoch_alter(workflow, max_epochs=single_epochs, arch_update_epoch=999, alpha_index=alpha_index, **kwargs)
        # 验证
        eval_measures = custom_eval_hook.after_train_epoch(self, force=True, alpha_index=alpha_index) 
        # 计算FLOPs
        if self.args.dataset == 'kitti':
            input_shape = (1, 3, 352, 1216)
        elif self.args.dataset == 'nyu':
            input_shape = (1, 3, 480, 640)
        else:
            assert False,'self.args.dataset is not in [kitti, nyu]'
        flops, macs, params = calculate_flops(model=self.model, 
                                            input_shape=input_shape,
                                            print_results=False,
                                            print_detailed=False,
                                            output_as_string=False,
                                            output_precision=2)
        
        return eval_measures, flops / 1e9

    def EC(self, workflow, custom_eval_hook, single_epochs=3, n_iter=30, population_size=40, min_op=True, crossover_type='int_one_point',**kwargs):

        self.logger.info('Supernet training completed. Starting evolutionary computation!')
        # reset optimizer
        self.reset_search_optim()
        self.logger.info('Finish reset optimizer!')
        # reset lr_hook
        hook_name = self.cfg.lr_config['policy'].title() + 'LrUpdaterHook'
        if not hasattr(lr_updater, hook_name):
            raise ValueError('"{}" does not exist'.format(hook_name))
        hook_cls = getattr(lr_updater, hook_name)
        self.unregister_hook(hook_cls)  # 删除旧的register_lr_hooks
        self.register_lr_hooks(self.cfg.lr_config_search)  # 注册新的register_lr_hooks
        self.logger.info('Finish reset lr_hooks!')
        
        # 使用包装函数传递实例方法
        def wrapper(workflow, custom_eval_hook, single_epochs, alpha_index, **kwargs):
            return self.predictor(workflow, custom_eval_hook, single_epochs, alpha_index, **kwargs)
        
        optimal_solutions, optimal_objective_values = evolution(crossover_type, n_iter, population_size, self.logger, wrapper, workflow, custom_eval_hook, single_epochs, min_op=min_op, **kwargs)
        
        self.logger.info("Optimal Objective Values:")
        for id, val in enumerate(optimal_objective_values):
            self.logger.info(f"id:{id}, silog={val[0]:.2f}, flops={val[1]:.2f}G")

        self.logger.info("Optimal Config:")
        for id, conf in enumerate(optimal_solutions):
            derived_archs = self.arch_gener.derive_archs(conf, logger=self.logger)
            self.logger.info(f"id:{id}, {derived_archs}")

        self.logger.info(f"==> Finished evolution!")

    def DetNAS(self, custom_eval_hook, n_iter=30, population_size=40, select_num=8, mutation_num=16, crossover_num=16, m_prob=0.2):
    # def DetNAS(self, custom_eval_hook, n_iter=2, population_size=3, select_num=1, mutation_num=1, crossover_num=1, m_prob=0.1):
        vis_dict={}
        memory=[]
        keep_top_k = {select_num:[],40:[]}
        _epoch = 0
        flops_limit = 17.8*1e9
        code_space = [
            6, 7, 7, 7,      # stage1
            6, 7, 7, 7,     # stage2
            6, 7, 7, 7, 7, 7,  # stage3
            6, 7, 7, 7, 7, 7,  # stage4
            6, 7, 7, 7,      # stage5
            6               # stage6
        ]

        self.logger.info('population_num = {} select_num = {} mutation_num = {} '
              'crossover_num = {} random_num = {} max_epochs = {}'.format(
                population_size, select_num, mutation_num,
                crossover_num,
                population_size - mutation_num - crossover_num,
                n_iter))

        def stack_random_cand(random_func,*,batchsize=10):
            while True:
                cands=[random_func() for _ in range(batchsize)]
                for cand in cands:
                    if cand not in vis_dict:
                        vis_dict[tuple(cand)]={}

                for cand in cands:
                    yield cand

        def update_top_k(candidates,*,k,key,reverse=False):
            assert k in keep_top_k
            print('select ......')
            t=keep_top_k[k]
            t+=candidates
            t.sort(key=key,reverse=reverse)
            keep_top_k[k]=t[:k]

        def get_mutation(k, mutation_num, m_prob):  # mutation_num in top-k by m_prob
            assert k in keep_top_k
            print('mutation ......')
            res = []
            seen = set()
            max_iters = mutation_num*10

            def random_func():
                cand=list(choice(keep_top_k[k]))
                for i in range(len(code_space)):
                    if np.random.random_sample()<m_prob:
                        # cand[i]=np.random.randint(code_space[i])
                        cand[i] = np.random.choice(range(code_space[i]), 
                                       p=balanced_flops_friendly_weights(code_space[i]))
                return tuple(cand)

            cand_iter=stack_random_cand(random_func)
            while len(res)<mutation_num and max_iters>0:
                cand=next(cand_iter)
                if not legal(cand):
                    # print('get_mutation illegal candidate:', cand)
                    max_iters-=1
                    continue
                if cand in seen:  # 去重
                    continue
                seen.add(cand)

                res.append(cand)
                print('mutation {}/{}'.format(len(res),mutation_num))
            
            if len(res) < mutation_num:
                print(f'[WARN] Only got {len(res)} mutation candidates, filling with random')
                res += random_can(mutation_num - len(res))

            print('mutation_num = {}'.format(len(res)))
            return res
        
        def get_crossover(k, crossover_num):
            assert k in keep_top_k
            print('crossover ......')
            res = []
            seen = set()
            max_iters = 10 * crossover_num
            def random_func():
                p1=choice(keep_top_k[k])
                p2=choice(keep_top_k[k])
                return tuple(choice([i,j]) for i,j in zip(p1,p2))
            cand_iter=stack_random_cand(random_func)
            while len(res)<crossover_num and max_iters>0:
                cand=next(cand_iter)
                if not legal(cand):
                    # print('get_crossover illegal candidate:', cand)
                    max_iters-=1
                    continue

                if cand in seen:  # 去重
                    continue
                seen.add(cand)

                res.append(cand)
                print('crossover {}/{}'.format(len(res),crossover_num))
                
            if len(res) < crossover_num:
                print(f'[WARN] Only got {len(res)} crossover candidates, filling with random')
                res += random_can(crossover_num - len(res))

            print('crossover_num = {}'.format(len(res)))
            return res
        
        def random_can(num):
            print('random select ........')
            candidates = []
            cand_iter=stack_random_cand(
                # lambda:tuple(np.random.randint(i) for i in code_space))
                lambda: sample_candidate(code_space))

            while len(candidates)<num:
                cand=next(cand_iter)

                if not legal(cand):
                    # print('random_can illegal candidate:', cand)
                    continue
                candidates.append(cand)
                print('random {}/{}'.format(len(candidates),num))

            print('random_num = {}'.format(len(candidates)))
            return candidates
    
        def legal(cand, if_limit=True):
            assert isinstance(cand,tuple) and len(cand)==len(code_space)
            if cand not in vis_dict:
                vis_dict[cand]={}
            info=vis_dict[cand]
            if 'visited' in info:
                return False
            # print('cand',cand)

            if if_limit:
                if 'flops' not in info:
                    if self.args.dataset == 'kitti':
                        input_shape = (1, 3, 352, 1216)
                    elif self.args.dataset == 'nyu':
                        input_shape = (1, 3, 480, 640)
                    else:
                        assert False,'self.args.dataset is not in [kitti, nyu]'
                    # self.model.module.backbone.alpha_index = decode_arch(cand)
                    # # print('alpha_index = ', self.model.module.backbone.alpha_index)
                    # if hasattr(self.model, 'module'): 
                    #     self.model.module.backbone = Dropped_Network(self.model.module.backbone)
                    # else:
                    #     self.model.backbone = Dropped_Network(self.model.backbone)

                    from mmcv import ConfigDict
                    from mmdet.models import build_detector
                    model_test_cfg = ConfigDict(
                        type='NASRetinaNetTrain',
                        # pretrained=ConfigDict(
                        #     use_load=False,
                        #     load_path='./seed_mbv2.pt',
                        #     seed_num_layers=[1, 1, 2, 3, 4, 3, 3, 1, 1] # mbv2
                        #     ),
                        backbone=ConfigDict(
                            type='FNA_Retinanet',
                            net_config=self.arch_gener.derive_archs(cand),
                            output_indices=[2, 3, 5, 7]
                            ),
                        neck=None,
                        bbox_head=ConfigDict(
                            type='NewcrfsDecoder',
                            dataset=self.args.dataset,
                            with_fapn=True,
                            max_depth=80.0,
                            ))
                    
                    model_test = build_detector(model_test_cfg, train_cfg=None, test_cfg=None)
                    model_test.train()
                    # from torchinfo import summary
                    # summary(self.model, (1, 3, 352, 1216), depth=5)
                    # assert False
                    # import sys
                    # with open('./output.log', 'w') as f:
                    #     original_stdout = sys.stdout
                    #     sys.stdout = f
                    flops, macs, params = calculate_flops(model=model_test, 
                                                    input_shape=input_shape,
                                                    print_results=False,
                                                    print_detailed=False,
                                                    output_as_string=False,
                                                    output_precision=2)
                    del model_test
                    # sys.stdout = original_stdout
                    info['flops']=flops
                flops = info['flops']
                if flops > flops_limit:
                    print(flops, flops_limit)
                    return False
                info['flops']=flops

            vis_dict[cand]=info
            return True

        candidates = random_can(population_size)
        while _epoch < n_iter:
            self.logger.info('epoch = {}'.format(_epoch))
            for cand in candidates:
                info = vis_dict[tuple(cand)]

                supernet_weights = self.load_weights_mmap()
                if hasattr(self.model, 'module'):
                    self.model.module.load_state_dict(supernet_weights, strict=True)
                else:
                    self.model.load_state_dict(supernet_weights, strict=True)

                eval_measures = custom_eval_hook.after_train_epoch(self, force=True, alpha_index=decode_arch(cand)) 

                info['err'] = eval_measures[0]
                # silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3, num
                # import random
                # info['err'] = random.random()  # TODO: debug
            
            memory.append([])
            for cand in candidates:
                memory[-1].append(cand)
                vis_dict[tuple(cand)]['visited'] = True
            
            update_top_k(candidates,k=select_num,key=lambda x:vis_dict[x]['err'])
            update_top_k(candidates,k=40,key=lambda x:vis_dict[x]['err'] )

            self.logger.info('epoch = {} : top {} result'.format(_epoch, len(keep_top_k[40])))
            for i,cand in enumerate(keep_top_k[40]):
                self.logger.info('No.{} {} SILog = {}'.format(i+1, cand, vis_dict[tuple(cand)]['err']))
                # ops = [blocks_keys[i] for i in cand]
                # self.logger.info(ops)

            mutation = get_mutation(select_num, mutation_num, m_prob)
            crossover = get_crossover(select_num, crossover_num)
            rand = random_can(population_size - len(mutation) -len(crossover))

            candidates = mutation+crossover+rand
            _epoch+=1

        self.logger.info(keep_top_k[select_num])
        for id, conf in enumerate(keep_top_k[select_num]):
            derived_archs = self.arch_gener.derive_archs(list(conf), logger=self.logger)
            self.logger.info(f"id:{id}, {derived_archs}")
        self.logger.info('DetNAS finish!')

    def run_epoch_alter(self, workflow, max_epochs, arch_update_epoch, **kwargs):
        """Start running. Arch and weight optimization alternates by epoch.

        Args:
            self.data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(self.data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        alpha_index = kwargs.pop('alpha_index')
        self._epoch = 0  # 复原
        self._iter = 0  # 复原
        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        # self.logger.info('Start running, host: %s, work_dir: %s',
        #                  get_host_info(), work_dir)
        # self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        self.call_hook('before_run', 'train')

        while self.epoch < max_epochs:
            # print('*********** ', self.epoch,' ***********')
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    assert mode in ['train', 'arch', 'val']
                    if mode in ['train', 'arch']:
                        epoch_runner = getattr(self, 'train'+'_epoch_alter')
                    else:
                        epoch_runner = getattr(self, 'val')
                elif callable(mode):  # custom train()
                    epoch_runner = mode
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    'callable function, not {}'.format(
                                        type(mode)))
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epochs:
                        return
                    elif mode in ['arch', 'val'] and self.epoch < arch_update_epoch:
                        break
                    # data_loader = self.data_loaders[0] if mode=='train' else self.data_loaders[1]
                    data_loader = self.data_loaders[1]
                    epoch_runner(data_loader, mode=mode, alpha_index=alpha_index)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run', 'train')


    def train_epoch_alter(self, data_loader, **kwargs):
        """
            Arch and weight optimization alternates by epoch.
        """
        alpha_index = kwargs.pop('alpha_index')
        mode = kwargs.pop('mode')
        self.mode=mode
        self.model.train()
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)
        self.call_hook('before_train_epoch', mode)
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter', mode)
            outputs = self.batch_processor(
                self.model, data_batch, mode=mode, net_type=self.cfg.type, alpha_index=alpha_index, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_train_iter', mode)
            self._iter += 1

        self.call_hook('after_train_epoch', mode)
        if mode == 'train':
            self._epoch += 1


    def run_step_alter(self, workflow, max_epochs, arch_update_epoch, alpha_index=None, **kwargs):
        """Start running. Arch and weight optimization alternates by step.

        Args:
            self.data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(self.data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        
        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        self.call_hook('before_run', 'train')

        while self.epoch < max_epochs:
            self.search_stage = 0 if self.epoch<self.cfg.arch_update_epoch else 1
            self.train_step_alter()

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run', 'train')

    def set_param_grad_state(self, stage):
        def set_grad_state(params, state):
            for group in params:
                for param in group['params']:
                    param.requires_grad_(state)
        if stage == 'arch':
            state_list = [True, False] # [arch, weight]
        elif stage == 'train':
            state_list = [False, True]  # TODO
        else:
            state_list = [False, False]
        set_grad_state(self.arch_optimizer.param_groups, state_list[0])
        set_grad_state(self.optimizer.param_groups, state_list[1])

    def train_step_alter(self, **kwargs):
        """
            Arch and weight optimization alternates by step.
        """
        self.model.train()
        self._max_iters = self._max_epochs * len(self.data_loaders[0])  # [datasets.train_data, datasets.arch_data]
        self._inner_iter = 0
        data_loader_iters = []
        for data_loader in self.data_loaders:
            data_loader_iters.append(iter(data_loader))
        self.data_loader = self.data_loaders[0]
        self.call_hook('before_train_epoch', 'train')  # 没进去
        self.call_hook('before_train_epoch', 'arch')
        while self._inner_iter < len(self.data_loaders[0]):  # 遍历batch次数

            for i, flow in enumerate(self.cfg.workflow):
                mode, steps = flow  # [('arch', 1),('train', 1)]
                self.mode = mode
                self.set_param_grad_state(mode)

                if mode=='arch' and self.search_stage == 0:
                    continue  # 0阶段禁止架构更新
                for _ in range(steps):  # 可改进化时单架构的训练轮次
                    try:
                        data_batch = next(data_loader_iters[i])
                    except:
                        data_loader_iters[i] = iter(self.data_loaders[i])
                        data_batch = next(data_loader_iters[i])
                    self.call_hook('before_train_iter', mode)  # 保存backbone
                    outputs = self.batch_processor(  # 采样+更换backbone+推理
                        self.model, data_batch, mode=mode, 
                        search_stage=self.search_stage, 
                        net_type=self.cfg.type, **kwargs)

                    if not isinstance(outputs, dict):
                        raise TypeError('batch_processor() must return a dict')
                    if 'log_vars' in outputs:
                        self.log_buffer.update(outputs['log_vars'],
                                            outputs['num_samples'])

                    self.outputs = outputs
                    if mode=='arch': # TODO: used for 5-step train
                        tmp = self._inner_iter
                        self._inner_iter -= 1  # 动机不明
                        self.call_hook('after_train_iter', mode)  # ArchDistOptimizerHook
                        self._inner_iter = tmp
                    else:
                        self.call_hook('after_train_iter', mode)  # DistOptimizerHook更新梯度 和 DropProcessHook恢复backbone
                    if mode=='train':
                        self._inner_iter += 1
                        self._iter += 1

        self.call_hook('after_train_epoch', 'train')  # CustomDistEvalHook（验证架构性能）
        self.call_hook('after_train_epoch', 'arch')  # ModelInfoHook（打印a_weight, 编码，MADDs和Param）
        self._epoch += 1


    def init_nas_optimizer(self, optimizers):
        """
        init nas optimizer: weight optimizer and architecture optimizer
        args:
            dict (weight_optim config & arch_optim config)
        returns:
            tuple(weight_optim, arch_optim)
        """

        if isinstance(optimizers, dict):
            assert hasattr(optimizers, 'weight_optim') and hasattr(optimizers, 'arch_optim')
            optim_list = []
            arch_params_id = list(map(id, self.model.module.backbone.arch_parameters
                if hasattr(self.model, 'module') else self.model.backbone.arch_parameters))
            weight_params = filter(lambda p: id(p) not in arch_params_id, self.model.parameters())
            arch_params = filter(lambda p: id(p) in arch_params_id, self.model.parameters())  # 不再使用

            for key, optim in optimizers.items():
                if key == 'weight_optim':
                    params = weight_params
                elif key == 'arch_optim':
                    params = arch_params
                else:
                    assert KeyError
                optimizer = obj_from_dict(optim.optimizer, torch.optim, dict(params=params))
                optim_list.append(optimizer)
        else:
            raise TypeError(
                'optimizer must be a dict, ''but got {}'.format(type(optimizers)))

        return tuple(optim_list)


    def current_lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate of all param groups.
        """
        if self.optimizer is None or self.arch_optimizer is None:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        if self.mode == 'train':
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self.mode == 'arch':
            return [group['lr'] for group in self.arch_optimizer.param_groups]
    
    def unregister_hook(self, hook_class):
        self._hooks = [h for h in self._hooks if not isinstance(h, hook_class)]

    def register_training_hooks(self,
                                lr_config,
                                weight_optim_config=None,
                                arch_optim_config=None,
                                checkpoint_config=None,
                                log_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - Weight/Arch_OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        if weight_optim_config is None:
            weight_optim_config = {}
        if arch_optim_config is None:
            arch_optim_config = {}
        if checkpoint_config is None:
            checkpoint_config = {}

        self.register_lr_hooks(lr_config)
        self.register_hook(self.build_hook(weight_optim_config, OptimizerHook))
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        # self.register_hook(ModelInfoHook(self.cfg.model_info_interval), priority='VERY_LOW')
        self.register_hook(DropProcessHook(), priority='LOW')
        self.register_hook(IterTimerHook())

        self.register_arch_hook(self.build_hook(arch_optim_config, ArchOptimizerHook))
        self.register_arch_hook(ModelInfoHook(self.cfg.model_info_interval), priority='VERY_LOW')
        self.register_arch_hook(DropProcessHook(), priority='LOW')
        self.register_arch_hook(IterTimerHook())
        if log_config is not None:
            self.register_logger_hooks(log_config) # logger_hook for arch_hook will be added inside


    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            if info.type == 'TextLoggerHook':
                logger_hook = NASTextLoggerHook(log_interval)
            else:
                logger_hook = obj_from_dict(
                    info, hooks, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority='VERY_LOW')
            del logger_hook.priority
            self.register_arch_hook(logger_hook, priority='VERY_LOW')


    def register_arch_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._arch_hooks) - 1, -1, -1):
            if priority >= self._arch_hooks[i].priority:
                self._arch_hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._arch_hooks.insert(0, hook)


    def call_hook(self, fn_name, mode='train'):
        hooks_run = self._hooks if mode=='train' else self._arch_hooks
        for hook in hooks_run:
            getattr(hook, fn_name)(self)


    def load_checkpoint(self, filename, map_location='cpu', strict=True):
        self.logger.info('load checkpoint from %s', filename)

        if filename.startswith(('http://', 'https://')):
            url = filename
            filename = '../' + url.split('/')[-1]
            if get_dist_info()[0]==0:
                if osp.isfile(filename):
                    os.system('rm '+filename)
                os.system('wget -N -q -P ../ ' + url)
            dist.barrier()

        return load_checkpoint(self.model, filename, map_location, strict,
                               self.logger)
