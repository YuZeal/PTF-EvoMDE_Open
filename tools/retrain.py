from __future__ import division

import argparse
import json
import os
import os.path as osp
import sys
import time
import gc
sys.path.append(osp.join(sys.path[0], '..'))

import torch
import torch.nn as nn
from torch.utils import data

import models
from mmcv import Config
from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed)
from tools.apis.train import train_detector
from mmdet.datasets import get_dataset
from mmdet.models import build_detector, detectors
from tools import utils
from mmdet.datasets import build_dataloader
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from dataloader import NewDataLoader

from seg_file.data.util.config import load_cfg_from_cfg_file
from seg_file.dataloader import GetDataloader
from med_dataloader.get_medDataloader import create_data_loaders
from med_dataloader.MedDataloader import GetMedDataloader
# os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"

import warnings
warnings.simplefilter("ignore")
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', type=str, help='the dir to save logs and models')
    parser.add_argument('--job_name', type=str, default='', help='job name for output path')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--port', type=int, default=23333, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    # Dataset
    parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='kitti', required=True)
    parser.add_argument('--data_path',                 type=str,   help='path to the data')
    parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data')
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file')
    parser.add_argument('--input_height',              type=int,   help='input height', default=352)
    parser.add_argument('--input_width',               type=int,   help='input width',  default=1120)
    parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=100, required=False)
    parser.add_argument('--batch_size',                type=int,   help='batch size per one GPU', default=4)

    # Online eval
    parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
    parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
    parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
    parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
    parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
    parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
    parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                        'if empty outputs to checkpoint folder', default='')
    parser.add_argument('--post_process',               type=bool,   help='filp image to eval', default=True)


    # Preprocessing
    parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
    parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

    parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)

    # cfg
    parser.add_argument('--devices',                    type=str, default='4,5,6,7', help='CUDA_VISIBLE_DEVICES value, e.g., "0" or "0,1"')
    parser.add_argument('--total_epochs',               type=int,   help='')
    parser.add_argument('--lr',                         type=float,   help='')
    parser.add_argument('--weight_decay',               type=float,   help='')
    parser.add_argument('--net_arch',                   type=str,   help='')
    parser.add_argument('--betas',                      type=float,   help='')
    
    parser.add_argument('--use_offi_ecd',               type=str,   help='Using official net with pretrained weight as backbone.e.g., mbv2, r101')
    parser.add_argument('--pretrain_from_imagenet',     action='store_true',   help='Using pretrained weight from training on ImageNet')
    parser.add_argument('--pretrained_path',            type=str,   help='The path of pretrained weight from training on ImageNet')

    # segmentation
    parser.add_argument('--config_seg', type=str, default=None, help='config_seg file')
    args = parser.parse_args()

    return args


def main():
    # 定义一个映射字典，将选项与相应的函数关联起来
    backbone_options = {
        'mbv2': ('deepFeatureExtractor_MobileNetV2', 'MobileNetv2_100'),
        'rs18': ('deepFeatureExtractor_ResNet18', 'ResNet18'),
        'eb0': ('deepFeatureExtractor_EfficientNetB0', 'EfficientNet_B0'),
        'mn1': ('deepFeatureExtractor_MNASNet1_0', 'MNASNet1_0'),
        'mn075': ('deepFeatureExtractor_MNASNet0_75', 'MNASNet0_75'),
        'sfv2': ('deepFeatureExtractor_ShuffleNet_V2', 'ShuffleNet_V2'),
        'mbv3s':('deepFeatureExtractor_Mobilenet_v3_small', 'Mobilenet_v3_small'),
        'mbv3L':('deepFeatureExtractor_Mobilenet_v3_large', 'Mobilenet_v3_large'),
        'rgy400':('deepFeatureExtractor_RegNet_Y_400MF','RegNet_Y_400MF'),
        'rgx400':('deepFeatureExtractor_RegNet_X_400MF','RegNet_X_400MF'),
        'cvnxt':('deepFeatureExtractor_ConvNeXt_Tiny','ConvNeXt_Tiny'),
        'ds121':('deepFeatureExtractor_DenseNet121','DenseNet121'),
        'rsnxt50':('deepFeatureExtractor_ResNeXt50_32x4d','ResNeXt50_32x4d'),
        'rgx8g':('deepFeatureExtractor_RegNet_X_8gF','RegNet_X_8gF'),
        'ds161':('deepFeatureExtractor_DenseNet161','DenseNet161'),
        'efb4':('deepFeatureExtractor_EfficientNetB4','EfficientNetB4'),
        'rgx8g_':('deepFeatureExtractor_RegNet_X_8gF_2','RegNet_X_8gF_2t16'),
        'ds161_':('deepFeatureExtractor_DenseNet161_2','DenseNet161_2t16'),
        'efb7_':('deepFeatureExtractor_EfficientNetB7_2','EfficientNetB7_2t16')
    }

    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.config_seg:
        cfg_seg = load_cfg_from_cfg_file(args.config_seg)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.devices
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        if args.job_name == '':
            args.job_name = 'output_XX_retrain'
        else:
            args.job_name = 'output_' + time.strftime("%Y%m%d-%H%M%S_") + args.job_name
        cfg.work_dir = osp.join(args.work_dir, args.job_name)
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus
    if args.dataset not in ["cityscapes"]:
        cfg.model['bbox_head']['max_depth'] = args.max_depth  # !!!

    # change cfg
    if args.total_epochs is not None:
        cfg.total_epochs = args.total_epochs
    if args.lr is not None:
        cfg.optimizer['lr'] = args.lr
    if args.weight_decay is not None:
        cfg.optimizer['weight_decay'] = args.weight_decay
    if args.betas is not None:
        cfg.optimizer['betas'] = (args.betas, cfg.optimizer['betas'][1])
    if args.net_arch is not None:
        cfg.model['backbone']['net_config'] = args.net_arch
    
    # 检查传入的选项是否在字典中
    if args.use_offi_ecd in backbone_options:
        module_name, model_name = backbone_options[args.use_offi_ecd]
        # 动态导入模块
        module = __import__('models.official_encoder', fromlist=[module_name])
        model_class = getattr(module, module_name)
        # 直接实例化并访问 dimList 属性
        cfg.model['bbox_head']['in_channels'] = getattr(model_class(), 'dimList', None)

    torch.cuda.empty_cache()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '%d' % args.port  # 会卡住
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    # logger = get_root_logger(cfg.log_level)
    utils.create_work_dir(cfg.work_dir)
    logger = utils.get_root_logger(cfg.work_dir, cfg.log_level)
    logger.info('Distributed training: {}'.format(args.distributed))
    logger.info('Retrain configs: \n'+str(cfg))
    logger.info('Retrain args: \n'+str(args))
    if args.config_seg:
        logger.info('Retrain segmentation configs: \n'+str(cfg_seg))

    if cfg.checkpoint_config is not None:
        # save mmdet version in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__, config=cfg.text)

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)
    
    # utils.set_data_path(args.data_path, cfg.data)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if not hasattr(model, 'neck'):
        model.neck = None

    # print("Model Parameters:")
    # for name, param in list(model.named_parameters()):
    #     print(f"Name: {name}, Shape: {param.shape}")

    # 打印网络结构
    # file_path = './net_{}.txt'.format(time.time())
    # with open(file_path, 'w') as f:
    #     f.write(str(model)) 
    # assert False,'test'

    assert not (args.use_offi_ecd and args.pretrain_from_imagenet), "Both 'use_offi_ecd' and 'pretrain_from_imagenet' cannot be set at the same time."

    if args.pretrain_from_imagenet:
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        model_tar = model.backbone.state_dict()
        model_pre = checkpoint['state_dict']
        keys_model_tar = model_tar.keys()
        keys_model_pre = model_pre.keys()

        # fpath = './keyOfTarget.txt'
        # with open(fpath, 'w') as f:
        #     for key in keys_model_tar:
        #         f.write(key+'\n')
        # fpath = './keyOfPretrain.txt'
        # with open(fpath ,'w') as f:
        #     for key in keys_model_pre:
        #         f.write(key+'\n')

        # 遍历模型 A 和模型 B 的权重，按顺序将模型 B 的权重赋值给模型 A
        for (key_tar, key_pre) in zip(keys_model_tar, keys_model_pre):
            if 'num_batches_tracked' in key_tar:
                continue
            if model_tar[key_tar].shape == model_pre[key_pre].shape:
                model_tar[key_tar] = model_pre[key_pre]  # 按顺序赋值
            else:
                raise ValueError(f"Shape mismatch for {key_tar}: "
                            f"model A shape {model_tar[key_tar].shape}, "
                            f"model B shape {model_pre[key_pre].shape}")
        model.backbone.load_state_dict(model_tar)  # 真正改变权重
        logger.info('Backbone weight has been loaded successfully from the ImageNet pretrained weight!!!')
        del checkpoint  # 删除checkpoint，释放内存
        gc.collect()  # 手动调用垃圾回收，确保内存被释放


    # 检查传入的选项是否在字典中
    if args.use_offi_ecd in backbone_options:
        module_name, model_name = backbone_options[args.use_offi_ecd]
        # 动态导入模块
        module = __import__('models.official_encoder', fromlist=[module_name])
        model_class = getattr(module, module_name)
        model.backbone = model_class()
        logger.info(f'!!!已使用官方{model_name}的backbone!!!')
    elif args.use_offi_ecd == 'mde_evonas':
        from yzh_paper1.mde_evonas import MdeEvoNAS
        model.backbone = MdeEvoNAS()
        logger.error(f'使用的编码器选项: {args.use_offi_ecd}')
    elif args.use_offi_ecd is not None:
        logger.error(f'未知的编码器选项: {args.use_offi_ecd}') 
                
    logger.info('Backbone net config: \n' + cfg.model.backbone.net_config)

    '''show param'''
    # 计算FLOPs 和 Params     # TODO
    '''https://github.com/MrYxJ/calculate-flops.pytorch?tab=readme-ov-file'''
    # print('*'*20, ' calflops ', '*'*20)
    # from calflops import calculate_flops
    # # from torchvision import models
    # import sys
    # with open('./output.log', 'w') as f:
    #     original_stdout = sys.stdout
    #     sys.stdout = f
    #     # model = models.resnet50()
    #     input_shape = (1, 3, 320, 320)
    #     flops, macs, params = calculate_flops(model=model, 
    #                                         input_shape=input_shape,
    #                                         output_as_string=True,
    #                                         output_precision=2)
    #     sys.stdout = original_stdout
    #     print("net FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))        
    #     #Alexnet FLOPs:4.2892 GFLOPS   MACs:2.1426 GMACs   Params:61.1008 M 
    # assert False, 'over'

    # 测fps
    # from cal_flops_fps import cal_flops_fps
    # cal_flops_fps(model=model)
    # assert False, 'over'

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())/1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
    logger.info(f"总参数量 (Total parameters): {total_params}M")
    logger.info(f"训练参数量 (Trainable parameters): {trainable_params}M")


    utils.get_network_madds(model.backbone, model.neck, model.bbox_head, 
                            cfg.image_size_madds, logger)

    if cfg.use_syncbn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # gpu_num = torch.cuda.device_count()
    # args.batch_size = int(args.batch_size / gpu_num)
    # train_dataset = get_dataset(cfg.data.train)

    if args.dataset in ['kitti','nyu']:
        train_dataset = NewDataLoader(args, 'train')
        eval_dataset  = NewDataLoader(args, 'online_eval')
    elif args.dataset == 'cityscapes':
        train_dataset = GetDataloader(cfg_seg, args, 'train')
        eval_dataset  = GetDataloader(cfg_seg, args, 'online_eval')
    elif args.dataset in ['smalldata', 'blender', 'colon']:
        train_dataset = GetMedDataloader(args, 'train')
        eval_dataset  = GetMedDataloader(args, 'online_eval')

    else:
        assert False,'The args.dataset is invalid name of dataset!'

    train_detector(
        model,
        (train_dataset, eval_dataset),
        cfg,
        args,
        distributed=args.distributed,
        validate=args.validate,
        logger=logger)

    logger.info('Backbone net config: \n' + cfg.model.backbone.net_config)
    utils.get_network_madds(model.backbone, model.neck, model.bbox_head, 
                            cfg.image_size_madds, logger)


if __name__ == '__main__':
    main()
