from __future__ import division

import argparse
import numpy as np
import os
import os.path as osp
import sys
sys.path.append(osp.join(sys.path[0], '..'))
import time
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"

import models
from mmcv import Config
from mmdet import __version__
from mmdet.apis import init_dist, set_random_seed
from mmdet.datasets import get_dataset
from mmdet.models import build_detector
from tools import utils
from tools.apis.fna_search_apis import search_detector
from tools.divide_dataset import build_divide_dataset
from dataloader import NewDataLoader

from seg_file.data.util.config import load_cfg_from_cfg_file
from seg_file.dataloader import GetDataloader

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
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    # parser.add_argument('--port', type=int, default=23333, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--load_supernet_path', type=str, default=None, help='will skip training supernet')

    # Dataset
    parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='kitti', required=True)
    parser.add_argument('--data_path',                 type=str,   help='path to the data')
    parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data')
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file')
    parser.add_argument('--input_height',              type=int,   help='input height', default=352)
    parser.add_argument('--input_width',               type=int,   help='input width',  default=1120)
    parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=80.0)
    parser.add_argument('--batch_size',                type=int,   help='batch size per one GPU', default=4)

    # Online eval
    parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
    parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation')
    parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation')
    parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation')
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
    parser.add_argument('--weight_lr',                  type=float,   help='')
    parser.add_argument('--weight_decay',               type=float,   help='')
    parser.add_argument('--arch_lr',                    type=float,   help='')
    parser.add_argument('--arch_update_epoch',          type=int,   help='')

    # segmentation
    parser.add_argument('--config_seg', type=str, default=None, help='config_seg file')

    # DetNAS method
    parser.add_argument('--DetNAS', help='if set, will use DetNAS method', action='store_true')
    
    # crossover
    parser.add_argument(
        "--crossover_type",
        type=str,
        default="int_two_point",  # TODO
        choices=["int_one_point", "int_two_point", "None"],
        help="Crossover operator to use. Use 'None' to disable crossover."
    )
    
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.config_seg:
        cfg_seg = load_cfg_from_cfg_file(args.config_seg)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.devices
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus
    cfg.model['bbox_head']['max_depth'] = args.max_depth  # !!!
    cfg.model['bbox_head']['dataset'] = args.dataset

    # change cfg
    if args.total_epochs is not None:
        cfg.total_epochs = args.total_epochs
    if args.weight_lr is not None:
        cfg.optimizer['weight_optim']['optimizer']['lr'] = args.weight_lr
    if args.arch_update_epoch is not None:
        cfg.arch_update_epoch = args.arch_update_epoch
    if args.weight_decay is not None:
        cfg.optimizer['weight_optim']['optimizer']['weight_decay'] = args.weight_decay
    if args.arch_lr is not None:
        cfg.optimizer_search['weight_optim']['optimizer']['lr'] = args.arch_lr

    # add args
    args.train_data_ratio = cfg.train_data_ratio

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
        assert False,'Non-distributed has been abandoned'
    else:
        args.distributed = True
        os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '%d' % args.port  # 在torchrun指令中加入 --master_port=29501 即可
        init_dist(args.launcher, **cfg.dist_params)
        
    utils.setup_work_dir(args,cfg)
    # init logger before other steps
    utils.create_work_dir(cfg.work_dir)
    logger = utils.get_root_logger(cfg.work_dir, cfg.log_level)
    logger.info('Distributed training: {}'.format(args.distributed))
    logger.info('Search args: \n'+str(args))
    logger.info('Search configs: \n'+str(cfg))
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

    # 打印网络结构
    # file_path = './net_{}.txt'.format(time.time())
    # with open(file_path, 'w') as f:
    #     f.write(str(model)) 
    # assert False,'test'
    
    model.backbone.get_sub_obj_list(cfg.sub_obj, (1, 3,)+cfg.image_size_madds)

    if cfg.use_syncbn:
        model = utils.convert_sync_batchnorm(model)

    # gpu_num = torch.cuda.device_count()
    # args.batch_size = int(args.batch_size / gpu_num)
    # 准备数据集
    if args.dataset in ['kitti','nyu']:
        arch_search_dataset = NewDataLoader(args, 'arch_search')
        eval_dataset = NewDataLoader(args, 'online_eval') if args.validate else None

    elif args.dataset == 'cityscapes':
        arch_search_dataset = GetDataloader(cfg_seg, args, 'arch_search')
        eval_dataset = GetDataloader(cfg_seg, args, 'online_eval') if args.validate else None

    elif args.dataset in ['smalldata', 'blender', 'colon']:
        from med_dataloader.MedDataloader import GetMedDataloader
        arch_search_dataset = GetMedDataloader(args, 'arch_search')
        eval_dataset  = GetMedDataloader(args, 'online_eval') if args.validate else None

    elif args.dataset == 'imagenet':
        from imagenet_dataloader import GetImageNetDataloader
        arch_search_dataset = GetImageNetDataloader(args, 'arch_search')
        eval_dataset  = GetImageNetDataloader(args, 'online_eval') if args.validate else None

    else:
        assert False,'The args.dataset is invalid name of dataset!'

    search_detector(model, 
                    arch_search_dataset,
                    eval_dataset,
                    cfg,
                    args,
                    distributed=args.distributed,
                    validate=args.validate,
                    logger=logger)


if __name__ == '__main__':
    main()
