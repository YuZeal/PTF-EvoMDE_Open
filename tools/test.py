import argparse
import json

import torch
import sys
import os.path as osp
sys.path.append(osp.join(sys.path[0], '..'))

import mmcv
from mmcv.parallel import MMDataParallel
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from models.fna_retinanet_detector_train import NASRetinaNetTrain
import os

import numpy as np
from utils_newcrfs import post_process_depth, compute_errors, flip_lr, compute_errors_colon

from tqdm import tqdm
from dataloader import NewDataLoader
from med_dataloader.MedDataloader import GetMedDataloader


os.environ["CUDA_VISIBLE_DEVICES"]="0"

### The script file in 'scripts/test/'

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--net_arch', default=None, type=str, help='net_config')

    # Dataset
    parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='kitti', required=True)
    parser.add_argument('--input_height',              type=int,   help='input height', default=352)
    parser.add_argument('--input_width',               type=int,   help='input width',  default=1120)
    parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=80)
    parser.add_argument('--batch_size',                type=int,   help='batch size per one GPU', default=4)

    # Online eval
    parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
    parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
    parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
    parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
    parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('--post_process',               type=bool,   help='filp image to eval', default=True)

    # Preprocessing
    parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')
    parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
    parser.add_argument('--with_fapn',                        help='if set, FaPN module will be used in decoder', action='store_true')

    args = parser.parse_args()
    return args

def single_test(model, data_loader, args):
    model.eval()
    eval_measures = torch.zeros(10).cuda()
    dataloader_eval = data_loader
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval, total=len(dataloader_eval), desc="Test Progress")):
        with torch.no_grad():
            image = eval_sample_batched['image'].cuda()
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if args.dataset not in ['colon'] and not has_valid_depth:
                # print('Invalid depth. continue.')
                continue
            # compute output
            pred_depth = model(image=image, return_loss=False)
            if args.post_process:
                image_flipped = flip_lr(image)
                pred_depth_flipped = model(image=image_flipped, return_loss=False)
                pred_depth = post_process_depth(pred_depth, pred_depth_flipped)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()
            
        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval
        
        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        if args.dataset == 'colon':
            measures = compute_errors_colon(gt_depth[valid_mask], pred_depth[valid_mask])
            eval_measures[:4] += torch.tensor(measures).cuda()
            eval_measures[4] += 1
        else:
            measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
            eval_measures[:9] += torch.tensor(measures).cuda()
            eval_measures[9] += 1
            

    if args.dataset == 'colon':
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[4].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples, post_process: {}'.format(int(cnt), args.post_process))
        print("{:>20}, {:>20}, {:>20}, {:>20}".format('mean_l1_error', 'mean_rel_l1_error', 'mean_rmse', 'd05'))

        eval_measures_str = ', '.join(['{:20.4f}'.format(eval_measures_cpu[i]) for i in range(3)])
        eval_measures_str += ', {:20.4f}'.format(eval_measures_cpu[3])
        print(eval_measures_str)
    else:
    # For kitti and nyu
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples, post_process: {}'.format(int(cnt), args.post_process))
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                    'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                    'd3'))

        eval_measures_str = ', '.join(['{:7.4f}'.format(eval_measures_cpu[i]) for i in range(8)])
        eval_measures_str += ', {:7.4f}'.format(eval_measures_cpu[8])
        print(eval_measures_str)

    return eval_measures_cpu

def main():
    torch.cuda.empty_cache()
    args = parse_args()
    args.distributed = False
    if args.config_seg:
        cfg_seg = load_cfg_from_cfg_file(args.config_seg)
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1
    
    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if args.net_arch is not None:
        cfg.model['backbone']['net_config'] = args.net_arch
    if args.dataset not in ['cityscapes']:
        cfg.model['bbox_head']['max_depth'] = args.max_depth
        cfg.model['bbox_head']['with_fapn'] = args.with_fapn
    if args.use_offi_ecd in backbone_options:
        module_name, model_name = backbone_options[args.use_offi_ecd]
        # 动态导入模块
        module = __import__('models.official_encoder', fromlist=[module_name])
        model_class = getattr(module, module_name)
        # 直接实例化并访问 dimList 属性
        cfg.model['bbox_head']['in_channels'] = getattr(model_class(pretrained=False), 'dimList', None)

    print('configs: \n'+str(cfg))
    print('args: \n'+str(args))

    model = build_detector(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
                
    print('Backbone net config: \n' + cfg.model.backbone.net_config)
    
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))
    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    model = MMDataParallel(model.cuda())
    print("== Model Initialized")

    load_checkpoint(model, args.checkpoint)

    if args.dataset in ['kitti','nyu']:
        eval_dataset  = NewDataLoader(args, 'online_eval')
    elif args.dataset in ['colon']:
        eval_dataset  = GetMedDataloader(args, 'test')

    single_test(model, eval_dataset.data, args)


if __name__ == '__main__':
    main()