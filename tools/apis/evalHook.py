import torch
import numpy as np
from tqdm import tqdm

import torch.distributed as dist
from mmdet.core import (DistEvalHook)
from utils_newcrfs import post_process_depth, compute_errors, flip_lr
from models.dropped_model import Dropped_Network

# 自定义验证hook
class CustomDistEvalHook(DistEvalHook):

    def __init__(self, dataset, args, logger, work_dir=None, interval=1, isSearch=False):
        # 直接将数据集赋值给 self.dataset，不做预处理
        self.dataset = dataset
        self.interval = interval
        self.do_kb_crop = args.do_kb_crop
        self.min_depth_eval = args.min_depth_eval
        self.max_depth_eval = args.max_depth_eval
        self.eigen_crop = args.eigen_crop
        self.garg_crop = args.garg_crop
        self.dataset_name = args.dataset
        self.post_process = args.post_process
        self.logger = logger
        self.isSearch = isSearch
        self.work_dir = work_dir
        self.best_loss = np.inf

    def after_train_epoch(self, runner, force=False, alpha_index=None):  # 继承并重写after_train_epoch
        if not self.every_n_epochs(runner, self.interval) and not force:
            return
        eval_measures = torch.zeros(10).cuda(device=runner.rank)

        if self.isSearch:  # 搜索阶段，网络架构变换
            # 存档网络（简陋的补丁）
            if hasattr(runner.model, 'module'):
                save_backbone = runner.model.module.backbone
            else:
                save_backbone = runner.model.backbone

        if alpha_index != None:
            # 修改网络
            runner.model.module.backbone.alpha_index = alpha_index  
            DroppedBackBone = Dropped_Network
            if hasattr(runner.model, 'module'): 
                runner.model.module.backbone = DroppedBackBone(runner.model.module.backbone)
            else:
                runner.model.backbone = DroppedBackBone(runner.model.backbone)
        # print(runner.model.module.backbone)  # 检查模型架构

        runner.model.eval()
        dataloader_eval = self.dataset
        if runner.rank == 0:
            pbar = tqdm(total=len(dataloader_eval), desc="Evaluation Progress")
        for _, eval_sample_batched in enumerate(dataloader_eval):
            with torch.no_grad():
                image = eval_sample_batched['image'].cuda(runner.rank, non_blocking=True)
                gt_depth = eval_sample_batched['depth']
                has_valid_depth = eval_sample_batched['has_valid_depth']
                if self.dataset_name in ['kitti', 'nyu'] and not has_valid_depth:
                    # print('Invalid depth. continue.')
                    if runner.rank == 0:
                        pbar.update(1)
                    continue
                # compute output
                pred_depth = runner.model(image=image, return_loss=False)
                if self.post_process:
                    image_flipped = flip_lr(image)
                    pred_depth_flipped = runner.model(image=image_flipped, return_loss=False)
                    pred_depth = post_process_depth(pred_depth, pred_depth_flipped)

                pred_depth = pred_depth.cpu().numpy().squeeze()
                gt_depth = gt_depth.cpu().numpy().squeeze()
            
            # 计算指标
            if self.do_kb_crop:
                height, width = gt_depth.shape
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
                pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
                pred_depth = pred_depth_uncropped

            pred_depth[pred_depth < self.min_depth_eval] = self.min_depth_eval
            pred_depth[pred_depth > self.max_depth_eval] = self.max_depth_eval
            pred_depth[np.isinf(pred_depth)] = self.max_depth_eval
            pred_depth[np.isnan(pred_depth)] = self.min_depth_eval
            
            valid_mask = np.logical_and(gt_depth > self.min_depth_eval, gt_depth < self.max_depth_eval)

            if self.garg_crop or self.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if self.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif self.eigen_crop:
                    if self.dataset_name == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    elif self.dataset_name == 'nyu':
                        eval_mask[45:471, 41:601] = 1

                valid_mask = np.logical_and(valid_mask, eval_mask)
            
            if self.dataset_name == 'smalldata':
                eval_mask = create_colon_mask()  # mask掉边角黑暗区域
                eval_mask = np.repeat(np.expand_dims(eval_mask, axis=0), gt_depth.shape[0], axis=0)  # 增加维度并重复
                if valid_mask.shape != eval_mask.shape:
                    raise ValueError("gt_depth and eval_mask must have the same shape.")
                valid_mask = np.logical_and(valid_mask, eval_mask)

            measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
            # print('measures:',measures, runner.rank)
            eval_measures[:9] += torch.tensor(measures).cuda(device=runner.rank)
            eval_measures[9] += 1

            # 更新进度条
            if runner.rank == 0:
                pbar.update(1)

        # 关闭进度条        
        if runner.rank == 0:
            pbar.close()

        dist.barrier()
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM)
        
        if self.isSearch:  # 搜索阶段，网络架构变换
            # 还原网络
            if hasattr(runner.model, 'module'):
                runner.model.module.backbone = save_backbone
            else:
                runner.model.backbone = save_backbone

        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        
        if runner.rank == 0:
            self.logger.info('Computing errors for {} eval samples, post_process: {}'.format(int(cnt), self.post_process))
            self.logger.info("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                        'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                        'd3'))

            eval_measures_str = ', '.join(['{:7.4f}'.format(eval_measures_cpu[i]) for i in range(8)])
            eval_measures_str += ', {:7.4f}'.format(eval_measures_cpu[8])
            self.logger.info(eval_measures_str)

            # save weight of the best model 
            if self.work_dir != None:
                cur_loss = eval_measures_cpu[1]  # abs_rel is metrics (this can change)
                if cur_loss < self.best_loss:
                    self.best_loss = cur_loss
                    runner.save_checkpoint(out_dir=self.work_dir, filename_tmpl='best.pth',save_optimizer=False)
                    self.logger.info(f'save the best weight with abs_rel:{cur_loss}')

        # assert False,'test'
        dist.barrier()
        return eval_measures_cpu


def create_colon_mask(size: int = 320, circle_radius: int = 180) -> np.ndarray:
    """创建掩码以去除colon数据集四角阴影区域。

    参数:
        size (int): 掩码的大小（宽度和高度）。
        circle_radius (int): 圆的半径。

    返回:
        np.ndarray: 布尔掩码，表示有效区域。
    """
    # 找到数组中心坐标
    center = size // 2
    # 生成网格坐标
    y, x = np.ogrid[:size, :size]
    # 计算到中心点的距离的平方
    distance_to_center_sq = (x - center) ** 2 + (y - center) ** 2
    # 计算圆的半径的平方
    radius_sq = circle_radius ** 2
    # 将距离中心点小于半径的位置设为true
    result_array = distance_to_center_sq < radius_sq

    return result_array