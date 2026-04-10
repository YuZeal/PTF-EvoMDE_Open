import torch
import numpy as np
from tqdm import tqdm

import torch.distributed as dist
from mmdet.core import (DistEvalHook)
from utils_newcrfs import post_process_depth, compute_errors, flip_lr
from models.dropped_model import Dropped_Network
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 自定义验证hook
class CustomDistEvalHook_seg(DistEvalHook):

    def __init__(self, dataset, args, logger, work_dir=None, interval=1, isSearch=False):
        # 直接将数据集赋值给 self.dataset，不做预处理
        self.dataset = dataset
        self.interval = interval
        self.classes = 19
        self.ignore_label = 255

        self.dataset_name = args.dataset

        self.logger = logger
        self.isSearch = isSearch
        self.work_dir = work_dir
        self.best_meter = 0

    def after_train_epoch(self, runner, force=False, alpha_index=None):  # 继承并重写after_train_epoch
        if not self.every_n_epochs(runner, self.interval) and not force:
            return

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

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
                target = eval_sample_batched['label'].cuda(runner.rank, non_blocking=True)

                # compute output
                output = runner.model(image=image, return_loss=False)
                # print('bf',output.shape)  # [8, 19, 192, 192]
                target = target.squeeze(1)
                output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
                # print('md',output.shape)  # [8, 19, 768, 768]
                output = output.max(1)[1]
                # print('af',output.shape)  # [8, 768, 768]
                intersection, union, target = intersectionAndUnionGPU(output, target, self.classes, self.ignore_label)

                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
                accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

            # 更新进度条
            if runner.rank == 0:
                pbar.update(1)

        # 关闭进度条        
        if runner.rank == 0:
            pbar.close()

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        dist.barrier()

        
        if self.isSearch:  # 搜索阶段，网络架构变换
            # 还原网络
            if hasattr(runner.model, 'module'):
                runner.model.module.backbone = save_backbone
            else:
                runner.model.backbone = save_backbone


        if runner.rank == 0:
            eval_measures_str = '{:7.4f}, {:7.4f}, {:7.4f}'.format(mIoU, mAcc, allAcc)
            self.logger.info('Computing Val result:')
            self.logger.info("{:>7}, {:>7}, {:>7}".format('mIoU', 'mAcc', 'allAcc'))
            self.logger.info(eval_measures_str)

            # save weight of the best model 
            if self.work_dir != None:
                cur_meter = mIoU  # abs_rel is metrics (this can change)
                if cur_meter > self.best_meter:
                    self.best_meter = cur_meter
                    runner.save_checkpoint(out_dir=self.work_dir, filename_tmpl='best.pth',save_optimizer=False)
                    self.logger.info(f'save the best weight with mIoU:{cur_meter}')
                runner.save_checkpoint(out_dir=self.work_dir, filename_tmpl='newest.pth',save_optimizer=False)

        # assert False,'test'

        return [mIoU, mAcc, allAcc]


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    '''code from https://github.com/houqb/SPNet'''
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()