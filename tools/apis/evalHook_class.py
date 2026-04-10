import torch
import numpy as np
from tqdm import tqdm

import torch.distributed as dist
from mmdet.core import (DistEvalHook)
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
class CustomDistEvalHook_class(DistEvalHook):

    def __init__(self, dataset, args, logger, work_dir=None, interval=1, isSearch=False):
        # 直接将数据集赋值给 self.dataset，不做预处理
        self.dataset = dataset
        self.interval = interval

        self.dataset_name = args.dataset

        self.logger = logger
        self.isSearch = isSearch
        self.work_dir = work_dir
        self.best_meter = np.inf

    def after_train_epoch(self, runner, force=False, alpha_index=None):  # 继承并重写after_train_epoch
        
        if not self.every_n_epochs(runner, self.interval) and not force:
            return
        # assert False,'developing'

        Top1_err = AverageMeter()
        Top5_err = AverageMeter()

        if self.isSearch:  # 搜索阶段，网络架构变换
            # 存档网络（简陋的补丁）
            if hasattr(runner.model, 'module'):
                save_backbone = runner.model.module.backbone
            else:
                save_backbone = runner.model.backbone

            DroppedBackBone = Dropped_Network
            if hasattr(runner.model, 'module'): 
                runner.model.module.backbone = DroppedBackBone(runner.model.module.backbone)
            else:
                runner.model.backbone = DroppedBackBone(runner.model.backbone)
        
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
                prec1, prec5 = Accuracy(output, target, topk=(1, 5))

                # Top1_err += 1 - prec1.item() / 100
                # Top5_err += 1 - prec5.item() / 100
                Top1_err.update(1 - prec1.item() / 100)
                Top5_err.update(1 - prec5.item() / 100)

                # dist.all_reduce(Top1_err), dist.all_reduce(Top5_err)
                
            # 更新进度条
            if runner.rank == 0:
                pbar.update(1)

        # 关闭进度条        
        if runner.rank == 0:
            pbar.close()

        epoch_Top1_err = Top1_err.avg
        epoch_Top5_err = Top5_err.avg
        
        if self.isSearch:  # 搜索阶段，网络架构变换
            # 还原网络
            if hasattr(runner.model, 'module'):
                runner.model.module.backbone = save_backbone
            else:
                runner.model.backbone = save_backbone

        if runner.rank == 0:
            eval_measures_str = '{:7.4f}, {:7.4f}'.format(epoch_Top1_err, epoch_Top5_err)
            self.logger.info('Computing Val result:')
            self.logger.info("{:>7}, {:>7}".format('Top1_err', 'Top5_err'))
            self.logger.info(eval_measures_str)

            # save weight of the best model 
            if self.work_dir != None and self.isSearch is False:
                cur_meter = epoch_Top1_err 
                if cur_meter < self.best_meter:
                    self.best_meter = cur_meter
                    runner.save_checkpoint(out_dir=self.work_dir, filename_tmpl='best.pth',save_optimizer=False)
                    # print(f'save the best weight {cur_loss}')

        # assert False,'test'

        return [epoch_Top1_err, epoch_Top5_err]



def Accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res