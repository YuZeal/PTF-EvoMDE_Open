import copy
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import kaiming_init
from mmdet.models.registry import BACKBONES
from tools.apis.param_remap_search import remap_for_archadapt
from tools.multadds_count import comp_multadds_fw

from .operations import OPS
from collections import defaultdict
import time, json
import numpy as np
layer_idx = 0

class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, stride, dilation, primitives, affine=False, track=False):
        super(MixedOp, self).__init__()

        self._ops = nn.ModuleList()  # 用于存储所有操作的模块列表
        self.primitives = primitives  # 操作类型列表
        # 遍历操作类型，创建操作实例并添加到模块列表中
        for primitive in primitives:
            op = OPS[primitive](C_in, C_out, stride, dilation, affine, track)
            self._ops.add_module('{}'.format(primitive), op)
    
    def set_requires_grad(self, selected_branch_index):  # yzh new add 2024_12_13
        """根据branch_indices设置未选路径的requires_grad为False"""
        # 冻结所有操作的权重
        for op in self._ops:
            for param in op.parameters():
                param.requires_grad = False
        
        # 只开启选中的操作的requires_grad
        selected_op = getattr(self._ops, self.primitives[selected_branch_index])
        for param in selected_op.parameters():
            param.requires_grad = True

    def forward(self, x, weights, branch_indices, mixed_sub_obj):
        # 确保branch_indices只有一个操作被选择
        assert len(branch_indices) == 1, "Each node should only select one branch"
        # 选择对应的操作
        selected_branch_index = branch_indices[0]
        # 设置未选路径的requires_grad为False
        self.set_requires_grad(selected_branch_index)
        # 计算输出
        return getattr(self._ops, self.primitives[selected_branch_index])(x), mixed_sub_obj[selected_branch_index]
    
    # def forward(self, x, weights, branch_indices, mixed_sub_obj):
    #     # 确保 branch_indices 只选择了一个分支
    #     assert len(branch_indices) == 1, "Each node should only select one branch"

    #     results = defaultdict(lambda: defaultdict(dict))
    #     global layer_idx
    #     org_input = x
    #     op_num = 6          # 与 cfg 保持一致
    #     WARMUP  = 10        # 预热次数
    #     REPEAT  = 100       # 计时次数

    #     with torch.no_grad():
    #         for op_id in range(op_num):
    #             op = getattr(self._ops, self.primitives[op_id])

    #             # ---------- 预热 ----------
    #             for _ in range(WARMUP):
    #                 out = op(org_input)

    #             # ---------- 正式计时 ----------
    #             times = []
    #             for _ in range(REPEAT):
    #                 torch.cuda.synchronize()
    #                 start_event = torch.cuda.Event(enable_timing=True)
    #                 end_event   = torch.cuda.Event(enable_timing=True)

    #                 start_event.record()
    #                 _ = op(org_input)
    #                 end_event.record()

    #                 # 等待事件完成，确保读到准确时间
    #                 torch.cuda.synchronize()

    #                 elapsed_ms = start_event.elapsed_time(end_event)   # 单位: 毫秒
    #                 times.append(elapsed_ms * 1e-3)                    # 转为秒

    #             mean_time = float(np.mean(times))
    #             var_time  = float(np.var(times))

    #             print(f"Layer {layer_idx}, Branch {op_id}, "
    #                 f"Mean: {mean_time:.6f} s, Variance: {var_time:.6e} s²")

    #             results[f"layer_{layer_idx}"][f"branch_{op_id}"] = {
    #                 "mean": mean_time,
    #                 "variance": var_time
    #             }

    #             # 记录层号
    #             if op_id == op_num - 1:
    #                 layer_idx += 1

    #             append_to_json_file(results, "Latency_table_for_kitti.json")

    #     # 与原逻辑保持一致：返回最后一次 op 的输出及对应 mixed_sub_obj
    #     return out, mixed_sub_obj[op_num - 1]


class Block(nn.Module):
    def __init__(self, C_in, C_out, stride, num_layer, search_params, bn_params=[True, True]):
        super(Block, self).__init__()
        self.layers = nn.ModuleList()  # 用于存储多个 MixedOp 实例的模块列表
        # 遍历每一层，创建并添加 MixedOp 实例
        for inner_idx in range(num_layer):
            if inner_idx == 0:
                # 第一层使用 primitives_reduce 操作类型
                self.layers.append(MixedOp(C_in, C_out, stride, 1, 
                                            search_params.primitives_reduce, 
                                            affine=bn_params[0], track=bn_params[1]))
            else:
                # 其余层使用 primitives_normal 操作类型
                self.layers.append(MixedOp(C_out, C_out, 1, 1, 
                                            search_params.primitives_normal, 
                                            affine=bn_params[0], track=bn_params[1]))

    def forward(self, x, weights_normal, weights_reduce, branch_index, block_sub_obj):  # key
        weights = []
        # 将 weights_reduce 和 weights_normal 组合成一个列表
        for weight in weights_reduce:
            weights.append(weight)
        for weight in weights_normal:
            weights.append(weight)

        count_sub_obj = []  # 用于存储每一层的子目标值

        # 遍历每一层、权重、分支索引和子目标
        for layer, weight, branch_idx, layer_sub_obj in zip(
                    self.layers, weights, branch_index, block_sub_obj):
            # 调用层的前向传播方法
            # print('x:', x.shape)
            x, sub_obj = layer(x, weight, branch_idx, layer_sub_obj)
            # 记录子目标值
            count_sub_obj.append(sub_obj)
        # 返回最终输出和子目标值的总和
        return x, sum(count_sub_obj)


class BaseBackbone(nn.Module):
    def __init__(self, search_params, output_indices=[2, 3, 5, 7]):
        super(BaseBackbone, self).__init__()
        self.output_indices = output_indices
        self.search_params = search_params
        self.net_scale = search_params.net_scale
        self.num_layers = self.net_scale.num_layers
        self.logger = logging.getLogger()
        
        self.primitives_reduce = search_params.primitives_reduce
        self.primitives_normal = search_params.primitives_normal
        self._initialize_alphas()

        self.blocks = nn.ModuleList()
        # 第一卷积块
        self.blocks.append(nn.Sequential(
                        nn.Conv2d(3, self.net_scale.chs[0], 3, 
                                self.net_scale.strides[0], 
                                padding=1,bias=False),
                        nn.BatchNorm2d(self.net_scale.chs[0], 
                                    affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True)))
        # 第二卷积块
        self.blocks.append(OPS['k3_e1'](
                                self.net_scale.chs[0], 
                                self.net_scale.chs[1], 
                                self.net_scale.strides[1], 
                                dilation=1, affine=True, track=True))

        C_in = self.net_scale.chs[1]
        # 其他卷积块
        for n_idx in range(len(self.num_layers)):
            C_out = self.net_scale.chs[n_idx+2]
            stride = self.net_scale.strides[n_idx+2]
            self.blocks.append(Block(C_in, C_out, stride, 
                                    self.num_layers[n_idx], 
                                    search_params,
                                    [search_params.affine, search_params.track]))
            C_in = C_out


    def _initialize_alphas(self):
        num_ops_normal = len(self.search_params.primitives_normal)
        num_ops_reduce = len(self.search_params.primitives_reduce)
        self.alphas_normal = nn.ParameterList()
        self.alphas_reduce = nn.ParameterList()
        for num_layer in self.num_layers:
            self.alphas_reduce.append(nn.Parameter(
                1e-3 * torch.randn(1, num_ops_reduce).cuda(), 
                requires_grad=True))
            if num_layer-1 == 0:  # yzh
                self.alphas_normal.append(nn.Parameter(
                    1e-3 * torch.randn(0).cuda(), 
                    requires_grad=True))
            else:
                self.alphas_normal.append(nn.Parameter(
                    1e-3 * torch.randn(num_layer-1, num_ops_normal).cuda(), 
                    requires_grad=True))


    @property
    def arch_parameters(self):
        arch_params = nn.ParameterList()
        arch_params.extend(self.alphas_reduce)
        arch_params.extend(self.alphas_normal)
        return arch_params


    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None and m.bias is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        
        if pretrained.use_load:
            model_dict = remap_for_archadapt(pretrained.load_path, self.state_dict(), pretrained.seed_num_layers)
            self.load_state_dict(model_dict)
        elif hasattr(pretrained, 'load_path') and pretrained.load_path is not None:
            checkpoint = torch.load(pretrained.load_path)
            checkpoint = checkpoint['state_dict']
            new_state_dict = {k.replace("backbone.", ""): v for k, v in checkpoint.items() if not k.startswith("bbox_head.")}
            self.load_state_dict(new_state_dict)
            logging.info('Loading pretrained weights on imagenet-1k finished!')


    def get_sub_obj_list(self, sub_obj_cfg, data_shape):
        if sub_obj_cfg.type=='flops':
            flops_list_sorted = self.get_flops_list(data_shape)
            self.sub_obj_list = copy.deepcopy(flops_list_sorted)
        else:
            assert False,'sub_obj_cfg.type is error'

    # 计算每个操作的 FLOPs 并返回包含所有块 FLOPs 的列表
    def get_flops_list(self, input_shape):
        data = torch.randn(input_shape)
        block_flops = []
        data = self.blocks[0](data)
        data = self.blocks[1](data)

        for block in self.blocks[2:]:
            layer_flops = []
            if hasattr(block, 'layers'):
                for layer in block.layers:
                    op_flops = []
                    for op in layer._ops:
                        flops, op_data = comp_multadds_fw(op, data, 'B', 'cpu')
                        op_flops.append(flops)
                    data = op_data
                    layer_flops.append(op_flops)
                block_flops.append(layer_flops)
        return block_flops


    def train(self, mode=True):
        super(BaseBackbone, self).train(mode)
        # if mode and self.freeze_bn:
        #     for m in self.modules():
        #         # trick: eval have effect on BatchNorm only
        #         if isinstance(m, BatchNorm):
        #             m.eval()


def append_to_json_file(new_data, file_path):
    if os.path.exists(file_path):
        # 如果文件存在，读取原内容
        with open(file_path, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}  # 文件为空时处理
    else:
        existing_data = {}

    # 合并新数据
    for key, value in new_data.items():
        if key in existing_data:
            existing_data[key].update(value)  # 合并子层级
        else:
            existing_data[key] = value

    # 写回文件
    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=4)