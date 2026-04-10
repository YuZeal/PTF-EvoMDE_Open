import torch.nn as nn

from mmcv.cnn import kaiming_init
from mmdet.models.registry import BACKBONES

from .derive_blocks import derive_blocks
from tools.apis.param_remap import remap_for_paramadapt


@BACKBONES.register_module
class FNA_Retinanet(nn.Module):  # for retrain
    def __init__(self, net_config, output_indices=[2, 3, 5, 7]):
        super(FNA_Retinanet, self).__init__()
        self.blocks, _ = derive_blocks(net_config)
        self.output_indices = output_indices

    def forward(self, x, stat=None):
        outs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.output_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            assert len(outs) == 4
            
            ### 可视化中间特征图
            # import numpy as np
            # import matplotlib.pyplot as plt
            # feat = outs[3]
            # print(feat.shape)
            # first_channel = feat[0, 4, :, :].cpu().detach().numpy()  # feat[0, 0, :, :] 选择第一个通道
            # first_channel = (first_channel - np.min(first_channel)) / (np.max(first_channel) - np.min(first_channel)) * 255
            # # 使用 matplotlib 可视化并保存图像
            # plt.imshow(first_channel, cmap='viridis')
            # plt.axis('off')  # 不显示坐标轴
            # plt.savefig('0000000006_mid_feats_c4.png', bbox_inches='tight', pad_inches=0)
            # assert False

            return tuple(outs)

    def train(self, mode=True):
        super(FNA_Retinanet, self).train(mode)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained is not None and pretrained.use_load:
            model_dict = remap_for_paramadapt(pretrained.load_path, self.state_dict(), 
                                                pretrained.seed_num_layers)
            self.load_state_dict(model_dict)
