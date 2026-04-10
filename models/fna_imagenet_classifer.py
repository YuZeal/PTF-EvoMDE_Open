from mmdet.models.registry import HEADS
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.utils_newcrfs import silog_loss

# 需要在__init__.py中引入NewcrfsDecoder
@HEADS.register_module
class ClassificationHead(nn.Module):

    def __init__(self, in_channel=320, n_class=1000, **kwargs):
        super(ClassificationHead, self).__init__()
        self.in_channel = in_channel
        self.n_class = n_class

        self.last_conv = ConvBNReLU(in_channel=self.in_channel, out_channel=1280, k_size=1, stride=1, padding=0)
        self.drop_out = nn.Dropout2d(p=0.2)
        self.global_pool = nn.AvgPool2d(7)
        self.fc = FC(in_channels=1280, out_channels=n_class)
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, feats):
        x = feats[-1]
        x = self.last_conv(x)
        x = self.drop_out(x)
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def loss(self, output, target):
        criterion_smooth = CrossEntropyLabelSmooth(self.n_class, 0.1)
        loss_function = criterion_smooth.cuda()
        loss = loss_function(output, target)
        return dict(loss_deep=loss)  # 方便与源码统一格式

class ConvBNReLU(nn.Module):

    def __init__(self, in_channel, out_channel, k_size, stride=1, padding=0, groups=1,
                 has_bn=True, has_relu=True, gaussian_init=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=k_size,
                              stride=stride, padding=padding,
                              groups=groups, bias=True)
        if gaussian_init:
            nn.init.normal_(self.conv.weight.data, 0, 0.01)

        if has_bn:
            self.bn = nn.BatchNorm2d(out_channel)

        self.has_bn = has_bn
        self.has_relu = has_relu
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        nn.init.normal_(self.fc.weight.data, 0, 0.01)

    def forward(self, x):
        return self.fc(x)


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * \
            targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
