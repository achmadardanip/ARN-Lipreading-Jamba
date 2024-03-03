import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from fightingcv_attention.attention.SEAttention import SEAttention

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se
        
        if(self.se):
            # self.se = SEBlock(planes)
            self.enhanced_se = EnhancedSEBlock(planes)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            

        # seblock
        if self.se:
            # out = self.se(out)
            out = self.enhanced_se(out)
        
        out = out + residual
        out = self.relu(out)

        return out

# original seblock
class SEBlock(nn.Module):
    def __init__(self, planes, reduction=16):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = conv1x1(planes, planes // reduction)
        self.conv2 = conv1x1(planes // reduction, planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        w = self.gap(x)
        w = self.conv1(w)
        w = self.relu(w)
        w = self.conv2(w).sigmoid()
        return x * w

# enhanced seblock
class EnhancedSEBlock(nn.Module):
    def __init__(self, planes, reduction=16):
        super(EnhancedSEBlock, self).__init__()
        self.channel_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes, planes // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes // reduction, planes, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_se = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel-wise attention
        chn_se = self.channel_se(x)
        chn_se = x * chn_se

        # Spatial attention
        spa_se = self.spatial_se(x)
        spa_se = x * spa_se

        # Combining both attentions
        out = chn_se + spa_se

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, se=False):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.se = se
        if self.se:
            # self.se_block = SEBlock(planes * self.expansion)
            self.se_block = EnhancedSEBlock(planes * self.expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.se:
            out = self.se_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, se=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.se = se
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.bn = nn.BatchNorm1d(512*block.expansion)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        return x        

class VideoCNN(nn.Module):
    def __init__(self, se=False):
        super(VideoCNN, self).__init__()
        
        # frontend3D
        self.frontend3D = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
                )
        # resnet
        self.resnet = ResNet(BasicBlock, [3, 4, 6, 3], se=se)
        # self.resnet = ResNet(Bottleneck, [3, 8, 36, 3], se=se)
        self.dropout = nn.Dropout(p=0.5)
        
        # initialize
        self._initialize_weights()
    
    def visual_frontend_forward(self, x):
        x = x.transpose(1, 2)
        x = self.frontend3D(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))

        x = self.resnet(x)
        return x        
    
    def forward(self, x):
        b, t = x.size()[:2]
        x = self.visual_frontend_forward(x)
        
        x = x.view(b, -1, 512)       
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()