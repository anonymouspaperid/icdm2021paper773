"""
    PreResNet model definition
    ported from https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/preresnet.py
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import math
import numpy as np
from .flat_linear import FlatLinear

__all__ = ['preresnet164_mp', 'preresnet1001_mp', 'preresnet164_split', 'preresnet1001_split']


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class PreResNet(nn.Module):

    def __init__(self, num_classes=10, depth=110):
        super(PreResNet, self).__init__()
        if depth >= 44:
            assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
            n = (depth - 2) // 6
            block = BasicBlock


        self.inplanes = 16

        self.layers = [nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)]
        self.layers += self._make_layer(block, 16, n)
        self.layers += self._make_layer(block, 32, n, stride=2)
        self.layers += self._make_layer(block, 64, n, stride=2)
        self.layers += [
            nn.BatchNorm2d(64 * block.expansion),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(8),
            FlatLinear(64 * block.expansion, num_classes),
        ]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers


class PreResNetBlock(nn.Module):

    def __init__(self, layers=[]):
        super(PreResNetBlock, self).__init__()
        self.layers = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.layers(x)
        return x


class PreResNetMP(nn.Module):

    def __init__(self, num_classes=10, depth=110, slices=[]):
        super(PreResNetMP, self).__init__()
        model = PreResNet(num_classes=num_classes, depth=depth)
        print("=> splitting {} = sum({}) slices.".format(len(model.layers), slices))
        assert np.sum(slices) == len(model.layers)
        for _slice in slices:
            assert _slice > 0
        splits = np.cumsum(slices)
        for idx in range(len(slices)):
            if idx == 0:
                block = PreResNetBlock(model.layers[:splits[idx]])
            else:
                block = PreResNetBlock(model.layers[splits[idx-1]:splits[idx]])
            block.cuda(idx)
            setattr(self, 'layer' + str(idx), block)

        self.nblocks = len(slices)

    def forward(self, x):
        for idx in range(self.nblocks):
            layer = getattr(self, 'layer' + str(idx))
            x = layer(x.cuda(idx))
        return x


def preresnet164_mp(num_classes=10, slices=[]):
    return PreResNetMP(num_classes, depth=164, slices=slices)

def preresnet1001_mp(num_classes=10, slices=[]):
    return PreResNetMP(num_classes, depth=1001, slices=slices)


def PreResNetSplit(num_classes=10, depth=110, slices=[]):
    blocks = []

    model = PreResNet(num_classes=num_classes, depth=depth)
    print("=> splitting {} = sum({}) slices.".format(len(model.layers), slices))
    assert np.sum(slices) == len(model.layers)
    for _slice in slices:
        assert _slice > 0
    splits = np.cumsum(slices)
    for idx in range(len(slices)):
        if idx == 0:
            block = PreResNetBlock(model.layers[:splits[idx]])
        else:
            block = PreResNetBlock(model.layers[splits[idx-1]:splits[idx]])
        blocks.append(block)

    return blocks


def preresnet164_split(num_classes=10, slices=[]):
    return PreResNetSplit(num_classes, depth=164, slices=slices)

def preresnet1001_split(num_classes=10, slices=[]):
    return PreResNetSplit(num_classes, depth=1001, slices=slices)
