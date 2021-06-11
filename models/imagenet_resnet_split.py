import torch
import torch.nn as nn
import numpy as np

from .flat_linear import FlatLinear


__all__ = [
    'resnet18_mp', 'resnet34_mp', 'resnet50_mp', 'resnet101_mp', 'resnet152_mp',
    'resnet18_split', 'resnet34_split', 'resnet50_split', 'resnet101_split', 'resnet152_split',
    'wideresnet_101_2_mp', 'wideresnet_101_2_split',
    'resnext101_32x8d_mp', 'resnext101_32x8d_split',
]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layers = [
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]

        self.layers += self._make_layer(block, 64, layers[0])
        self.layers += self._make_layer(block, 128, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0])
        self.layers += self._make_layer(block, 256, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])
        self.layers += self._make_layer(block, 512, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2])

        self.layers += [
            nn.AdaptiveAvgPool2d((1, 1)),
            FlatLinear(512 * block.expansion, num_classes),
        ]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return layers


class ResNetBlock(nn.Module):

    def __init__(self, layers=[], zero_init_residual=False):
        super(ResNetBlock, self).__init__()
        self.layers = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.layers(x)
        return x


class ResNetMP(nn.Module):

    def __init__(self, num_classes=1000, depth=18, slices=[]):
        super(ResNetMP, self).__init__()
        if depth == 18:
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif depth == 34:
            block = BasicBlock
            layers = [3, 4, 6, 3]
        elif depth == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
        elif depth == 101:
            block = Bottleneck
            layers = [3, 4, 23, 3]
        elif depth == 152:
            block = Bottleneck
            layers = [3, 8, 36, 3]
        model = ResNet(block=block, layers=layers, num_classes=num_classes)

        print("=> splitting {} = sum({}) slices.".format(len(model.layers), slices))
        assert np.sum(slices) == len(model.layers)
        for _slice in slices:
            assert _slice > 0
        splits = np.cumsum(slices)
        for idx in range(len(slices)):
            if idx == 0:
                block = ResNetBlock(model.layers[:splits[idx]])
            else:
                block = ResNetBlock(model.layers[splits[idx-1]:splits[idx]])
            block.cuda(idx)
            setattr(self, 'layer' + str(idx), block)

        self.nblocks = len(slices)

    def forward(self, x):
        for idx in range(self.nblocks):
            layer = getattr(self, 'layer' + str(idx))
            x = layer(x.cuda(idx))
        return x


def resnet18_mp(slices=[]):
    return ResNetMP(depth=18, slices=slices)

def resnet34_mp(slices=[]):
    return ResNetMP(depth=34, slices=slices)

def resnet50_mp(slices=[]):
    return ResNetMP(depth=50, slices=slices)

def resnet101_mp(slices=[]):
    return ResNetMP(depth=101, slices=slices)

def resnet152_mp(slices=[]):
    return ResNetMP(depth=152, slices=slices)


def ResNetSplit(num_classes=1000, depth=18, slices=[]):
    if depth == 18:
        block = BasicBlock
        layers = [2, 2, 2, 2]
    elif depth == 34:
        block = BasicBlock
        layers = [3, 4, 6, 3]
    elif depth == 50:
        block = Bottleneck
        layers = [3, 4, 6, 3]
    elif depth == 101:
        block = Bottleneck
        layers = [3, 4, 23, 3]
    elif depth == 152:
        block = Bottleneck
        layers = [3, 8, 36, 3]
    model = ResNet(block=block, layers=layers, num_classes=num_classes)

    print("=> splitting {} = sum({}) slices.".format(len(model.layers), slices))
    assert np.sum(slices) == len(model.layers)
    for _slice in slices:
        assert _slice > 0
    splits = np.cumsum(slices)
    blocks = []
    for idx in range(len(slices)):
        if idx == 0:
            block = ResNetBlock(model.layers[:splits[idx]])
        else:
            block = ResNetBlock(model.layers[splits[idx-1]:splits[idx]])
        blocks.append(block)

    return blocks

def resnet18_split(slices=[]):
    return ResNetSplit(depth=18, slices=slices)

def resnet34_split(slices=[]):
    return ResNetSplit(depth=34, slices=slices)

def resnet50_split(slices=[]):
    return ResNetSplit(depth=50, slices=slices)

def resnet101_split(slices=[]):
    return ResNetSplit(depth=101, slices=slices)

def resnet152_split(slices=[]):
    return ResNetSplit(depth=152, slices=slices)


class WideResNetMP(nn.Module):

    def __init__(self, layers, slices=[]):
        super(WideResNetMP, self).__init__()
        model = ResNet(block=Bottleneck, layers=layers, width_per_group=64*2)

        print("=> splitting {} = sum({}) slices.".format(len(model.layers), slices))
        assert np.sum(slices) == len(model.layers)
        for _slice in slices:
            assert _slice > 0
        splits = np.cumsum(slices)
        for idx in range(len(slices)):
            if idx == 0:
                block = ResNetBlock(model.layers[:splits[idx]])
            else:
                block = ResNetBlock(model.layers[splits[idx-1]:splits[idx]])
            block.cuda(idx)
            setattr(self, 'layer' + str(idx), block)

        self.nblocks = len(slices)

    def forward(self, x):
        for idx in range(self.nblocks):
            layer = getattr(self, 'layer' + str(idx))
            x = layer(x.cuda(idx))
        return x


def WideResNetSplit(layers, slices):
    model = ResNet(block=Bottleneck, layers=layers, width_per_group=64*2)

    print("=> splitting {} = sum({}) slices.".format(len(model.layers), slices))
    assert np.sum(slices) == len(model.layers)
    for _slice in slices:
        assert _slice > 0
    splits = np.cumsum(slices)
    blocks = []
    for idx in range(len(slices)):
        if idx == 0:
            block = ResNetBlock(model.layers[:splits[idx]])
        else:
            block = ResNetBlock(model.layers[splits[idx-1]:splits[idx]])
        blocks.append(block)

    return blocks

def wideresnet_101_2_mp(slices=[]):
    return WideResNetMP([3, 4, 23, 3], slices=slices)

def wideresnet_101_2_split(slices=[]):
    return WideResNetSplit([3, 4, 23, 3], slices=slices)


class ResNeXtMP(nn.Module):

    def __init__(self, layers, groups=32, width_per_group=8, slices=[]):
        super(ResNeXtMP, self).__init__()
        model = ResNet(block=Bottleneck, layers=layers, groups=groups, width_per_group=width_per_group)

        print("=> splitting {} = sum({}) slices.".format(len(model.layers), slices))
        assert np.sum(slices) == len(model.layers)
        for _slice in slices:
            assert _slice > 0
        splits = np.cumsum(slices)
        for idx in range(len(slices)):
            if idx == 0:
                block = ResNetBlock(model.layers[:splits[idx]])
            else:
                block = ResNetBlock(model.layers[splits[idx-1]:splits[idx]])
            block.cuda(idx)
            setattr(self, 'layer' + str(idx), block)

        self.nblocks = len(slices)

    def forward(self, x):
        for idx in range(self.nblocks):
            layer = getattr(self, 'layer' + str(idx))
            x = layer(x.cuda(idx))
        return x


def ResNeXtSplit(layers, groups, width_per_group, slices):
    model = ResNet(block=Bottleneck, layers=layers, groups=groups, width_per_group=width_per_group)

    print("=> splitting {} = sum({}) slices.".format(len(model.layers), slices))
    assert np.sum(slices) == len(model.layers)
    for _slice in slices:
        assert _slice > 0
    splits = np.cumsum(slices)
    blocks = []
    for idx in range(len(slices)):
        if idx == 0:
            block = ResNetBlock(model.layers[:splits[idx]])
        else:
            block = ResNetBlock(model.layers[splits[idx-1]:splits[idx]])
        blocks.append(block)

    return blocks

def resnext101_32x8d_mp(slices=[]):
    return ResNeXtMP([3, 4, 23, 3], 32, 8, slices=slices)

def resnext101_32x8d_split(slices=[]):
    return ResNeXtSplit([3, 4, 23, 3], 32, 8, slices=slices)
