import torch
import torch.nn as nn
import numpy as np

from .flat_linear import FlatLinear


__all__ = [
    'vgg19_mp', 'vgg19_split',
]


class VGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(VGG, self).__init__()
        self.layers = self.make_layers(cfg, batch_norm)
        self.layers += [
            nn.AdaptiveAvgPool2d((7, 7)),
            FlatLinear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        ]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return layers


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGBlock(nn.Module):

    def __init__(self, layers=[]):
        super(VGGBlock, self).__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class VGGMP(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1000, slices=[]):
        super(VGGMP, self).__init__()
        model = VGG(cfg, batch_norm, num_classes)

        print("=> splitting {} = sum({}) slices.".format(len(model.layers), slices))
        assert np.sum(slices) == len(model.layers)
        for _slice in slices:
            assert _slice > 0
        splits = np.cumsum(slices)
        for idx in range(len(slices)):
            if idx == 0:
                block = VGGBlock(model.layers[:splits[idx]])
            else:
                block = VGGBlock(model.layers[splits[idx-1]:splits[idx]])
            block.cuda(idx)
            setattr(self, 'layer' + str(idx), block)

        self.nblocks = len(slices)

    def forward(self, x):
        for idx in range(self.nblocks):
            layer = getattr(self, 'layer' + str(idx))
            x = layer(x.cuda(idx))
        return x


def vgg19_mp(slices=[]):
    return VGGMP(cfgs['E'], batch_norm=False, slices=slices)


def VGGSplit(cfg, batch_norm=False, num_classes=1000, slices=[]):
    model = VGG(cfg, batch_norm, num_classes)

    print("=> splitting {} = sum({}) slices.".format(len(model.layers), slices))
    assert np.sum(slices) == len(model.layers)
    for _slice in slices:
        assert _slice > 0
    splits = np.cumsum(slices)
    blocks = []
    for idx in range(len(slices)):
        if idx == 0:
            block = VGGBlock(model.layers[:splits[idx]])
        else:
            block = VGGBlock(model.layers[splits[idx-1]:splits[idx]])
        blocks.append(block)

    return blocks

def vgg19_split(slices=[]):
    return VGGSplit(cfgs['E'], batch_norm=False, slices=slices)
