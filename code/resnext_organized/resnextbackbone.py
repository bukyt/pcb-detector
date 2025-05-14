
import torch
import torch.nn as nn

# Basic convolutional block with BatchNorm
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.bn(self.c(x))

# Bottleneck residual block with ResNeXt-style group conv
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, first=False, cardinatlity=32):
        super().__init__()
        self.C = cardinatlity
        self.downsample = stride == 2 or first
        res_channels = out_channels // 2
        self.c1 = ConvBlock(in_channels, res_channels, 1, 1, 0)
        self.c2 = ConvBlock(res_channels, res_channels, 3, stride, 1, self.C)
        self.c3 = ConvBlock(res_channels, out_channels, 1, 1, 0)

        self.relu = nn.ReLU()

        if self.downsample:
            self.p = ConvBlock(in_channels, out_channels, 1, stride, 0)

    def forward(self, x):
        f = self.relu(self.c1(x))
        f = self.relu(self.c2(f))
        f = self.c3(f)

        if self.downsample:
            x = self.p(x)

        h = self.relu(torch.add(f, x))
        return h


class ResNeXtBackbone(nn.Module):
    def __init__(self, config_name=50, in_channels=3, C=32):
        super().__init__()

        configurations = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
        no_blocks = configurations[config_name]
        out_features = [256, 512, 1024, 2048]

        self.conv1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.ModuleList([ResidualBlock(64, 256, 1, True, cardinatlity=C)])
        for i in range(len(out_features)):
            if i > 0:
                self.blocks.append(ResidualBlock(out_features[i-1], out_features[i], 2, cardinatlity=C))
            for _ in range(no_blocks[i] - 1):
                self.blocks.append(ResidualBlock(out_features[i], out_features[i], 1, cardinatlity=C))

        self.out_channels = 2048

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        for block in self.blocks:
            x = block(x)
        return x
