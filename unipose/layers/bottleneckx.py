import torch
import torch.nn as nn

from .selayer import SELayer


class BottleneckX(nn.Module):
    def __init__(self, in_channels, channels, stride=1, groups=32, downsample=None, reduction=None):
        """
        @param in_channels: The number of input channels.
        @param channels: The number of hidden channels, and the output channels is 4 times of this.
        @param stride: The stride for the grouped convolutions.
        @param groups: The number of groups for grouped convolutions.
        @param downsample: The downsample layer.
        @param reduction: The reduction ratio for the SE layer. If None, no SE layer is used.
        """
        super(BottleneckX, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=0.1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.1)
        self.conv3 = nn.Conv2d(channels, channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        if reduction is not None:
            self.se = SELayer(channels * 4, reduction)

        self.stride = stride
        self.downsample = downsample
        self.reduction = reduction

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

        if self.reduction is not None:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


if __name__ == "__main__":
    from torchinfo import summary

    downsample = nn.Sequential(
        nn.Conv2d(3, 256, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(256, momentum=0.1)
    )
    model = BottleneckX(3, 64, stride=2, groups=16, downsample=downsample, reduction=16)
    summary(model, input_size=(1, 3, 256, 256), depth=10)
