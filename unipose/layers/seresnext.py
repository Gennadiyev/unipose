import torch
import torch.nn as nn

from .bottleneckx import BottleneckX


class SEResNeXt(nn.Module):
    def __init__(self, channels=64, groups=32, reduction=16, layers=[2, 2, 2, 2]):
        super(SEResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(3, channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.channels = channels
        self.groups = groups
        self.reduction = reduction

        self.layer1 = self._make_layer(channels, layers[0], stride=1)
        self.layer2 = self._make_layer(channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(channels * 8, layers[3], stride=2)

    def _make_layer(self, channels, layer_number, stride=1):
        downsample = None
        if stride != 1 or self.channels != channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.channels, channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * 4, momentum=0.1)
            )
        
        layers = []
        if downsample is not None:
            layers.append(
                BottleneckX(
                    in_channels=self.channels,
                    channels=channels,
                    stride=stride,
                    groups=self.groups,
                    downsample=downsample,
                    reduction=self.reduction
                )
            )
        else:
            layers.append(
                BottleneckX(
                    in_channels=self.channels,
                    channels=channels,
                    stride=stride,
                    groups=self.groups,
                    downsample=downsample,
                    reduction=None
                )
            )
        self.channels = channels * 4
        for i in range(1, layer_number):
            layers.append(
                BottleneckX(
                    in_channels=self.channels,
                    channels=channels,
                    stride=1,
                    groups=self.groups,
                    downsample=None,
                    reduction=None
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary
    model = SEResNeXt(channels=64, groups=32, reduction=16, layers=[2, 2, 2, 2])
    summary(model, input_size=(1, 3, 256, 256), depth=10)
