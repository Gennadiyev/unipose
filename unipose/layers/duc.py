import torch
import torch.nn as nn


class DUC(nn.Module):
    def __init__(self, in_channels, channels, upscale_factor=2, layer_number=2):
        super(DUC, self).__init__()
        layers = []
        layers.append(nn.PixelShuffle(upscale_factor))
        in_channels //= 4
        for i in range(1, layer_number):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(in_channels, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(channels, momentum=0.1))
        layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    from torchinfo import summary

    model = DUC(64, 256, 2, 3)
    summary(model, input_size=(1, 64, 4, 4))
