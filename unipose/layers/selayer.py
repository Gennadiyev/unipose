import torch
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        @param channels: The number of input channels.
        @param reduction: The reduction factor for the number of channels and the dimension of hidden layer is channels // reduction.
        """
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


if __name__ == "__main__":
    from torchinfo import summary

    model = SELayer(64)
    summary(model, input_size=(1, 64, 16, 16), depth=10)
