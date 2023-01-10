from math import sqrt

import torch
import torch.nn as nn

from unipose.losses.graph_learning import GLES


class UniSkel(nn.Module):
    """
    Model for generating skeleton based on joint features.
    """
    def __init__(self, num_in_channels=3, num_out_channels=8, heatmap_shape=(64,64),
                 num_gles_feature=None, expect_edges=None, lmbda=.5, max_iter=5, mode="undirected"):

        super(UniSkel, self).__init__()

        num_in = (num_out_channels + 1) * heatmap_shape[0] * heatmap_shape[1]
        num_gles = num_gles_feature if num_gles_feature is not None else int(sqrt(num_in)) + 1
        
        self.backbone = nn.Sequential(
            nn.Conv2d(num_in_channels, 32, kernel_size=3, stride=1, padding=1),  # 3x256x256 -> 32x256x256
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),               # 32x256x256 -> 32x256x256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                               # 32x256x256 -> 32x128x128
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),               # 32x128x128 -> 32x128x128
            nn.ReLU(),
            nn.Conv2d(32, num_out_channels, kernel_size=3, stride=1, padding=1), # 32x128x128 -> 64x128x128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                               # 64x128x128 -> 64x64x64
        )

        self.flat = nn.Flatten()
        self.fc = nn.Linear(num_in, num_gles)
        self.gles = GLES(expect_edges=expect_edges, lmbda=lmbda, max_iter=max_iter, mode=mode)

    def forward(self, x, heatmap, mode="train"):
        assert (mode == "train") or (mode == "predict"), "Mode should be 'train' or 'predict', but get {}".format(mode)
        assert heatmap.dim() == 4, "Heatmap should be 4D tensor, (batch_size, N, H, W), but get {}.".format(heatmap.shape)
        input = self.backbone(x)
        output = None
        for i in range(heatmap.shape[0]):
            z = None
            for j in range(heatmap.shape[1]):
                feature = self.flat(input[i] * heatmap[i,j])
                feature = torch.cat((feature.view(-1), heatmap[i,j].view(-1)), dim=0)
                if z is None:
                    z = feature.unsqueeze(dim=0)
                else:
                    z = torch.cat((z, feature.unsqueeze(dim=0)), dim=0)

            if output is None:
                output = z.unsqueeze(dim=0)
            else:
                output = torch.cat((output, z.unsqueeze(dim=0)), dim=0)

        assert output.dim() == 3, "Output should be 3D tensor, (batch_size, N, F),  but get {}.".format(output.shape)
        output = self.fc(output)

        z = None
        for i in range(output.shape[0]):
            if z is None:
                z = self.gles(output[i]).unsqueeze(dim=0)
            else:
                z = torch.cat((z, self.gles(output[i]).unsqueeze(dim=0)), dim=0)

        if mode == "train":
            return output, z
        return z


