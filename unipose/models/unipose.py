import torch
import torch.nn as nn
import torchvision.models as models

from unipose.layers import DUC, SEResNeXt


class UniPose(nn.Module):
    """UniPose model.
    """
    def __init__(
        self, keypoint_count, channels=64, groups=32, reduction=16, resnet_layers=[2, 2, 2, 2], duc_layers=[4, 2, 1]
    ):
        """
        The UniPose model, which contains a SEResNeXt backbone to get the feature map and several DUC layers to upsample.

        @param keypoint_count: The number of keypoints, the output channels of the last layer.
        @param channels The number of hidden channels for the SEResNeXt.
        @param groups: The number of groups for the grouped convolution for the SEResNeXt.
        @param reduction: The reduction factor for the number of channels for the SEResNeXt.
        @param resnet_layers: The number of layers for each stage of the SEResNeXt.
        @param duc_layers: The number of layers for each stage of the DUC.
        """
        super(UniPose, self).__init__()
        self.backbone = SEResNeXt(
            channels=channels,
            groups=groups,
            reduction=reduction,
            layers=resnet_layers,
        )

        if resnet_layers == [2, 2, 2, 2]:
            pretrained = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif resnet_layers == [3, 4, 6, 3]:
            pretrained = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif resnet_layers == [3, 4, 23, 3]:
            pretrained = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif resnet_layers == [3, 8, 36, 3]:
            pretrained = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif resnet_layers == [3, 24, 36, 3]:
            pretrained = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            raise ValueError("Invalid resnet_layers: {}".format(resnet_layers))
        backbone_state = self.backbone.state_dict()
        pretrained_state = {
            k: v
            for k, v in pretrained.state_dict().items()
            if k in self.backbone.state_dict() and v.size() == self.backbone.state_dict()[k].size()
        }
        backbone_state.update(pretrained_state)
        self.backbone.load_state_dict(backbone_state)

        self.duc1 = DUC(32 * channels, 16 * channels, upscale_factor=2, layer_number=duc_layers[0])
        self.duc2 = DUC(16 * channels, 8 * channels, upscale_factor=2, layer_number=duc_layers[1])
        self.duc3 = DUC(8 * channels, 4 * channels, upscale_factor=2, layer_number=duc_layers[2])

        self.conv_out = nn.Conv2d(4 * channels, keypoint_count, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.duc1(x)
        x = self.duc2(x)
        x = self.duc3(x)
        x = self.conv_out(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    model = UniPose(17, channels=64, groups=32, reduction=16, resnet_layers=[3, 8, 36, 3], duc_layers=[4, 2, 1])
    summary(model, input_size=(1, 3, 256, 256), depth=10)
