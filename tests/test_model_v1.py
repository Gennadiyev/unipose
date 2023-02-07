from unipose.models import UniPose
from unipose.layers import SEResNeXt, DUC

def test_model_structure():
    from torchinfo import summary
    model = UniPose(17, channels=64, groups=32, reduction=16, resnet_layers=[3, 8, 36, 3], duc_layers=[4, 2, 1])
    summary(model, input_size=(1, 3, 256, 256), depth=10)

def test_layers():
    from torchinfo import summary
    model_duc = DUC(64, 256, 2, 3)
    summary(model_duc, input_size=(1, 64, 4, 4))
    model_seresnext = SEResNeXt(channels=64, groups=32, reduction=16, layers=[2, 2, 2, 2])
    summary(model_seresnext, input_size=(1, 3, 256, 256), depth=10)

