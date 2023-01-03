import torch

from unipose.losses import GLES


def test_gles():
    _lambda = 2.0
    x = torch.randn((13, 32 * 32))
    model = GLES(lmbda=_lambda, num_features=32 * 32)
    graph = model(x)
    assert graph.shape == torch.Size([13, 13])
