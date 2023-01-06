import pytest
import torch

from unipose.losses import GLES

@pytest.mark.skip(reason="Graph learning is out of scope for now")
def test_gles():
    _lambda = 2.0
    x = torch.randn((13, 32 * 32))
    model = GLES(lmbda=_lambda, num_features=32 * 32)
    graph = model(x)
    assert graph.shape == torch.Size([13, 13])
