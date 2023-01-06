import pytest
import torch
from torchvision.transforms import ToTensor

from unipose.datasets.utils import enlarge_bounding_box, process_batch

def test_enlarge_bbox_int():
    base_bbox = torch.tensor([[400, 200, 120, 160]])
    square_bbox = enlarge_bounding_box(base_bbox)
    target_bbox = torch.tensor([[380, 200, 160, 160]], dtype=torch.float32)
    assert torch.allclose(square_bbox, target_bbox)

def test_enlarge_bbox_float():
    base_bbox = torch.tensor([[400.0, 200.0, 130.0, 160.0]])
    square_bbox = enlarge_bounding_box(base_bbox)
    target_bbox = torch.tensor([[385.0, 200.0, 160.0, 160.0]], dtype=torch.float32)
    assert torch.allclose(square_bbox, target_bbox)

def test_enlarge_bbox_batch():
    base_bbox = torch.randint(0, 1000, (10, 4), dtype=torch.float32)
    square_bbox = enlarge_bounding_box(base_bbox)
    assert torch.allclose(square_bbox[:, 2], square_bbox[:, 3])

