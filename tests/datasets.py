import pytest
import torch

from unipose.datasets import COCODataset, AnimalKingdomDataset, MPIIDataset


@pytest.mark.slow
@pytest.mark.contains_absolute_path
def test_coco():
    dataset = COCODataset("/home/dl2022/d3d/unipose/datasets/coco", split="train")
    assert len(dataset) == 64115
    data = dataset[0]
    assert data["unipose_keypoints"].shape == torch.Size([13, 2])
    assert data["bounding_box"].shape == torch.Size([4])
    assert len(data["extra_tokens"]) <= 4
    assert data["image"].shape[0] == 3


@pytest.mark.slow
@pytest.mark.contains_absolute_path
def test_animal_kingdom():
    dataset = AnimalKingdomDataset(
        "/home/dl2022/d3d/unipose/datasets/animal_kingdom", sub_category="ak_P3_amphibian", split="train"
    )
    assert len(dataset) == 5188
    data = dataset[0]
    assert data["unipose_keypoints"].shape == torch.Size([13, 2])
    assert data["bounding_box"].shape == torch.Size([4])
    assert data["image"].shape[0] == 3


@pytest.mark.contains_absolute_path
def test_mpii():
    dataset = MPIIDataset("/home/dl2022/d3d/unipose/datasets/mpii", split="train")
    assert len(dataset) == 22246
    data = dataset[0]
    assert data["unipose_keypoints"].shape == torch.Size([13, 2])
    assert data["bounding_box"].shape == torch.Size([4])
    assert data["image"].shape[0] == 3
