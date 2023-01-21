import pytest
import torch

from unipose.datasets import COCODataset, AnimalKingdomDataset, MPIIDataset, AP10KDataset


@pytest.mark.contains_absolute_path
def test_coco():
    dataset = COCODataset("datasets/coco", split="train")
    assert len(dataset) == 64115
    data = dataset[0]
    assert data["unipose_keypoints"].shape == torch.Size([13, 2])
    assert data["bounding_box"].shape == torch.Size([4])
    assert len(data["extra_tokens"]) <= 4
    assert data["image"].shape[0] == 3


@pytest.mark.contains_absolute_path
def test_animal_kingdom():
    dataset = AnimalKingdomDataset(
        "datasets/animal_kingdom", sub_category="ak_P3_amphibian", split="train"
    )
    assert len(dataset) == 5188
    data = dataset[0]
    assert data["unipose_keypoints"].shape == torch.Size([13, 2])
    assert data["bounding_box"].shape == torch.Size([4])
    assert data["image"].shape[0] == 3


@pytest.mark.contains_absolute_path
def test_mpii():
    dataset = MPIIDataset("datasets/mpii", split="train")
    assert len(dataset) == 22246
    data = dataset[0]
    assert data["unipose_keypoints"].shape == torch.Size([13, 2])
    assert data["bounding_box"].shape == torch.Size([4])
    assert data["image"].shape[0] == 3


@pytest.mark.contains_absolute_path
def test_ap10k():
    dataset = AP10KDataset("datasets/ap10k", split="val")
    assert len(dataset) == 995
    data = dataset[0]
    assert data["unipose_keypoints"].shape == torch.Size([13, 2])
    assert data["bounding_box"].shape == torch.Size([4])
    assert data["image"].shape[0] == 3