"""Base class for datasets that have keypoints and images.

@author: Qi Fan
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from .utils import process_batch


class BaseJointDataset(Dataset, ABC):
    """Base class for datasets that have keypoints and images.

    @cvar MAPPING: Mapping from the dataset's keypoints to the unipose format, an OrderedDict of {index_in_unipose: index_in_dataset}.
    @cvar EXTRA_TOKENS: Extra tokens that are present in the dataset, an OrderedDict of {name: index_in_dataset}. These will be added to the batch if they are present in the mask.
    @cvar SKELETON: Skeleton of the dataset, an OrderedDict of {name: list of indices_in_unipose that a limb consists of}.
    """

    MAPPING = OrderedDict({i: i for i in range(13)})

    EXTRA_TOKENS = OrderedDict()

    SKELETON = OrderedDict()

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    def _apply_mapping(self, keypoints: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Apply the mapping from any dataset to the unipose format. This is used to reorder the keypoints and mask to the unipose format, according to the C{MAPPING} and C{EXTRA_TOKENS} class variables.

        @param keypoints: The keypoints to reorder.
        @param mask: The mask to reorder.
        @return: The reordered keypoints (shape: (13, 2)), the reordered mask (shape: (13,)), the extra keypoints (shape: (N, 2)), and the names of the extra keypoints (list of strings of length N)
        """
        reordered_keypoints = torch.stack([keypoints[self.MAPPING[i]] for i in self.MAPPING])
        reordered_mask = torch.stack([mask[self.MAPPING[i]] for i in self.MAPPING])
        if any(mask[i] != 0 for _, i in self.EXTRA_TOKENS.items()):
            # If any of the extra tokens are present, we need to add them to the batch
            extra_token_keypoints = torch.stack([keypoints[i] for _, i in self.EXTRA_TOKENS.items() if mask[i] != 0])
            extra_token_names = [
                i for i in self.EXTRA_TOKENS if mask[self.EXTRA_TOKENS[i]] != 0
            ]  # Assuming dict is ordered...
            return reordered_keypoints, reordered_mask, extra_token_keypoints, extra_token_names
        else:
            return reordered_keypoints, reordered_mask, torch.zeros(0, 2), []

    def make_dataloader(self, image_size: int = 256, scale_factor: int = 4, *args, **kwargs) -> DataLoader:
        """Make a dataloader for the dataset.

        A unipose dataset returns a dictionary with the following keys: "images", "keypoint_images", "masks", "extra_keypoints", "extra_tokens".

        @param image_size: The size of the image to resize to.
        @param scale_factor: The factor to scale the image by.
        @param args: Other arguments to pass to the DataLoader.
        @param kwargs: Other keyword arguments to pass to the DataLoader.
        @return: A DataLoader for the dataset.
        """

        if "collate_fn" in kwargs:
            raise ValueError("collate_fn should not be set for BaseJointDataset, a default is provided")

        def collate_fn_for_unipose(batch):
            images = list([data["image"].unsqueeze(0) for data in batch])
            keypoints = list([data["unipose_keypoints"].unsqueeze(0) for data in batch])
            masks = list([data["unipose_mask"].unsqueeze(0) for data in batch])
            bounding_boxes = list([data["bounding_box"].unsqueeze(0) for data in batch])
            extra_keypoints = list([data["extra_keypoints"] for data in batch])
            extra_tokens = list([data["extra_tokens"] for data in batch])
            skeleton = torch.tensor(list([data["skeleton"] for data in batch]))
            images_resized = []
            keypoint_images = []
            for i in range(len(batch)):
                image, keypoint_image = process_batch(
                    images[i], bounding_boxes[i], keypoints[i], masks[i], image_size, scale_factor
                )
                images_resized.append(image)
                keypoint_images.append(keypoint_image)
            images_resized = torch.cat(images_resized, dim=0)
            keypoint_images = torch.cat(keypoint_images, dim=0)
            masks = torch.cat(masks, dim=0)
            return {
                "images": images_resized,
                "keypoint_images": keypoint_images,
                "masks": masks,
                "extra_keypoints": extra_keypoints,
                "extra_tokens": extra_tokens,
                "skeleton": skeleton,
            }

        return DataLoader(self, *args, **kwargs, collate_fn=collate_fn_for_unipose)


class ConcatJointDataset(ConcatDataset):
    """A dataset that concatenates multiple datasets.

    @param datasets: The datasets to concatenate.
    """

    def __init__(self, datasets):
        super().__init__(datasets)

    def make_dataloader(self, image_size: int = 256, scale_factor: int = 4, *args, **kwargs):
        """Make a dataloader for the concat-ed dataset.

        A unipose dataset returns a dictionary with the following keys: "images", "keypoint_images", "masks", "extra_keypoints", "extra_tokens".

        @param image_size: The size of the image to resize to.
        @param scale_factor: The factor to scale the image by.
        @param args: Other arguments to pass to the DataLoader.
        @param kwargs: Other keyword arguments to pass to the DataLoader.
        """
        if "collate_fn" in kwargs:
            raise ValueError("collate_fn should not be set for BaseJointDataset, a default is provided")

        def collate_fn_for_unipose(batch):
            images = list([data["image"].unsqueeze(0) for data in batch])
            keypoints = list([data["unipose_keypoints"].unsqueeze(0) for data in batch])
            masks = list([data["unipose_mask"].unsqueeze(0) for data in batch])
            bounding_boxes = list([data["bounding_box"].unsqueeze(0) for data in batch])
            extra_keypoints = list([data["extra_keypoints"] for data in batch])
            extra_tokens = list([data["extra_tokens"] for data in batch])
            skeleton = torch.tensor(list([data["skeleton"] for data in batch]))
            images_resized = []
            keypoint_images = []
            for i in range(len(batch)):
                image, keypoint_image = process_batch(
                    images[i], bounding_boxes[i], keypoints[i], masks[i], image_size, scale_factor
                )
                images_resized.append(image)
                keypoint_images.append(keypoint_image)
            images_resized = torch.cat(images_resized, dim=0)
            keypoint_images = torch.cat(keypoint_images, dim=0)
            masks = torch.cat(masks, dim=0)
            return {
                "images": images_resized,
                "keypoint_images": keypoint_images,
                "masks": masks,
                "extra_keypoints": extra_keypoints,
                "extra_tokens": extra_tokens,
                "skeleton": skeleton,
            }

        return DataLoader(self, *args, **kwargs, collate_fn=collate_fn_for_unipose)
