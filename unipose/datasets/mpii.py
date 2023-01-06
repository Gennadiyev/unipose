import os
from typing import Union, Optional, List, Tuple, Dict, Any, Callable, Literal

import cv2
import torch
import orjson
from torchvision.transforms import ToTensor

from .base_joint_dataset import BaseJointDataset


class MPIIDataset(BaseJointDataset):

    MAPPING = {
        0: 8,  # Nose... Well, neck
        1: 13,  # Left shoulder
        2: 14,  # Left elbow
        3: 15,  # Left wrist
        4: 12,  # Right shoulder
        5: 11,  # Right elbow
        6: 10,  # Right wrist
        7: 3,  # Left hip
        8: 4,  # Left knee
        9: 5,  # Left ankle
        10: 2,  # Right hip
        11: 1,  # Right knee
        12: 0,  # Right ankle
    }

    EXTRA_TOKENS = {"pelvis": 6, "head_top": 9}

    def __init__(self, path: str, split: Literal["train", "val"] = "train"):
        if split not in ["train", "val"]:
            raise ValueError(f"Invalid split {split} (must be 'train' or 'val')")
        self.image_root = os.path.join(path, "images")
        self.split = split
        self.json_path = os.path.join(path, "annotations", f"{self.split}.json")
        self.load_labels()

    def load_labels(self):
        with open(self.json_path, "rb") as f:
            self.annotations = orjson.loads(f.read())

    def _apply_mapping(self, keypoints: torch.Tensor, mask: torch.Tensor):
        return super()._apply_mapping(keypoints, mask)

    def __getitem__(self, idx):

        annotation = self.annotations[idx]

        # Get image
        image_path = os.path.join(self.image_root, annotation["image"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract information
        keypoints = torch.tensor(annotation["joints"])  # shape: (16, 2)
        num_keypoints = len(keypoints)  # 16
        mask = torch.tensor(annotation["joints_vis"])  # shape: (16,)
        # Retrieve BBOX (x, y, w, h) based on the keypoints
        bbox = torch.tensor(
            [
                torch.min(keypoints[:, 0]),
                torch.min(keypoints[:, 1]),
                torch.max(keypoints[:, 0]) - torch.min(keypoints[:, 0]),
                torch.max(keypoints[:, 1]) - torch.min(keypoints[:, 1]),
            ]
        )

        # Convert to tensors
        image = ToTensor()(image)
        num_keypoints = map(torch.tensor, [num_keypoints])

        # Apply mapping
        keypoints_unipose, mask_unipose, extra_keypoints, extra_token_names = self._apply_mapping(keypoints, mask)

        return {
            "image": image,
            "bounding_box": bbox,
            "unipose_keypoints": keypoints_unipose,
            "unipose_mask": mask_unipose,
            "extra_keypoints": extra_keypoints,
            "extra_tokens": extra_token_names,
        }

    def __len__(self):
        return len(self.annotations)
