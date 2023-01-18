import os
from typing import Union, Optional, List, Tuple, Dict, Any, Callable, Literal

import cv2
import torch
from unipose.cocotools import COCO
from torchvision.transforms import ToTensor

from .base_joint_dataset import BaseJointDataset


class COCODataset(BaseJointDataset):

    MAPPING = {
        0: 0,   # Nose
        1: 5,   # Left shoulder
        2: 7,   # Left elbow
        3: 9,   # Left wrist
        4: 6,   # Right shoulder
        5: 8,   # Right elbow
        6: 10,  # Right wrist
        7: 11,  # Left hip
        8: 13,  # Left knee
        9: 15,  # Left ankle
        10: 12, # Right hip
        11: 14, # Right knee
        12: 16, # Right ankle
    }

    EXTRA_TOKENS = {"left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4}

    SKELETON = {
        "left_fore_limb": [1, 2, 3],
        "right_fore_limb": [4, 5, 6],
        "left_hind_limb": [7, 8, 9],
        "right_hind_limb": [10, 11, 12],
    }
    
    def __init__(self, path: str, split: Literal["train", "val"] = "train"):
        if split not in ["train", "val"]:
            raise ValueError(f"Invalid split {split} (must be 'train' or 'val')")
        self.path = path
        self.split = split
        self.json = os.path.join(path, "annotations", f"person_keypoints_{self.split}2017.json")
        self.coco = COCO(self.json)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _apply_mapping(self, keypoints: torch.Tensor, mask: torch.Tensor):
        return super()._apply_mapping(keypoints, mask)

    def __getitem__(self, idx):

        _id = self.ids[idx]

        # Get image
        image_path = self.coco.loadImgs(_id)[0]["file_name"]
        image = cv2.imread(os.path.join(self.path, f"{self.split}2017", image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anns = self.coco.loadAnns(self.coco.getAnnIds(_id))
        ann = anns[0]  # Although there may be one annotation per image, we only use the first one.
        # keys:
        # segementation, num_keypoints, area, iscrowd, keypoints, image_id, bbox, category_id, id

        # Extract information
        keypoints = ann["keypoints"]
        num_keypoints = ann["num_keypoints"]
        bbox = ann["bbox"]
        _data = torch.tensor(keypoints).reshape(-1, 3)
        keypoints = _data[:, :2]
        mask = _data[:, 2]
        mask[mask != 0] = 1  # 2 is visible, 1 is occluded, 0 is not labeled

        # Convert to tensors
        image = ToTensor()(image)
        bbox, num_keypoints = map(torch.tensor, (bbox, num_keypoints))

        # Apply mapping
        keypoints_unipose, mask_unipose, extra_keypoints, extra_token_names = self._apply_mapping(keypoints, mask)
        skeleton = [self.SKELETON[_] for _ in self.SKELETON]

        return {
            "image": image,
            "bounding_box": bbox,
            "unipose_keypoints": keypoints_unipose,
            "unipose_mask": mask_unipose,
            "extra_keypoints": extra_keypoints,
            "extra_tokens": extra_token_names,
            "skeleton": skeleton,
        }

    def __len__(self):
        return len(self.ids)
