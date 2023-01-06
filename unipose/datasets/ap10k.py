import os
from typing import Literal

import cv2
import torch
from pycocotools.coco import COCO
from torchvision.transforms import ToTensor

from .base_joint_dataset import BaseJointDataset


class AP10KDataset(BaseJointDataset):

    MAPPING = {
        0: 2,   # Nose
        1: 5,   # Left shoulder
        2: 6,   # Left elbow
        3: 7,   # Left wrist
        4: 8,   # Right shoulder
        5: 9,   # Right elbow
        6: 10,  # Right wrist
        7: 11,  # Left hip
        8: 12,  # Left knee
        9: 13,  # Left ankle
        10: 14, # Right hip
        11: 15, # Right knee
        12: 16, # Right ankle 
    }

    EXTRA_TOKENS = {
        "left_eye": 0,
        "right_eye": 1,
        "neck": 3,
        "root_of_tail": 4
    }

    SKELETON = {
        "left_fore_limb": [1, 2, 3],
        "right_fore_limb": [4, 5, 6],
        "left_hind_limb": [7, 8, 9],
        "right_hind_limb": [10, 11, 12],
    }

    def __init__(
        self,
        path: str,
        sub_split: Literal[
            1, 2, 3
        ] = 1,
        split: Literal["train", "test", "val"] = "train",
    ):
        if sub_split not in [
            1,
            2,
            3
        ]:
            raise ValueError(
                "Invalid sub_split {} (must be one of 1, 2, 3)".format(
                    sub_split
                )
            )
        if split not in ["train", "test", "val"]:
            raise ValueError(
                "Invalid split {} (must be one of 'train', 'test', 'val')".format(split))
        self.path = path
        self.sub_split = sub_split
        self.split = split
        self.json = os.path.join(path, "annotations", f"ap10k-{split}-split{sub_split}.json")
        self.ap10k = COCO(self.json)
        self.ids = list(sorted(self.ap10k.imgs.keys()))

    def __getitem__(self, idx):

        _id = self.ids[idx]

        # Get image
        image_path = self.ap10k.loadImgs(_id)[0]["file_name"]
        image = cv2.imread(os.path.join(self.path, "data", image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anns = self.ap10k.loadAnns(self.ap10k.getAnnIds(_id))
        ann = anns[0]  # Although there may be one annotation per image, we only use the first one.
        # keys
        # id, image_id, category_id, animal_parent_class, animal_class, animal_subclass, animal, protocol
        # train_test, area, scale, center, bbox, iscrowd, num_keypoints, keypoints

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
