import os
from typing import Literal

import cv2
import torch
from pycocotools.coco import COCO
from torchvision.transforms import ToTensor

from .base_joint_dataset import BaseJointDataset


class AnimalKingdomDataset(BaseJointDataset):
    
    MAPPING = {
        0 : 3 , # Nose
        1 : 7 , # Left shoulder
        2 : 9 , # Left elbow
        3 : 11, # Left wrist
        4 : 8 , # Right shoulder
        5 : 10, # Right elbow
        6 : 12, # Right wrist
        7 : 14, # Left hip
        8 : 16, # Left knee
        9 : 18, # Left ankle
        10: 15, # Right hip
        11: 17, # Right knee
        12: 19, # Right ankle
    }
    
    EXTRA_TOKENS = {
        "head_top"    : 0,
        "left_eye"    : 1,
        "right_eye"   : 2,
        "mouth_left"  : 4,
        "mouth_right" : 5,
        "mouth_bottom": 6,
        "torso_middle": 13,
        "tail_top"    : 20,
        "tail_middle" : 21,
        "tail_end"    : 22,
    }
    
    def __init__(
        self,
        path: str,
        sub_category: Literal["ak_P1", "ak_P2", "ak_P3_amphibian", "ak_P3_bird", "ak_P3_fish", "ak_P3_mammal", "ak_P3_reptile"]="ak_P1",
        split: Literal["train", "test"]="train"
    ):
        if sub_category not in ["ak_P1", "ak_P2", "ak_P3_amphibian", "ak_P3_bird", "ak_P3_fish", "ak_P3_mammal", "ak_P3_reptile"]:
            raise ValueError("Invalid sub-category {} (must be one of 'ak_P1', 'ak_P2', 'ak_P3_amphibian', 'ak_P3_bird', 'ak_P3_fish', 'ak_P3_mammal', 'ak_P3_reptile')".format(sub_category))
        if split not in ["train", "test"]:
            raise ValueError("Invalid split {} (must be one of 'train', 'test')".format(split))
        self.path = path
        self.sub_category = sub_category
        self.split = split
        self.json = os.path.join(
            path, "annotation_coco", self.sub_category, f"{self.split}.json"
        )
        self.animal_kingdom = COCO(self.json)
        self.ids = list(sorted(self.animal_kingdom.imgs.keys()))

    def __getitem__(self, idx):
        
        _id = self.ids[idx]
        
        # Get image
        image_path = self.animal_kingdom.loadImgs(_id)[0]["file_name"]
        image = cv2.imread(
            os.path.join(self.path, "dataset", image_path)
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        anns = self.animal_kingdom.loadAnns(self.animal_kingdom.getAnnIds(_id))
        ann = anns[0] # Although there may be one annotation per image, we only use the first one.
        # keys
        # id, image_id, category_id, animal_parent_class, animal_class, animal_subclass, animal, protocol
        # train_test, area, scale, center, bbox, iscrowd, num_keypoints, keypoints
        
        # Extract information
        # Extract information
        keypoints = ann["keypoints"]
        num_keypoints = ann["num_keypoints"]
        bbox = ann["bbox"]
        _data = torch.tensor(keypoints).reshape(-1, 3)
        keypoints = _data[:, :2]
        mask = _data[:, 2]
        mask[mask != 0] = 1 # 2 is visible, 1 is occluded, 0 is not labeled

        # Convert to tensors
        image = ToTensor()(image)
        bbox, num_keypoints = map(torch.tensor, (bbox, num_keypoints))
        
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
        return len(self.ids)

