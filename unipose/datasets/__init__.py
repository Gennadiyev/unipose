"""Datasets for pose estimation.

Contains the following datasets:

- COCO (only human annotations)
- Animal Kingdom (animals from COCO)
- MPII (human pose estimation dataset)

@author: Yikun Ji
"""

from .base_joint_dataset import BaseJointDataset, ConcatJointDataset
from .animal_kingdom import AnimalKingdomDataset
from .coco import COCODataset
from .mpii import MPIIDataset

__all__ = ["COCODataset", "AnimalKingdomDataset", "MPIIDataset", "BaseJointDataset", "ConcatJointDataset"]
