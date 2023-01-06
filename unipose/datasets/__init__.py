"""Datasets for pose estimation. All datasets are subclasses of L{unipose.dataset.BaseJointDataset}.

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
from .ap10k import AP10KDataset

__all__ = ["COCODataset", "AnimalKingdomDataset", "MPIIDataset", "AP10KDataset", "BaseJointDataset", "ConcatJointDataset"]
