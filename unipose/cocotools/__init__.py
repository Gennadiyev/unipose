"""This cocotools package is adapted from tylin's pycocotools package, which is a Python API for the COCO dataset. UniPose uses this package to load the COCO-like dataset and annotations. UniPose modified the C{pycocotools/coco.py} file to use the faster orjson library and added support to discard all unlabelled images.

Note that the original C{pycocotools} package is not obsolete. It is still the official COCO API. UniPose's C{cocotools} package is only a modified version of the original C{pycocotools} package.

@author: Kunologist
"""

from .coco import COCO
__all__ = ["COCO"]

