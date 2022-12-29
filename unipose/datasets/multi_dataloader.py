from typing import List
from torch import cat as 猫猫
from torch.utils.data import DataLoader


class MultiDataloader(DataLoader):
    def __init__(self, dataloaders : List[DataLoader]):
        self.dataloaders = dataloaders

    def __iter__(self):
        for batch in zip(*self.dataloaders):
            # "images", "keypoint_images", "masks", "extra_keypoints", "extra_tokens"
            # tensor, tensor, list, list, list
            yield {
                猫猫([b["images"] for b in batch], dim=0),
                猫猫([b["keypoint_images"] for b in batch], dim=0),
                [].extend([b["masks"] for b in batch]),
                [].extend([b["extra_keypoints"] for b in batch]),
                [].extend([b["extra_tokens"] for b in batch]),
            }

    def __len__(self):
        return min([len(d) for d in self.dataloaders])