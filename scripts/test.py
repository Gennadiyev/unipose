import os

import argparse
import logging
import torch

from unipose.datasets import AnimalKingdomDataset, MPIIDataset, COCODataset, ConcatJointDataset
from unipose.models import UniPose


def parse_args():
    parser = argparse.ArgumentParser(description="Test script for UniPose")
    # For model
    parser.add_argument("--load", type=str, default="path/to/model")
    # For input
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--scale_factor", type=int, default=4)
    # For dataset
    parser.add_argument("--coco", action="store_true", default=False)
    parser.add_argument("--mpii", action="store_true", default=False)
    parser.add_argument("--animal_kingdom", action="store_true", default=False)
    parser.add_argument("--coco_path", type=str, default="path/to/coco")
    parser.add_argument("--mpii_path", type=str, default="path/to/mpii")
    parser.add_argument("--animal_kingdom_path", type=str, default="path/to/animal_kingdom")
    parser.add_argument("--all", action="store_true", default=False)
    # For dataloader
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    # For device and log
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()
    return args


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def distance_accuracy(output, keypoint_images, masks):
    # output: (batch_size, num_joints, H, W)
    # keypoints_images: (batch_size, num_joints, H, W)
    # masks: (batch_size, num_joints)
    batch_size, num_joints = masks.shape
    x_pred = torch.zeros(batch_size, num_joints, device=output.device)
    y_pred = torch.zeros(batch_size, num_joints, device=output.device)
    x_label = torch.zeros(batch_size, num_joints, device=output.device)
    y_label = torch.zeros(batch_size, num_joints, device=output.device)
    for i in range(batch_size):
        for j in range(num_joints):
            if masks[i, j] == 0:
                continue
            index_pred = torch.argmax(output[i, j])
            index_label = torch.argmax(keypoint_images[i, j])
            x_pred[i, j] = index_pred % output.shape[3]
            y_pred[i, j] = index_pred // output.shape[3]
            x_label[i, j] = index_label % output.shape[3]
            y_label[i, j] = index_label // output.shape[3]
    distance = torch.sqrt((x_pred - x_label) ** 2 + (y_pred - y_label) ** 2)
    return torch.mean(distance)



@torch.no_grad()
def test(model, dataloader, device, log_interval):
    model.eval()
    acc_all = AverageMeter()
    for i, data in enumerate(dataloader):
        images = data["images"].to(device)
        keypoint_images = data["keypoint_images"].to(device)
        masks = data["masks"].to(device)

        output = model(images)
        acc = distance_accuracy(output, keypoint_images, masks)
        acc_all.update(acc)

        if (i + 1) % log_interval == 0:
            logging.info(f"Testing {i + 1} / {len(dataloader)}, Average accuracy: {acc_all.avg:.6f}")


def main(args):
    # device and logging setting
    device = f"cuda:{args.device}" if args.device != -1 else "cpu"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # Load model
    model = UniPose(13, resnet_layers=[3, 8, 36, 3])
    model.load_state_dict(torch.load(args.load))
    model.to(device)
    # Load dataset
    datasets = []
    if args.all:
        args.coco = args.mpii = args.animal_kingdom = True
    if args.coco:
        logging.info("Loading COCO dataset")
        datasets.append(COCODataset(path=args.coco_path))
    if args.mpii:
        logging.info("Loading MPII dataset")
        datasets.append(MPIIDataset(path=args.mpii_path))
    if args.animal_kingdom:
        logging.info("Loading Animal Kingdom dataset")
        datasets.append(AnimalKingdomDataset(path=args.animal_kingdom_path))
    if len(datasets) == 0:
        raise ValueError("No dataset selected")
    elif len(datasets) >= 2:
        dataset = ConcatJointDataset(datasets)
    else:
        dataset = datasets[0]
    # Load dataloader
    dataloader = dataset.make_dataloader(
        image_size=args.image_size,
        scale_factor=args.scale_factor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    # Test
    logging.info("Start testing")
    test(model, dataloader, device, args.log_interval)
    logging.info("Testing finished")


if __name__ == "__main__":
    args = parse_args()
    main(args)

# python test.py --load ../exp/model_0.pth --coco --coco_path ../datasets/coco
# python test.py --load /date/dl2022/d3d/exp_ext/model_run-bb6a_ep-100.pth --coco --coco_path ../datasets/coco
