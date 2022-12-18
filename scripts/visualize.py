import argparse
import os

import cv2
from loguru import logger
import numpy as np
from PIL import Image, ImageDraw
import torch

from unipose.models import UniPose


def inference(image_path, model, /, device=torch.device("cpu")):
    model.eval()
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = np.array(image)
    image = torch.from_numpy(image).float().div(255).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image to be processed")
    parser.add_argument("--checkpoint", type=str, default="exp/model-latest.pth")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="exp/output")
    args = parser.parse_args()

    cwd = os.getcwd()
    base_dir = os.path.dirname(cwd) if os.path.basename(cwd) == "scripts" else cwd
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Exit if checkpoint not found
    checkpoint_path = os.path.join(base_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        logger.error("Checkpoint not found: {}".format(checkpoint_path))
        exit(1)

    # Set device
    if args.gpu == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    # Create model
    logger.debug("Creating model...")
    model = UniPose(13, resnet_layers=[3, 8, 36, 3])
    model = model.to(device)
    logger.info("Loading checkpoint from {}...", checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Inference
    logger.info("Processing image: {}", args.image_path)
    # Copy source image to output directory
    output_image_path = os.path.join(output_dir, os.path.basename(args.image_path))
    import shutil
    shutil.copyfile(args.image_path, output_image_path)
    logger.debug("Copied source image to {}", output_image_path)
    # Inference
    ret = inference(args.image_path, model, device=device)
    for i in range(ret.shape[1]):
        cv2.imwrite(f"{i}.png", ret[0, i, :, :].clamp(0, 1).cpu().numpy() * 255)
    import pdb; pdb.set_trace()
