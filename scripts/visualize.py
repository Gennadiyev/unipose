# Also import deprecated
from deprecated import deprecated

import argparse
import os

import cv2
from loguru import logger
import numpy as np
from PIL import Image, ImageDraw
import torch

from unipose.models import UniPose
from unipose.losses import GLES


def inference(image_path, model, /, device=torch.device("cpu")):
    model.eval()
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = np.array(image)
    # remove alpha channel
    if image.shape[2] == 4:
        image = image[:, :, :3]
    image = torch.from_numpy(image).float().div(255).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image to be processed")
    parser.add_argument("--checkpoint", type=str, default="exp/model-latest.pth")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID. Set to -1 to use CPU")
    parser.add_argument("--image_size", type=int, default=256, help="Image size of input image.")
    parser.add_argument(
        "--scale_factor",
        type=int,
        default=4,
        help="Patch size of output keypoint heatmaps. Must be 4 if Unipose is used without modification",
    )
    parser.add_argument("--output_dir", type=str, default="exp/output_graph")
    args = parser.parse_args()

    raise DeprecationWarning("This script is deprecated. Please use vis_graph.py instead.")
    cwd = os.getcwd()
    base_dir = os.path.dirname(cwd) if os.path.basename(cwd) == "scripts" else cwd
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Exit if checkpoint not found
    # checkpoint_path = os.path.join(base_dir, args.checkpoint)
    checkpoint_path = args.checkpoint
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
    ret = inference(args.image_path, model, device=device)

    """
    for i in range(ret.shape[1]):
        _image_arr = ret[0, i, :, :].cpu().numpy()
        # Normalize to 0, 255
        _image_arr = (_image_arr - _image_arr.min()) / (_image_arr.max() - _image_arr.min())
        _image_arr = _image_arr * 255
        cv2.imwrite(os.path.join(output_dir, "heatmap_{}.png".format(i)), _image_arr)
    """

    image = Image.open(args.image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((256, 256), resample=Image.Resampling.BICUBIC)
    image = np.array(image)
    x = []
    y = []
    for i in range(ret.shape[1]):
        _image_arr = ret[0, i, :, :].cpu().numpy()
        _image_arr = (_image_arr - _image_arr.min()) / (_image_arr.max() - _image_arr.min())
        op = os.path.join(output_dir, "heatmap_{}.png".format(i))
        cv2.imwrite(op, _image_arr * 255)
        print("->", op)
        idx = np.argmax(_image_arr)
        x.append(idx % _image_arr.shape[0] * 256 // _image_arr.shape[1])
        y.append(idx // _image_arr.shape[0] * 256 // _image_arr.shape[0])
        image = cv2.circle(image, (x[-1], y[-1]), 1, (255, 255, 255), 1)

    ret = ret.squeeze()
    ret = ret.reshape(ret.shape[0], -1)
    gles = GLES()
    graph = gles(ret)
    assert graph.max() >= 0
    print(graph.max())

    edge_count = 0
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if int(2 * graph[i][j]) > 0:
                p1 = (x[i], y[i])
                p2 = (x[j], y[j])
                image = cv2.line(image, p1, p2, (255, 255, 255), int(2 * graph[i][j]))
                edge_count += 1

    logger.info("Number of edges: {}", edge_count)
    cv2.imwrite(os.path.join(output_dir, "graph.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
