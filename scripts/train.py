import os
from time import time
from datetime import datetime
import argparse

from loguru import logger
import torch


from unipose.models import UniPose


def train(model, dataloader, criterion, optimizer, /, device=torch.device("cpu")):
    model.train()
    # tic = time()

    for i, data in enumerate(dataloader):
        # logger.debug("[Perf] Dataloader took {:.3f} seconds to provide a batch of data", time() - tic)

        images = data["images"].to(device)
        keypoint_images = data["keypoint_images"].to(device)
        masks = data["masks"].to(device)

        # tic = time()
        output = model(images)
        # logger.debug("[Perf] A forward pass took {:.3f} seconds", time() - tic)

        # tic = time()
        masks = masks.unsqueeze(2).unsqueeze(3)
        loss = criterion(output.mul(masks), keypoint_images.mul(masks))
        optimizer.zero_grad()
        # logger.debug("[Perf] Loss computation took {:.3f} seconds", time() - tic)

        # tic = time()
        loss.backward()
        # logger.debug("[Perf] Backward pass took {:.3f} seconds", time() - tic)

        optimizer.step()

        if i % 50 == 0:
            logger.info("Iteration: {} / {}, Loss: {}".format(i, len(dataloader), loss.item()))
        # tic = time()


if __name__ == "__main__":
    # Run: python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=3390 train.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--scale_factor", type=float, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="exp")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    cwd = os.getcwd()
    parent_dir = (
        os.path.dirname(cwd) if os.path.basename(cwd) == "scripts" else cwd
    )  # Because the train script runs from the scripts folder
    model_save_path = os.path.join(parent_dir, args.output_dir)
    os.makedirs(model_save_path, exist_ok=True)

    # Log to file
    logger_path = os.path.join(model_save_path, "train_{}.log".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    logger.add(logger_path, level="DEBUG")
    logger.info("Logging to {}...", logger_path)

    # Set device
    if args.gpu == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    # Create model
    logger.debug("Creating model...")
    model = UniPose(13, resnet_layers=[3, 8, 36, 3])
    model = model.to(device)

    # Define loss
    logger.debug("Defining losses and optimizer...")
    criterion = torch.nn.MSELoss()
    criterion = criterion.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Load dataset
    dataset_path = "/home/dl2022/d3d/unipose/datasets/animal_kingdom"
    logger.debug("Loading dataset from {}...", dataset_path)
    from unipose.datasets.animal_kingdom import AnimalKingdomDataset

    dataset = AnimalKingdomDataset(path=dataset_path)
    dataloader = dataset.make_dataloader(
        image_size=args.image_size,
        scale_factor=args.scale_factor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    # Train
    logger.info("Start training...")
    for epoch in range(args.num_epochs):

        train(model, dataloader, criterion, optimizer, device=device)
        logger.success("Epoch {} / {} completed", epoch, args.num_epochs)

        checkpoint_path = os.path.join(model_save_path, "model_{}.pth".format(epoch))
        torch.save(model.state_dict(), checkpoint_path)
        logger.info("Checkpoint saved at {}", checkpoint_path)

        # Softlink the latest checkpoint
        if os.path.exists(os.path.join(model_save_path, "model-latest.pth")):
            os.remove(os.path.join(model_save_path, "model-latest.pth"))
        os.symlink(checkpoint_path, os.path.join(model_save_path, "model-latest.pth"))

        optimizer_path = os.path.join(model_save_path, "optimizer_{}.pth".format(epoch))
        torch.save(optimizer.state_dict(), optimizer_path)
        logger.info("Optimizer saved at {}", optimizer_path)
        raise KeyboardInterrupt
