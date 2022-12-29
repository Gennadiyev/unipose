import sys
from uuid import uuid4
import os
from time import time
from datetime import datetime
import argparse

from loguru import logger
import torch
from PIL import Image


from unipose.models import UniPose


def save_tensor_image_to_file(tensor: torch.Tensor, file_path: str):

    if tensor.dim() == 3:
        # (C, H, W)
        _image_data = tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
        _image = Image.fromarray(_image_data if _image_data.shape[2] == 3 else _image_data[:, :, 0])
        _image.save(file_path)
    elif tensor.dim() == 2:
        # (H, W)
        _image_data = tensor.mul(255).byte().cpu().numpy()
        _image = Image.fromarray(_image_data)
        _image.save(file_path)


def train(model, dataloader, criterion, optimizer, scheduler=None, /, device=torch.device("cpu")):
    model.train()
    # tic = time()

    for i, data in enumerate(dataloader):
        # logger.debug("[Perf] Dataloader took {:.3f} seconds to provide a batch of data", time() - tic)

        images = data["images"].to(device)
        keypoint_images = data["keypoint_images"].to(device)
        masks = data["masks"].to(device)
        # import pdb; pdb.set_trace()

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
        if scheduler is not None:
            scheduler.step()

        if i % 50 == 0:
            logger.info("Iteration: {} / {}, Loss: {}".format(i, len(dataloader), loss.item()))
        # tic = time()


cwd = os.getcwd()
cwd = (
    os.path.dirname(cwd) if os.path.basename(cwd) == "scripts" else cwd
)  # Because the train script runs from the scripts folder
def get_abs_path(dir_path: str, create_if_not_exists: bool = False):
    """Gets the absolute path of any path: relative or absolute."""
    if not os.path.isabs(dir_path):
        _path = os.path.join(cwd, dir_path)
        if create_if_not_exists:
            os.makedirs(_path, exist_ok=True)
        return _path
    else:
        if create_if_not_exists:
            os.makedirs(dir_path, exist_ok=True)
        return dir_path


if __name__ == "__main__":
    # Run: python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=3390 train.py
    parser = argparse.ArgumentParser()
    # Model configurations
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--scale_factor", type=float, default=4)
    # Training process configurations
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    # Datasets configurations
    parser.add_argument("--coco", action="store_true", default=False)
    parser.add_argument("--coco_path", type=str, default="datasets/coco")
    parser.add_argument("--mpii", action="store_true", default=False)
    parser.add_argument("--mpii_path", type=str, default="datasets/mpii")
    parser.add_argument("--animal_kingdom", action="store_true", default=False)
    parser.add_argument("--animal_kingdom_path", type=str, default="datasets/animal_kingdom")
    parser.add_argument("-a", "--all", action="store_true", default=False)
    # Output configuration
    parser.add_argument("-o", "--output_dir", type=str, default="exp")
    args = parser.parse_args()
    session_id = str(uuid4())[:4]

    output_dir = get_abs_path(args.output_dir, create_if_not_exists=True)

    # Log to file
    logger_path = os.path.join(output_dir, "train_{}_run-{}.log".format(datetime.now().strftime("%m-%d_%H%M%S"), session_id))
    logger.add(logger_path, level="DEBUG")
    logger.info("Logging to {}...", logger_path)
    
    # Log complete command to file
    logger.info("Command: {}", " ".join(sys.argv))

    # Set device
    if args.gpu == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info("Using device {} (GPU {})", device, args.gpu)

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
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.96)

    # Load dataset
    dataset_path = {
        "animal_kingdom": get_abs_path(args.animal_kingdom_path),
        "mpii": get_abs_path(args.mpii_path),
        "coco": get_abs_path(args.coco_path),
    }

    datasets = []
    logger.info("Loading datasets...")
    from unipose.datasets import AnimalKingdomDataset, MPIIDataset, COCODataset, ConcatJointDataset
    if args.animal_kingdom or args.all:
        _path = dataset_path.get("animal_kingdom")
        logger.debug("Loading Animal Kingdom dataset from {}...", _path)
        dataset_animal_kingdom_amphibian = AnimalKingdomDataset(path=_path, sub_category="ak_P3_amphibian")
        datasets.append(dataset_animal_kingdom_amphibian)
        dataset_animal_kingdom_mammal = AnimalKingdomDataset(path=_path, sub_category="ak_P3_mammal")
        datasets.append(dataset_animal_kingdom_mammal)
    if args.mpii or args.all:
        _path = dataset_path.get("mpii")
        logger.debug("Loading MPII dataset from {}...", _path)
        dataset_mpii = MPIIDataset(path=_path)
        datasets.append(dataset_mpii)
    if args.coco or args.all:
        _path = dataset_path.get("coco")
        logger.debug("Loading COCO dataset from {}...", _path)
        dataset_coco = COCODataset(path=_path)
        datasets.append(dataset_coco)
    if len(datasets) == 0:
        logger.error("No dataset selected: --coco, --mpii, --animal_kingdom or -a / --all must be specified")
        exit(1)
    elif len(datasets) >= 2:
        dataset = ConcatJointDataset(datasets)
    else:
        dataset = datasets[0]
    
    dataloader = dataset.make_dataloader(
        image_size=args.image_size,
        scale_factor=args.scale_factor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    # Train
    logger.info("Start training [{}]", session_id)
    for epoch in range(args.num_epochs):

        train(model, dataloader, criterion, optimizer, None, device=device)
        logger.success("Epoch {} / {} completed", epoch + 1, args.num_epochs)

        if epoch % 5 == 4 or epoch == args.num_epochs - 1:
            checkpoint_path = os.path.join(output_dir, "model_run-{}_ep-{}.pth".format(session_id, epoch + 1))
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("Checkpoint saved at {}", checkpoint_path)

            # Softlink the latest checkpoint
            if os.path.exists(os.path.join(output_dir, "model-latest.pth")):
                os.remove(os.path.join(output_dir, "model-latest.pth"))
            os.symlink(checkpoint_path, os.path.join(output_dir, "model-latest.pth"))

            optimizer_path = os.path.join(output_dir, "optimizer_run-{}_ep-{}.pth".format(session_id, epoch + 1))
            torch.save(optimizer.state_dict(), optimizer_path)
            logger.info("Optimizer saved at {}", optimizer_path)

            # scheduler_path = os.path.join(output_dir, "scheduler_run-{}_ep-{}.pth".format(session_id, epoch + 1))
            # torch.save(scheduler.state_dict(), scheduler_path)
            # logger.info("Scheduler saved at {}", scheduler_path)

