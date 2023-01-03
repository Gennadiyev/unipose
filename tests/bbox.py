import pytest
import torch
from torchvision.transforms import ToTensor

from unipose.datasets.utils import enlarge_bounding_box, process_batch


def test_enlarge_bbox_int():
    base_bbox = torch.tensor([[400, 200, 120, 160]])
    square_bbox = enlarge_bounding_box(base_bbox)
    target_bbox = torch.tensor([[380, 200, 160, 160]], dtype=torch.float32)
    assert torch.allclose(square_bbox, target_bbox)


def test_enlarge_bbox_float():
    base_bbox = torch.tensor([[400.0, 200.0, 130.0, 160.0]])
    square_bbox = enlarge_bounding_box(base_bbox)
    target_bbox = torch.tensor([[385.0, 200.0, 160.0, 160.0]], dtype=torch.float32)
    assert torch.allclose(square_bbox, target_bbox)


def test_enlarge_bbox_batch():
    base_bbox = torch.randint(0, 1000, (10, 4), dtype=torch.float32)
    square_bbox = enlarge_bounding_box(base_bbox)
    assert torch.allclose(square_bbox[:, 2], square_bbox[:, 3])


@pytest.mark.contains_absolute_path
def test_process_batch():
    from PIL import Image, ImageDraw

    image = Image.open("/home/dl2022/d3d/unipose/datasets/mpii/images/000142573.jpg")
    bbox = torch.tensor([[400.0, 200.0, 120.0, 160.0]])
    keypoints = torch.tensor([[[500.0, 300.0], [450.0, 350.0]]])
    label_mask = torch.tensor([[1, 1]])
    image_tensor = ToTensor()(image).reshape(1, 3, image.height, image.width)
    im_tmp = Image.fromarray(image_tensor.numpy().transpose(0, 2, 3, 1)[0].astype("uint8"))
    im_tmp.save("cache/test_process_batch_before_crop.png")
    image_cropped, keypoints_images = process_batch(image_tensor, bbox, keypoints, label_mask, 128, 4)
    assert image_cropped.shape == (1, 3, 128, 128)
    assert keypoints_images.shape == (1, 2, 32, 32)

    image_with_bbox = Image.fromarray(image_tensor.numpy().transpose(0, 2, 3, 1)[0].astype("uint8"))
    raise ValueError(image_tensor.numpy().transpose(0, 2, 3, 1)[0].astype("uint8"))
    imdraw = ImageDraw.Draw(image_with_bbox)
    # Draw box
    imdraw.rectangle(((400, 200), (520, 360)), outline="red")
    # Draw keypoints
    for i in range(keypoints.shape[1]):
        imdraw.ellipse(
            (keypoints[0, i, 0] - 2, keypoints[0, i, 1] - 2, keypoints[0, i, 0] + 2, keypoints[0, i, 1] + 2),
            fill="white",
        )
    image_with_bbox.save("cache/test_process_batch_with_bbox.png")
    del imdraw

    image_after_crop = Image.fromarray(image_cropped.numpy().transpose(0, 2, 3, 1)[0].astype("uint8"))
    imdraw = ImageDraw.Draw(image_after_crop)
    image_after_crop.save("cache/test_process_batch_after_crop.png")
