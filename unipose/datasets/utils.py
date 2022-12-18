from typing import List

import cv2
import numpy as np
import torch


def enlarge_bounding_box(bbox: torch.Tensor) -> torch.Tensor:
    """Enlarge the bounding box to the smallest square that contains it without changing its center.
    
    @param bbox: A tensor of shape (4) containing the bounding box in the format [x, y, w, h].
    @return: A tensor of shape (4) containing the enlarged bounding box in the format [x1, y1, w1, h1].
    """
    # Diff of height and width
    diff = bbox[:, 2] - bbox[:, 3]
    bbox_enlarged = bbox.clone().float()
    for i in range(len(bbox_enlarged)):
        if diff[i] > 0:
            # w > h
            bbox_enlarged[i, 1] -= diff[i] / 2
            bbox_enlarged[i, 3] += diff[i]
        elif diff[i] < 0:
            # w < h
            bbox_enlarged[i, 0] += diff[i] / 2
            bbox_enlarged[i, 2] -= diff[i]
    return bbox_enlarged


def crop_image(image: torch.Tensor, bbox_rounded: torch.Tensor, image_size: int) -> torch.Tensor:
    """Crop the image to the bounding box and resize it to the given size.
    
    @param image: A tensor of shape (batchsize, channels, height, width) containing the image.
    @param bbox_rounded: A tensor of shape (batchsize, 4) containing the bounding box in the format [x, y, w, h].
    @param image_size: The size to which the image should be resized.
    @return: A tensor of shape (batchsize, channels, image_size, image_size) containing the cropped and resized image.
    """
    batchsize, channels = image.shape[0], image.shape[1]
    image_resized = torch.zeros(batchsize, channels, image_size, image_size)
    for i in range(batchsize):
        # image_cropped = image[i, :, bbox_rounded[i, 1]:bbox_rounded[i, 1] + bbox_rounded[i, 3],
        #                             bbox_rounded[i, 0]:bbox_rounded[i, 0] + bbox_rounded[i, 2]]
        # print(image_cropped.shape)
        # image_resized[i, :, :, :] = F.interpolate(image_cropped.unsqueeze(0), size=(image_size, image_size), mode='area').squeeze(0)
        point1 = np.array([
            [bbox_rounded[i, 0].item(), bbox_rounded[i, 1].item()],
            [bbox_rounded[i, 0].item(), bbox_rounded[i, 1].item() + bbox_rounded[i, 3].item()],
            [bbox_rounded[i, 0].item() + bbox_rounded[i, 2].item(), bbox_rounded[i, 1].item() + bbox_rounded[i, 3].item()],
            [bbox_rounded[i, 0].item() + bbox_rounded[i, 2].item(), bbox_rounded[i, 1].item()]
        ], dtype=np.float32)
        point2 = np.array([
            [0, 0],
            [0, image_size],
            [image_size, image_size],
            [image_size, 0]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(point1, point2)
        image_cropped = cv2.warpPerspective(image[i].cpu().permute(1, 2, 0).numpy(), M, (image_size, image_size))
        image_resized[i] = torch.from_numpy(image_cropped).permute(2, 0, 1)
    image_resized = image_resized.to(image.device)
    return image_resized


def crop_keypoints(keypoints: torch.Tensor, bbox_rounded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Crop the keypoints to the bounding box.
    
    @param keypoints: A tensor of shape (batchsize, num_keypoints, 2) containing the keypoints.
    @param bbox: A tensor of shape (batchsize, 4) containing the bounding box in the format [x, y, w, h].
    @param mask: A tensor of shape (batchsize, num_keypoints) containing the mask.
    @return: A tensor of shape (batchsize, num_keypoints, 3) containing the cropped keypoints.
    """
    keypoints_cropped = keypoints.clone()
    for i in range(len(keypoints_cropped)):
        keypoints_cropped[i, :, 0] -= bbox_rounded[i, 0]
        keypoints_cropped[i, :, 1] -= bbox_rounded[i, 1]
        keypoints_cropped[i, :, 0] /= bbox_rounded[i, 2]
        keypoints_cropped[i, :, 1] /= bbox_rounded[i, 3]
    keypoints_cropped = keypoints_cropped.mul(mask.unsqueeze(2).float())
    return keypoints_cropped


def get_keypoints_images(keypoints: torch.Tensor, image_size: int, mask: torch.Tensor) -> torch.Tensor:
    """Generate the images of the keypoints.
    
    @param keypoints: A tensor of shape (batchsize, num_keypoints, 2) containing the keypoints.
    @param image_size: The size of the image.
    @return: A tensor of shape (batchsize, num_keypoints, image_size, image_size) containing the images of the keypoints.
    """
    batchsize, num_keypoints, _ = keypoints.shape
    images = torch.zeros(batchsize, num_keypoints, image_size, image_size).to(keypoints.device)
    keypoints_multiplied = keypoints.clone()
    keypoints_multiplied *= image_size
    keypoints_rounded = torch.round(keypoints_multiplied).long().clamp(0, image_size - 1)
    for i in range(batchsize):
        for j in range(num_keypoints):
            images[i, j, keypoints_rounded[i, j, 1], keypoints_rounded[i, j, 0]] = 1
    images = images.mul(mask.unsqueeze(2).unsqueeze(3).float())
    return images


def process_batch(image: torch.Tensor, bbox: torch.Tensor, keypoints: torch.Tensor, label_mask: torch.Tensor, image_size: int, scale_factor: int) -> List[torch.Tensor]:
    """Process a batch of data.
    
    @param image: A tensor of shape (batchsize, channels, height, width) containing the image.
    @param bbox: A tensor of shape (batchsize, 4) containing the bounding box in the format [x, y, w, h].
    @param keypoints: A tensor of shape (batchsize, num_keypoints, 2) containing the keypoints.
    @param label_mask: A tensor of shape (batchsize, num_keypoints) containing the mask.
    @param image_size: The size to which the image should be resized.
    @param scale_factor: The factor by which the image should be scaled down, which is determined by the model.
    @return: A list of tensors containing the cropped and resized image and the images of the keypoints.
    """
    bbox_enlarged = enlarge_bounding_box(bbox)
    # print(f"{bbox_enlarged=}")
    bbox_rounded = torch.round(bbox_enlarged).long()
    # print(f"{bbox_rounded=}")
    image_cropped = crop_image(image, bbox_rounded, image_size)
    # print(f"{image_cropped.shape=}")
    keypoints_cropped = crop_keypoints(keypoints, bbox_rounded, label_mask)
    # print(f"{keypoints=}")
    # print(f"{keypoints_cropped=}")
    keypoints_images = get_keypoints_images(keypoints_cropped, image_size // scale_factor, label_mask)
    # print(f"{keypoints_images.shape=}")
    return [image_cropped, keypoints_images]
