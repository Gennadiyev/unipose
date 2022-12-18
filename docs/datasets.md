# Datasets

Each item accessed from a unipose dataset contains the following fields:

```js
{
    "image": image,
    "bounding_box": bbox,
    "unipose_keypoints": keypoints_unipose,
    "unipose_mask": mask_unipose,
    "extra_keypoints": extra_keypoints,
    "extra_tokens": extra_token_names,
    "original_keypoints": keypoints,
    "original_mask": mask,
}
```

where:

- `image` is a `torch.Tensor` of shape `(W, H, 3)` containing the image.
- `bounding_box` is a `torch.Tensor` of shape `(4,)` containing the bounding box of the main object in the image.
- `unipose_keypoints` is a `torch.Tensor` of shape `(K, 2)` containing the keypoints of the main object in the image.
- `unipose_mask` is a `torch.Tensor` of shape `(K,)` defines whether each keypoint in `unipose_keypoints` is visible or not. (Usually models should only be trained on visible keypoints, or `unipose_mask[i] == 1`.)
- `extra_keypoints` is a `torch.Tensor` of shape `(N, 2)` containing the keypoints of the `N` extra tokens that are present in the image. Varies across datasets. (Even in one dataset, this field can have variable length given the masking of the keypoints in a certain image.)
- `extra_tokens` is a `torch.Tensor` of shape `(N,)` containing the names of the `N` extra tokens that are present in the image. Varies across datasets. Each token is a string.
- `original_keypoints` is a `torch.Tensor` of shape `(K', 2)` containing the keypoints of the main object in the image. This is extracted raw from the dataset, and is not aligned to the unipose keypoints.
- `original_mask` is a `torch.Tensor` of shape `(K',)` defines whether each point in `original_keypoints` is visible or not.
