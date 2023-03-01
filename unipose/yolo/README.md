# YOLO Integration for Unipose

+ main contributor: [Xiangyun Rao](https://github.com/xyrrrrrrrr)

## Intro

For videos and images without bounding boxes, YOLO is a pre-processing tool for [Unipose](https://github.com/Gennadiyev/unipose) to retrieve the main creature in the image, classify it, and then return the bounding box of the creature.

The project bases heavily on [YOLOv5](https://github.com/ultralytics/yolov5).

## How to use

To see how it works, you can run `yolo_unipose.py` directly. You may also change the image path in the code or in the command line:

```bash
python yolo_unipose.py --source your_image_path
```

## Reference

+ [YOLOv5](https://github.com/ultralytics/yolov5)
