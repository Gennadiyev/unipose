# yolo_unipose

+ main contributor: [Xiangyun Rao](https://github.com/xyrrrrrrrr)

## Intro

yolo_unipose is a pre-processing tool for the [Unipose](https://github.com/Gennadiyev/unipose), which is based on the [YOLOv5](https://github.com/ultralytics/yolov5). It is used to get the main creature in the image, judge its kind, and then return the bounding box of the creature.

## How to use

If you want to see how it works, you can run the `yolo_unipose.py` file directly. The default image is `rxy.jpg`. You can also change the image path in the code or in the command line.

```bash
python yolo_unipose.py --source your_image_path
```

## Reference

+ [YOLOv5](https://github.com/ultralytics/yolov5)