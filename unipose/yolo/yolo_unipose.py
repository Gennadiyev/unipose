'''
This is the main file for the yolo-unipose class. 

@author:Xiangyun Rao
'''

import argparse
import os
import sys
from pathlib import Path
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes)
from utils.plots import Annotator, colors
from utils.segment.general import process_mask
from utils.torch_utils import select_device


class yolo_unipose(torch.nn.Module):
    """
    yolo-unipose model
    """
    def __init__(
        self,
        weights:str = './yolov5m-seg.pt',  # model.pt path(s)
        source:str = 'data/images',  # file/dir/URL/glob, 0 for webcam
        data:str = ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz:tuple = (640, 640),  # inference size (height, width)
        conf_thres:float = 0.25,  # confidence threshold
        iou_thres:float = 0.8,  # NMS IOU threshold
        device:int or str = '',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes:int or list = [0,13,15,16,17,18,19,20,21,22,23],  # filter by class: --class 0, or --class 0 2 3
        project:str = 'yolo_results',  # save results to project/name
        name:str = 'images',  # save results to project/name
        line_thickness:int = 3,  # bounding box thickness (pixels)
        vid_stride:int = 1,  # video frame-rate stride
    ):
        """
        The yolo-unipose model, which is a pre-process part unipose.
        @param weights: model.pt path(s)
        @param data: dataset.yaml path
        @param imgsz: inference size (height, width)
        @param conf_thres: confidence threshold
        @param iou_thres: NMS IOU threshold
        @param device: cuda device, i.e. 0 or 0,1,2,3 or cpu
        @param classes: filter by class: --class 0, or --class 0 2 3
        @param project: save results to project/name
        @param name: save results to project/name
        @param line_thickness: bounding box thickness (pixels)
        @param vid_stride: video frame-rate stride
        
        """
        super().__init__()
        device = select_device(device)
        self.device = device
        self.source = source
        self.model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
        self.classes = classes
        self.project = Path(project)
        self.name = name
        self.line_thickness = line_thickness
        self.vid_stride = vid_stride
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device


    def forward(self, image: str):
        """
        The forward function of the yolo-unipose model.
        @param image: the input image, which is a address of the image or video.
        @return: bounding box, class and confidence of the main object.
        """
        source = str(image) if image != None else self.source
        device = select_device(self.device)
        save_img = not source.endswith('.txt')  # save inference images
        is_video = source.endswith('.mp4')
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download
        # Directories
        save_dir = increment_path(Path(self.project) / self.name, exist_ok=False)  # increment run
        (save_dir / 'labels').mkdir(parents=True, exist_ok=True)  # make dir
        # Load model
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        # Dataloader
        bs = 1  # batch_size
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=self.vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs
        # Run inference
        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                pred, proto = self.model(im, augment=False, visualize=False)[:2]

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, True, max_det=1, nm=32)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                imc = im0 
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
                if len(det):
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        if save_img :  # Add bbox to image
                            c = int(cls)  # integer class
                            label = (names[c])
                            annotator.box_label(xyxy, label, color=colors(c, True))
                im0 = annotator.result()
                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
            # Print results
            t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            if save_img:
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
                
            if pred!=None:
                pred = pred[0]
                return pred[:,:4].cpu().numpy()[0], names[int(pred[:,6])], pred[:,5].cpu().numpy()
            else:
                return None, None, None       

def parse_opt():
    """
    Parse command line arguments if run as a script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5m-seg.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='../tests/data/rxy.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.8, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3',default=[0,13,15,16,17,18,19,20,21,22,23])
    parser.add_argument('--project', default='yolo_results', help='save results to project/name')
    parser.add_argument('--name', default='images', help='save results to project/name')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    """
    Main function if run as script
    """
    img = '../tests/data/rxy.jpg'
    model = yolo_unipose(**vars(opt))
    bounding_box, box_cls, confidence = model(img)
    if bounding_box is not None:
        print(bounding_box)
        print(box_cls)
        print(confidence)
    else:
        print("No object detected")
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
