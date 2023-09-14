#This code snipit detect the objects from the given images paths and  print the label of the detected
#also show the image with bounding boxes draw on it. 
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import io
import base64
import datetime

ROOT = '/home/rcai/Desktop/yolov5'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
import utils
from utils.augmentations import letterbox 
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial import distance as dist
import imutils
from threading import Thread
import playsound
import threading 
import pandas as pd
import numpy
from datetime import datetime 
from imutils import face_utils

device = select_device('cpu') # Set 0 if you have GPU
model = DetectMultiBackend('yolov5s.pt', device=device, dnn=False, data='data/coco128.yaml')
model.classes = [0,1,2,3,4,5,6,7,8,9,10,12]
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size((640, 640), s=stride)  # check image size

dataset = LoadImages('road.jpeg', img_size=imgsz, stride=stride, auto=pt)
def custom_infer(img0, 
        weights='./best.pt',  # model.pt path(s),
        data='data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.35,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=[0,1,2,3,4,5,6,7,8,9,10,12],  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=model):
    img = letterbox(img0, 640, stride=stride, auto=True)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    im = torch.from_numpy(img).to(device)
    im = im.float() # uint8 to fp16/32
    im /= 255  # pixcels convert from 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    dt = [0.0,0.0,0.0]    
    pred = model(im, augment=augment, visualize=visualize)
    seen = 0
    if 1<2: # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame =  'webcam.jpg', img0.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if 1<2:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        print(label)
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
    return im0,pred

# Load and preprocess the image
image_path = 'road.jpeg'
img = cv2.imread(image_path)  # Load the image using OpenCV
img = np.ascontiguousarray(img)  # Ensure contiguous memory layout
img_size = 640  # You can adjust the image size based on your model
img = cv2.resize(img, (img_size, img_size))  # Resize the image

print("These are the detected object")
pred_img = custom_infer(img0 = img)[0]
print(pred_img,"Predicted image") # print array of predicted image pixels 
cv2.imshow("frame", pred_img)
cv2.waitKey(0) #This line will cause the program to wait until a key is pressed before closing the window.
