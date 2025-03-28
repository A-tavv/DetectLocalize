# -*- coding: utf-8 -*-
"""My Model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PeZQDzhOIzU5pWs_lSSeed3XwLg0ynwG

#Set up the environement





##Phase 1. Detection by YOLOv8
##phase 2. capture by webcam/ image or video
##Phase 3. Distance/Location/Counting
##Phase 4. Audio feedback

#Phase 1.
diving into YOLOv8 to create our model
###pre work"Creating our model"
"""

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)

!pip install --upgrade torch ultralytics
import torch
from torchvision import transforms
from torchvision import torch
from ultralytics import YOLO
from torch.utils.data import DataLoader
import zipfile
import os
from google.colab import drive

# Specify the path to ZIP file on Drive (replace with the actual path)
file_path = '/content/drive/MyDrive/Master-thesis.zip'
# Unzip the file
!unzip {file_path}
#importing the Model for our case YOLOv8 version Large
model = YOLO("yolov8m.pt")

results = model.train( data = "/content/data.yaml", epochs = 20  , imgsz = 640 , lr0=0.001 , lrf=0.01 , cos_lr=True , patience = 10 , save_period=5 , batch = 16 , weight_decay = 0.0005 , dropout = 0.3 , verbose = True , name = 'exp' , val = True , save = True  )

result = model.val()

result = model("/content/milk.png")

result = model.export(format = "onnx")

infer = YOLO("/content/runs/detect/train/weights/best.pt")

infer.predict("/content/test/images" , save = True , save_txt = True )

infer.predict("/content/IMG.MOV" , conf = 0.70, save = True , save_txt = True)