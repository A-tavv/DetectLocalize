!pip install --upgrade torch ultralytics
import torch
from torchvision import transforms
from torchvision import torch
from ultralytics import YOLO
from torch.utils.data import DataLoader
import zipfile
import os
from google.colab import drive

# the path to ZIP file on Drive
# please note that the dataset is not exist here anymore
file_path = '/content/drive/MyDrive/Master-thesis.zip'
# Unzip the file
!unzip {file_path}

# importing the Model for our case YOLOv8m version
# the YOLOv8 comes with 5 sizes nano, small, medium, large, extended 
model = YOLO("yolov8m.pt")

results = model.train( data = "/content/data.yaml", epochs = 20  , imgsz = 640 , lr0=0.001 , lrf=0.01 , cos_lr=True , patience = 10 , save_period=5 , batch = 16 , weight_decay = 0.0005 , dropout = 0.3 , verbose = True , name = 'exp' , val = True , save = True  )

result = model.val()

result = model("/content/sample_image.png")

result = model.export(format = "onnx")
# my model was named feedback.pt

infer = YOLO("/content/runs/detect/train/weights/best.pt")

infer.predict("/content/test/images" , save = True , save_txt = True )

infer.predict("/content/IMG.MOV" , conf = 0.70, save = True , save_txt = True)
