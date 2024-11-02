import torch
import pandas
import matplotlib
# import bpy
from ultralytics import YOLO

import cv2
# from pytorchyolo import detect, models

def yolo11():
    # Load a COCO-pretrained YOLO11n model
    model = YOLO("/home/taylor/Duke/AMLL/Synthetic_Data_Generation/pythonProject1/yolo11n.pt")

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="/home/taylor/Duke/AMLL/Synthetic_Data_Generation/pythonProject1/datasets/MADD_overhead_dsiac_test/data.yaml",
                          epochs=3,
                          imgsz=640)
    debug=1
    # # Run inference with the YOLO11n model on the 'bus.jpg' image
    # results = model("path/to/bus.jpg")

def main():
    yolo11()

if __name__ == '__main__':
    main()
