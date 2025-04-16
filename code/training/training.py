import os
import torch
from ultralytics import YOLO
import torchvision
import yaml
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
import tensorflow as tf
import keras.utils
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, DepthwiseConv2D
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, Adadelta
from PIL import Image
import glob

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Ensure GPU is available
def train_yolo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Dataset path (update if needed)
    dataset_yaml = "C:\\Users\\joona\\ylikool\\baka\\pcb-detector\\code\\training\\newdata_868\\data.yaml"
    # Load YOLOv11 keypoint model
    model = YOLO(model="yolo11n-pose.pt")
    os.chdir("C:\\Users\\joona\\ylikool\\baka\\pcb-detector\\code\\training\\newdata_868")

    # Train model
    model.train(
        data=dataset_yaml,
        epochs=100,  # Number of training epochs
        imgsz=640,  # Image size
        batch=16,    # Adjust batch size based on GPU memory
        device=device,  # Use GPU if available
        project="yolov11_keypoint_train",
        name="experiment_2",
        val=False  # Run validation
    )

    # Save trained model
    model_path = "yolov11_keypoint_train/experiment_2/weights/best.pt"
    print(f"Training complete. Model saved at: {model_path}")


# Load dataset configuration from data.yaml
def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == '__main__':
    train_yolo()