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

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Ensure GPU is available
def train_yolo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Dataset path (update if needed)
    dataset_yaml = "C:\\Users\\joona\\ylikool\\bakarepo\\pcb-detector\\code\\training\\yolo_dataset\\data.yaml"
    # Load YOLOv11 keypoint model
    model = YOLO(model="yolo11n-pose.pt")#.load("yolo11n-pose.pt")   # You can change to yolov11s-pose.pt or other versions
    os.chdir("C:\\Users\\joona\\ylikool\\bakarepo\\pcb-detector\\code\\training\\yolo_dataset\\")

    # Train model
    model.train(
        data=dataset_yaml,
        epochs=100,  # Number of training epochs
        imgsz=640,  # Image size
        batch=16,    # Adjust batch size based on GPU memory
        device=device,  # Use GPU if available
        project="yolov11_keypoint_train",
        name="experiment_1",
        val=False  # Run validation
    )

    # Save trained model
    model_path = "yolov11_keypoint_train/experiment_1/weights/best.pt"
    print(f"Training complete. Model saved at: {model_path}")


# Load dataset configuration from data.yaml
def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


# Custom dataset class
class SkeletonDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = [f for f in os.listdir("C:\\Users\\joona\\ylikool\\baka\\pcb-detector\\code\\training\\yolo_dataset\\images") if f.endswith(".jpg")]
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_files[idx])
        label_path = "code\\training\\yolo_dataset\\labels"+img_path.replace(".jpg", ".txt").replace("train.txt","")
        
        image = Image.open("code\\training\\yolo_dataset\\images"+img_path.replace("train.txt","")).convert("RGB")
        with open(label_path, "r") as file:
            labels = [list(map(float, line.strip().split())) for line in file.readlines()]
        
        boxes = torch.tensor([l[1:] for l in labels], dtype=torch.float32)  # x1, y1, x2, y2
        
        labels = torch.tensor([int(l[0]) for l in labels], dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels}
        if self.transform:
            image = self.transform(image)
        return image, target
    
    def __len__(self):
        return len(self.image_files)

# Load datasets
data_transform = ToTensor()
train_dataset = SkeletonDataset(train_dir, transform=data_transform)
val_dataset = SkeletonDataset(val_dir, transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

model = Sequential()

#https://github.com/VL97/Pose-estimation-using-Deep-CNN-encoder/blob/master/train.py
model.add(DepthwiseConv2D(kernel_size=(1,1), strides=(4, 4), depth_multiplier=1,\
                use_bias=False,input_shape=(220,220,3)))
model.add(Conv2D(96, (11, 11),activation='relu', padding='same'))
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(16*2, activation='tanh'))

model.compile(loss=tf.losses.MeanSquaredError, optimizer=Adam())
model.summary()