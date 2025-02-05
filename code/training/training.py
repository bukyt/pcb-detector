import os
import torch
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Ensure GPU is available
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
