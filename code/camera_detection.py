import cv2
import sys
from ultralytics import YOLO
import time
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchvision.transforms.v2 as T
from tqdm import tqdm
import os
from training.model import OBBFasterRCNN
from training.obb_dataset import OBBDataset
import numpy as np
# Function to capture and display real-time YOLO model inference


def draw_obb_box(frame, cx, cy, w, h, angle_deg, label):
    h_img, w_img = frame.shape[:2]

    # Convert normalized coords to pixel values
    cx *= w_img
    cy *= h_img
    w *= w_img
    h *= h_img

    # Define the rotated rectangle
    rect = ((cx, cy), (w, h), angle_deg)

    # Get corner points
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)

    # Draw the box and label
    cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(frame, label, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)


def getImageResNeXt(model, vc, frame):
    frame_disp = frame.copy()
    frame = cv2.resize(frame, (640, 640))
    h_img, w_img = frame.shape[:2]

    #frame = (frame / 255.0).astype("float32")
    img_tensor = ToTensor()(frame).float()

    with torch.no_grad():
        results = model([img_tensor])[0]  # Single image batch
    print(results)
    boxes = results["boxes"]
    labels = results["labels"]
    scores = results["scores"]

    CONFIDENCE_THRESHOLD = 0

    for box, label, score in zip(boxes, labels, scores):
        if score < CONFIDENCE_THRESHOLD:
            continue
        cx, cy, w, h = box.tolist()
        print(cx,cy,w,h)
        label_text = f"{label.item()}:{score.item():.2f}"
        draw_obb_box(frame_disp, cx, cy, w, h, 0, label_text)

    cv2.imshow("resnext Detection", frame_disp)
    return frame_disp
#image resnext detection

focus=0
vc = cv2.VideoCapture(0)

def change_focus(newfocus):
    global vc
    vc.set(28,newfocus*5)

def resNeXt_detection():
    focus=82
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    vc.set(28,focus)
    cv2.namedWindow("resnext Detection")
    cv2.createTrackbar('Focus', 'resnext Detection', focus, 255, change_focus)
    ###########train_model()##############
    model = OBBFasterRCNN(num_classes=2)
    model.load_state_dict(torch.load("training/obb_fasterrcnn_full.pth"))
    model.eval()
    # Check if the video capture is open
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        rval, frame = vc.read()  # Read a new frame
        if not rval:
            break
        getImageResNeXt(model,vc,frame)
        # Exit on ESC key press
        if cv2.waitKey(20) == 27:
            break
    time.sleep(1)

    # Clean up resources
    cv2.destroyAllWindows()
    vc.release()

def yolo_detection():
    focus = 82
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    vc.set(28, focus)
    cv2.namedWindow("YOLO Detection")
    cv2.createTrackbar('Focus', 'YOLO Detection', focus, 255, change_focus)

    model = YOLO("training\\yolo_dataset\\yolov11_keypoint_train\\experiment_14\\weights\\best.pt")

    if not vc.isOpened():
        print("Video capture not opened!")
        return

    keypoint_history = []
    NUM_FRAMES = 4

    while True:
        ret, frame = vc.read()
        if not ret:
            break

        # Resize for inference
        input_frame = cv2.resize(frame, (640, 640))
        results = model(input_frame, conf=0.0001)

        if results:
            keypoints = results[0].keypoints
            if keypoints is not None:
                keypoints_np = keypoints.data.cpu().numpy()[0]  # (num_keypoints, 3)
                keypoint_history.append(keypoints_np)

        # Keep only last N frames of keypoints
        if len(keypoint_history) > NUM_FRAMES:
            keypoint_history.pop(0)

        # Draw all accumulated keypoints on the current frame
        display_frame = input_frame.copy()
        for kp_set in keypoint_history:
            a = 255
            b = 0
            for x, y, conf in kp_set:
                a, b = b, a
                if conf > 0.5:
                    cv2.circle(display_frame, (int(x), int(y)), 4, (a, 0, b), -1)

        cv2.imshow("YOLO Detection", display_frame)

        if cv2.waitKey(20) == 27:
            break

    vc.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    #yolo_detection()
    resNeXt_detection()