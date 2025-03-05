import cv2
import sys
from ultralytics import YOLO
import time

def train_model():
    # Initialize YOLO model
    model = YOLO("yolo11n.pt")  # Start with YOLOv11 pre-trained weights
    
    # Train the model
    model.train(
        data="./data.yaml",  # Path to your dataset YAML file
        epochs=50,                   # Number of epochs to train
        imgsz=640,                   # Image size for training
        batch=16,                    # Batch size
        device=0                     # Use GPU (0) or CPU (-1)
    )

# Function to capture and display real-time YOLO model inference
def getImage(model,vc,frame):
    print("Camera dimensions: ", 3496, 4656)
    frame=cv2.resize(frame, (1920, round(3496/4656*1920)), fx = 0.1, fy = 0.1)
    results = model([frame])  # Run inference on the current frame

    for result in results:
        # Iterate over detected boxes and display on the frame
        for box in result.boxes:
            # Get box coordinates and convert them to integers
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert tensor to list and cast to int
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if box.conf is not None and box.cls is not None:
                label = f"{model.names[int(box.cls.item())]}: {float(box.conf.item()):.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow("YOLO Detection", frame)
    return frame

focus=0
vc = cv2.VideoCapture(0)

def change_focus(newfocus):
    global vc
    vc.set(28,newfocus*5)

if __name__ == "__main__":
    focus=82
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 4978)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 3264)
    vc.set(28,focus)
    cv2.namedWindow("YOLO Detection")
    cv2.createTrackbar('Focus', 'YOLO Detection', focus, 255, change_focus)
    #train
    ###########train_model()##############
    # Load a YOLO model
    model = YOLO("training/yolo_dataset/yolov11_keypoint_train/experiment_14/weights/best.pt")  # Ensure "yolo11n.pt" model path is correct
    
    # Set camera properties if necessary
    # vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    # vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Check if the video capture is open
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        rval, frame = vc.read()  # Read a new frame
        if not rval:
            break
        getImage(model,vc,frame)
        # Exit on ESC key press
        if cv2.waitKey(20) == 27:
            break
    time.sleep(1)

    # Clean up resources
    cv2.destroyAllWindows()
    vc.release()