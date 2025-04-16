import cv2
import sys
from ultralytics import YOLO
import time


# Function to capture and display real-time YOLO model inference
def getImage(model,vc,frame):
    print("Camera dimensions: ", 3496, 4656)
    frame=cv2.resize(frame, (640, 640), fx = 0.1, fy = 0.1)
    frame=frame/255.0
    results = model(frame, conf=0.001)  # Run inference on the current frame
    CONFIDENCE_THRESHOLD=0.01
    for result in results:
        # Iterate over detected boxes and display on the frame
        for box in result.boxes:
            if box.conf is not None and box.cls is not None:
                confidence = float(box.conf.item())
                if confidence >= CONFIDENCE_THRESHOLD:  # Lower confidence threshold
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{model.names[int(box.cls.item())]}: {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    vc.set(28,focus)
    cv2.namedWindow("YOLO Detection")
    cv2.createTrackbar('Focus', 'YOLO Detection', focus, 255, change_focus)
    ###########train_model()##############
    model = YOLO("training\\newdata_868\\yolov11_keypoint_train\\experiment_212\\weights\\best.pt",task="pose")

    
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