import cv2
import sys
import time
def getImage(vc,frame):
    frame=cv2.resize(frame, (1920, round(3496/4656*1920)), fx = 0.1, fy = 0.1)
    return frame
def change_focus(newfocus):
    global vc
    vc.set(28,newfocus*5)

focus=101
vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 4978)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 3264)
vc.set(28,focus)
cv2.namedWindow("Image")
cv2.createTrackbar('Focus', 'Image', focus, 255, change_focus)
if vc.isOpened():
        rval, frame = vc.read()
else:
    rval = False
i=0
while rval and i<1000:
    rval, frame = vc.read()  # Read a new frame
    frame=cv2.resize(frame, (1920, round(3496/4656*1920)), fx = 0.1, fy = 0.1)
    if not rval:
        break
    getImage(vc,frame)
    cv2.imshow("Image", frame)
    cv2.imwrite("images/image"+str(i)+".jpg",frame)
    # Exit on ESC key press
    i+=1
    time.sleep(0.1)
    if cv2.waitKey(20) == 27:
        break
time.sleep(1)

# Clean up resources
cv2.destroyAllWindows()
vc.release()

