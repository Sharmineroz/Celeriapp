import numpy as np
import cv2
import tensorflow as tf

cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print(ret)
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parser = tf.cast(frame,tf.float32)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(frame.shape)
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


