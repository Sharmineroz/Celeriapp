#import necesary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

#set up camera and grab reference for raw camera capture
camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640,480))

#time to let camera to warm up
time.sleep(0.1)

#capture frames from camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    image= frame.array

    #show frame in "image" window
    cv2.imshow("Image", image)
    key = cv2.waitKey(1) & 0xFF

    #clear stream in preparation for the next one
    rawCapture.truncate(0)

    #break the loop if q pressed
    if key == ord("q"):
        break

#close "image window"
cv2.destroyAllWindows()
