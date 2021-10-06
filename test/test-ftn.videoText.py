import cv2
import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from module import ftn



device = 0
frameWidth  = 640
frameHeight = 480

vid = cv2.VideoCapture(device)
vid.set(3, frameWidth)
vid.set(4, frameHeight)

frame = 0
baseTime = time.time()
prevTime = 0
kernel = np.ones((5, 5), np.uint8)

strRuntime = ""
strFPS     = ""


while True:
    success, img = vid.read()

    imgBlank    = np.zeros((100, 100), np.uint8)
    imgGray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur     = cv2.GaussianBlur(imgGray, (7, 7), 0)
    imgCanny    = cv2.Canny(imgBlur, 100, 200)
    imgDilation = cv2.dilate(imgCanny, kernel, iterations=2)
    imgEroded   = cv2.erode(imgDilation, kernel, iterations=2)

    imgStack = ftn.stackImages(0.8, ([img, imgGray, imgBlur],
                                     [imgCanny, imgDilation, imgEroded]))

    frame, prevTime, strFPS = ftn.videoText(imgStack, frame, baseTime, prevTime, strRuntime, strFPS)

    cv2.imshow('Result', imgStack)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

vid.release()
cv2.destroyAllWindows()

