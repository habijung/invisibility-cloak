import cv2
import numpy as np
import ftn


path = "car-lime.jpg"
img = cv2.imread(path)
kernel = np.ones((5, 5), np.uint8)

imgBlank    = np.zeros((100, 100), np.uint8)
imgGray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur     = cv2.GaussianBlur(imgGray, (7, 7), 0)
imgCanny    = cv2.Canny(imgBlur, 100, 200)
imgDilation = cv2.dilate(imgCanny, kernel, iterations=2)
imgEroded   = cv2.erode(imgDilation, kernel, iterations=2)

imgStack = ftn.stackImages(0.8, ([img, imgGray, imgBlur],
                                 [imgCanny, imgDilation, imgEroded]))

cv2.imshow('Result', imgStack)
cv2.waitKey(0)
