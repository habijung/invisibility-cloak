import cv2
import numpy as np
import ftn

def onChange(pos):
    pass

src = "car-lime.jpg"


''' Create HSV Trackbar '''
title = "HSV Detector"
hsvParameter  = ["Hue Min", "Hue Max", "Sat Min", "Sat Max", "Val Min", "Val Max"]
hsvInit       = [42, 63, 96, 255, 145, 255]
cv2.namedWindow(title)
cv2.resizeWindow(title, 500, 400)

for i in range(6):
    if (i < 2):
        cv2.createTrackbar(hsvParameter[i], title, hsvInit[i], 179, lambda x: x)
    else: 
        cv2.createTrackbar(hsvParameter[i], title, hsvInit[i], 179, lambda x: x)


''' Show Result '''
while cv2.waitKey(1) != ord('q'):

    img = cv2.imread(src)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ''' Get Trackbar Position '''
    for i in range(6):
        hsvInit[i] = cv2.getTrackbarPos(hsvParameter[i], title)

    lower = np.array([hsvInit[0], hsvInit[2], hsvInit[4]])
    upper = np.array([hsvInit[1], hsvInit[3], hsvInit[5]])

    imgMask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=imgMask)
    imgStack = ftn.stackImages(0.7, ([img, imgHSV], [imgMask, imgResult]))

    cv2.imshow(title, imgStack)

cv2.destroyAllWindows()
