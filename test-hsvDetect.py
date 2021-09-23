import cv2
import numpy as np
import ftn

def onChange(pos):
    pass

src = "car-lime.jpg"
barName  = ["Hue Min", "Hue Max", "Sat Min", "Sat Max", "Val Min", "Val Max"]
barVal   = [42, 63, 96, 255, 145, 255]
barMax   = [179, 179, 255, 255, 255, 255]

cv2.namedWindow("HSV Bar")
cv2.resizeWindow("HSV Bar", 500, 500)

for i in range(6):
    cv2.createTrackbar(barName[i], "HSV Bar", barVal[i], barMax[i], onChange)

while cv2.waitKey(1) != ord('q'):
    img = cv2.imread(src)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for i in range(6):
        barVal[i] = cv2.getTrackbarPos(barName[i], "HSV Bar")

    lower = np.array([barVal[0], barVal[2], barVal[4]])
    upper = np.array([barVal[1], barVal[3], barVal[5]])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)

    imgStack = ftn.stackImages(0.7, ([img, imgHSV], [mask, imgResult]))

    # cv2.imshow("Result", imgStack)
    cv2.imshow("HSV Bar", imgStack)

cv2.destroyAllWindows()
