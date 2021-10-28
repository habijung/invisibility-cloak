# sample1.py
import cv2
import numpy as np
import time


## Preparation for writing the ouput video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

## Reading from the webcam
device = 0
cap = cv2.VideoCapture(device)

## Allow the system to sleep for 3s before the webcam starts
time.sleep(3)
count = 0
background = 0

## Capture the background in range of 60
for i in range(60):
    ret, background = cap.read()

background = np.flip(background, axis=1)

## Read every frame from the webcam, until the camera is open
while (cap.isOpened()):
    ret, img = cap.read()

    if not ret:
        break

    count += 1
    img = np.flip(img, axis=1)

    ## Convert the color space from BGR to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## Generate masks to detect red color
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask01 = cv2.inRange(imgHSV, lower_red, upper_red)

    lower_green = np.array([30, 70, 46])
    upper_green = np.array([70, 255, 255])
    mask02 = cv2.inRange(imgHSV, lower_green, upper_green)

    mask = mask01 + mask02
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    img[np.where(mask == 255)] = background[np.where(mask == 255)]

    ## Generating the Result and writing
    # out.write(img)
    cv2.imshow("Result", img)
    cv2.imshow("mask", mask)
    cv2.imshow("mask01", mask01)
    cv2.imshow("mask02", mask02)

    ## Stop & Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
