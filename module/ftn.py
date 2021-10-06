# Reference : https://youtu.be/Wv0PSs0dmVI
import cv2
import numpy as np
import time


def stackImages(scale, imgArray):
    arr = imgArray
    rows = len(arr)
    cols = len(arr[0])
    rowsAvailable = isinstance(imgArray[0], list)
    
    if (rowsAvailable):
        for x in range(0, rows):
            for y in range(0, cols):
                imgFirst = arr[0][0]
                imgSize  = (imgFirst.shape[1], imgFirst.shape[0])

                if (arr[x][y].shape[:2] == arr[0][0].shape[:2]):
                    arr[x][y] = cv2.resize(arr[x][y], (0, 0), None, scale, scale, interpolation=cv2.INTER_LINEAR)

                else:
                    arr[x][y] = cv2.resize(arr[x][y], (imgSize[0], imgSize[1]), None, scale, scale, interpolation=cv2.INTER_LINEAR)
                
                if (len(arr[x][y].shape) == 2):
                    arr[x][y] = cv2.cvtColor(arr[x][y], cv2.COLOR_GRAY2BGR)

        imgBox = np.zeros((imgSize[0], imgSize[1], 3), np.uint8)
        axisHor  = [imgBox] * rows

        for x in range(0, rows):
            axisHor[x] = np.hstack(arr[x])

        axisVer = np.vstack(axisHor)

    else:
        for x in range(0, rows):
            imgFirst = arr[0]
            imgSize = (imgFirst.shape[1], imgFirst.shape[0])

            if (arr[x].shape[:2] == arr[0].shape[:2]):
                arr[x] = cv2.resize(arr[x], (0, 0), None, scale, scale, interpolation=cv2.INTER_LINEAR)

            else:
                arr[x] = cv2.resize(arr[x], (imgSize[0], imgSize[1]), None, scale, scale, interpolation=cv2.INTER_LINEAR)

            if (len(arr[x].shape) == 2):
                arr[x] = cv2.cvtColor(arr[x], cv2.COLOR_GRAY2BGR)

        axisHor = np.hstack(arr)
        axisVer = axisHor

    return axisVer


def videoText(img, f, base, prev, sRun, sFPS):
    f += 1
    currTime = time.time()
    runTime  = currTime - base
    timeDiff = currTime - prev

    sRun = "Runtime : %d" % runTime

    if (timeDiff > 0.3):
        fps = f / timeDiff
        sFPS = "FPS : %3.1f" % fps
        f = 0
        prev = currTime

    cv2.putText(img, sRun, (5, 20), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 0, 0), 2)
    cv2.putText(img, sFPS, (5, 40), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 0, 0), 2)

    return f, prev, sFPS
