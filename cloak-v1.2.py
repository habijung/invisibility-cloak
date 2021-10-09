''' v1.2 '''
## Import statements (libraries)
import sys
import time
import cv2
import numpy as np
from module import ftn


## Functions
def init():
    pyVersion = ""

    for i in range(3):
        pyVersion += (str(sys.version_info[i]))

        if (i != 2):
            pyVersion += "."

    print("\n=====================================================")
    print("==                                                 ==")
    print("==         Invisible Cloak Project v1.2            ==")
    print("==         Python Version : " + pyVersion + "                 ==")
    print("==         OpenCV Version : " + cv2.__version__ + "                  ==")
    print("==                                                 ==")
    print("=====================================================\n")


def main(argv):
    
    ## Variables
    camDevice = 0
    background = 0
    hsvTitle = "HSV Detector"
    hsvParameter  = ["Hue Min", "Hue Max", "Sat Min", "Sat Max", "Val Min", "Val Max"]
    hsvValue = [0, 179, 0, 255, 0, 255]
    frame = 0
    baseTime = time.time()
    prevTime = 0
    strRun = ""
    strFPS = ""
    color = ''


    ## Video open
    video = cv2.VideoCapture(camDevice)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
    time.sleep(3)

    # Capture background image
    for i in range(30):
        ret, background = video.read()
    
    background = np.flip(background, axis=1)


    ## Create HSV Trackbar
    cv2.namedWindow(hsvTitle)
    cv2.resizeWindow(hsvTitle, 600, 400)

    for i in range(6):
        if i < 2: # Hue Min, Max
            cv2.createTrackbar(hsvParameter[i], hsvTitle, hsvValue[i], 179, lambda x: x)

        else: # Saturation & Value Min, Max
            cv2.createTrackbar(hsvParameter[i], hsvTitle, hsvValue[i], 255, lambda x: x)
    

    # If capture failed to open, try again
    if not video.isOpened():
        print("Video open failed. Try again.")
        video.open(camDevice)


    # Only attemp to read if video is opened
    if video.isOpened:
        while True:
            ret, img = video.read()

            img = np.flip(img, axis=1)
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # imgBlur = cv2.GaussianBlur(imgHSV, (35, 35), 0)

            ## Get trackbar position
            for i in range(6):
                hsvValue[i] = cv2.getTrackbarPos(hsvParameter[i], hsvTitle)
            
            # lower <- min values && upper <- max values
            if color == 'r':
                # Color : Red only
                hsvValue = [0, 10, 120, 255, 50, 255]
                lower = np.array([hsvValue[0], hsvValue[2], hsvValue[4]])
                upper = np.array([hsvValue[1], hsvValue[3], hsvValue[5]])
                mask1 = cv2.inRange(imgHSV, lower, upper)

                hsvValue = [170, 179, 120, 255, 70, 255]
                lower = np.array([hsvValue[0], hsvValue[2], hsvValue[4]])
                upper = np.array([hsvValue[1], hsvValue[3], hsvValue[5]])
                mask2 = cv2.inRange(imgHSV, lower, upper)

                mask = mask1 + mask2

            else:
                if color == 'g':
                    # Color : Green
                    hsvValue = [73, 100, 25, 255, 58, 255]
                    
                lower = np.array([hsvValue[0], hsvValue[2], hsvValue[4]])
                upper = np.array([hsvValue[1], hsvValue[3], hsvValue[5]])
                mask = cv2.inRange(imgHSV, lower, upper)

            # Set mask image
            mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
            mask_dilate = cv2.morphologyEx(mask_open, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)
            # mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
            # img[np.where(mask == 255)] = background[np.where(mask == 255)]
            
            mask_inv = cv2.bitwise_not(mask)

            res1 = cv2.bitwise_and(background, background, mask=mask_dilate)
            res2 = cv2.bitwise_and(img, img, mask=mask_inv)

            imgBlank = np.zeros((100, 100), np.uint8)
            imgResult = cv2.addWeighted(res1, 1, res2, 1, 0)


            imgStack1  = ftn.stackImages(0.7, ([img, mask, res1, res2], [mask_open, mask_dilate, imgResult, imgBlank]))
            # imgStack2  = ftn.stackImages(0.7, ([res1, res2], [imgResult, imgBlank]))


            # Only display the image if it is not empty
            if ret:
                frame, prevTime, strFPS = ftn.videoText(imgStack1, frame, baseTime, prevTime, strRun, strFPS)
                # frame, prevTime, strFPS = ftn.videoText(imgStack2, frame, baseTime, prevTime, strRun, strFPS)
                cv2.imshow("Result1", imgStack1)
                # cv2.imshow("Result2", imgStack2)

                out.write(imgResult)


            # If it is empty abort
            else:
                print("Error reading capture device.")
                break

            
            # Program exit with key(q)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elif cv2.waitKey(1) & 0xFF == ord('b'):
                print("Re-generate background.")

                for i in range(30):
                    ret, background = video.read()

                background = np.flip(background, axis=1)

            elif cv2.waitKey(1) & 0xFF == ord('r'):
                print("Color : RED")
                color = 'r'

            elif cv2.waitKey(1) & 0xFF == ord('g'):
                print("Color : Green")
                color = 'g'


        
        # End video
        video.release()
        out.release()
        cv2.destroyAllWindows()

    # If video.isOpened == false
    else:
        print("Failed to open capture device.")
    




# start main()
if __name__ == "__main__":
    init()
    main(sys.argv)
