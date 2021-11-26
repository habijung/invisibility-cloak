import cv2
import numpy as np
import sys
import time
from modules import ftn


# Functions
def kernel(n):
    return np.ones((n, n), np.uint8)


def init():
    pyVersion = ""

    for i in range(3):
        pyVersion += (str(sys.version_info[i]))

        if (i != 2):
            pyVersion += "."

    strFormat = "%-13s%-35s%2s"
    print("=" * 50)
    print("%s%48s" % ("==", "=="))
    print(strFormat % ("==", "Invisible Cloak Project", "=="))
    print(strFormat % ("==", "Project ver. : v1.3", "=="))
    print(strFormat % ("==", "Python ver.  : " + pyVersion, "=="))
    print(strFormat % ("==", "OpenCV ver.  : " + cv2.__version__, "=="))
    print("%s%48s" % ("==", "=="))
    print("=" * 50)


def main(argv):
    # Variables
    camDevice = 0
    background = 0
    hsvTitle = "HSV Detector"
    hsvParameter  = ["Hue Min", "Hue Max", "Sat Min", "Sat Max", "Val Min", "Val Max"]
    hsvValue = [0, 13, 80, 255, 30, 255] # red color
    
    frame = 0
    baseTime = time.time()
    prevTime = 0
    strRun = ""
    strFPS = ""
    color = ''

    # Open video.
    video = cv2.VideoCapture(camDevice)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    resolution = (640, 480)
    out = cv2.VideoWriter('capture_color.mp4', fourcc, 30.0, resolution)
    out_mask = cv2.VideoWriter('capture_mask.mp4', fourcc, 30.0, resolution)
    out_result = cv2.VideoWriter('capture_result.mp4', fourcc, 30.0, resolution)

    # Capture background image.
    time.sleep(2)
    for i in range(30):
        ret, background = video.read()
    
    background = np.flip(background, axis=1)

    # Create HSV Trackbar.
    cv2.namedWindow(hsvTitle)
    cv2.resizeWindow(hsvTitle, 600, 400)

    for i in range(6):
        if i < 2: # Hue Min, Max
            cv2.createTrackbar(hsvParameter[i], hsvTitle, hsvValue[i], 179, lambda x: x)

        else: # Saturation & Value Min, Max
            cv2.createTrackbar(hsvParameter[i], hsvTitle, hsvValue[i], 255, lambda x: x)
    

    # If capture failed to open, try again.
    if not video.isOpened():
        print("Video open failed. Try again.")
        video.open(camDevice)


    # Only attemp to read if video is opened.
    if video.isOpened:
        while True:
            ret, img = video.read()

            img = np.flip(img, axis=1)
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # imgBlur = cv2.GaussianBlur(imgHSV, (35, 35), 0)

            # Get trackbar position.
            for i in range(6):
                hsvValue[i] = cv2.getTrackbarPos(hsvParameter[i], hsvTitle)
            
            # lower -> min values && upper -> max values
            color = 'r'
            if color == 'r':
                # Color : Red only
                # hsvValue = [0, 10, 120, 255, 50, 255]
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

            # Set mask image.
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel(3), iterations=3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel(7), iterations=3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel(3), iterations=4)
            # img[np.where(mask == 255)] = background[np.where(mask == 255)]
            
            # Mask overlapping.
            # res1 -> background ∩ mask | res2 -> img - mask
            mask_inv = cv2.bitwise_not(mask)
            res1 = cv2.bitwise_and(background, background, mask=mask)
            res2 = cv2.bitwise_and(img, img, mask=mask_inv)

            imgBlank = np.zeros((100, 100), np.uint8)
            imgResult = cv2.addWeighted(res1, 1, res2, 1, 0)
            imgStack  = ftn.stackImages(0.7, ([img, imgHSV, mask], [res1, res2, imgResult]))


            # Only display the image if it is not empty
            if ret:
                frame, prevTime, strFPS = ftn.videoText(imgStack, frame, baseTime, prevTime, strRun, strFPS)
                cv2.imshow("Result", imgStack)

                out.write(img)
                out_mask.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
                out_result.write(imgResult)

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
        out_mask.release()
        out_result.release()
        cv2.destroyAllWindows()

    # If video.isOpened == false
    else:
        print("Failed to open capture device.")
    


# start main()
if __name__ == "__main__":
    init()
    main(sys.argv)
