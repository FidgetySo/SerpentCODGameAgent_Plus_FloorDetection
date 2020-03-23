import numpy as np
import cv2
from mss import mss
from PIL import Image
#Define im as frame
while True:
    with mss() as sct:
        monitor_var = sct.monitors[1]
        monitor = np.array(sct.grab(monitor_var))
    gray = cv2.cvtColor(cv2.UMat(monitor), cv2.COLOR_RGB2GRAY)
    areaArray = []
    count = 1

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    #first sort the array by area
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

    #find the nth largest contour [n-1][1], in this case 2
    largestcontour = sorteddata[0][1]

    #draw it
    x, y, w, h = cv2.boundingRect(largestcontour)
    cv2.drawContours(gray, largestcontour, -1, (255, 0, 0), 2)
    cv2.rectangle(gray, (x, y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow("hey you!", gray)
    cv2.waitKey(100)