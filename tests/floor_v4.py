import numpy as np
import cv2
from mss import mss
from PIL import Image
from pynput.mouse import Button, Controller
import pyautogui
mouse = Controller()
def set_pos_aimbot(x, y):
    mouse.position = (x, y)
#Define im as frame
y1= 0
x1= 0
y2= 1080
x2= 1920
while True:
    with mss() as sct:
        monitor_var = sct.monitors[1]
        monitor = np.array(sct.grab(monitor_var))
    gray = cv2.cvtColor(monitor, cv2.COLOR_BGR2GRAY) 
    blur = cv2.medianBlur(gray, 25)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,27,6)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilate = cv2.dilate(close, kernel, iterations=2)
    roi = dilate[y1:y2, x1:x2]
    roi2 = monitor[y1:y2, x1:x2]
    cnts = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    areaArray = []
    for i, c in enumerate(contours):
        area1 = cv2.contourArea(c)
        areaArray.append(area1)
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
    if len(sorteddata) == 0:
        pass
    else:
        largest = sorteddata[0][1]
        x1, y1, w1, h1 = cv2.boundingRect(largest)
        print("Area: X1: " + str(x1) + " Y1:" + str(y1) + " W:" + str(w1) + " H:" + str(h1))
        cv2.rectangle(roi2, (x1, y1), (x1+w1, y1+h1), (0,255,0), 2)
    minimum_area = 1000
    max_area = 9223372036854775
    for c in cnts:
        area = cv2.contourArea(c)
        if area > minimum_area and area < max_area:
            print(area)
            # Find centroid
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(roi2, (cX, cY), 20, (36, 255, 12), 2) 
            x,y,w,h = cv2.boundingRect(c)
            cv2.putText(roi2, 'Radius: {}'.format(w/2), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
            cv2.drawContours(roi2,c,-1,(255,255,0), 4)
    x_c, y_c, w_c, h_c = cv2.boundingRect(largest)
    center_X_area = x_c +(w_c / 2)
    center_Y_area = y_c +(h_c / 2)
    current_x_area, current_y_area = pyautogui.position()
    diff_cc_x_area = current_x_area - center_X_area
    diff_cc_y_area = current_y_area - center_Y_area
    if (diff_cc_x_area >= 375 and diff_cc_x_area >= 150) or (diff_cc_x_area <= -375 and diff_cc_x_area <= -150):
        set_pos_aimbot(center_X_area, center_Y_area)
    cv2.imshow("Image", roi2)
    cv2.waitKey(33)