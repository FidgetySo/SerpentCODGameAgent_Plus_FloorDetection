from mss import mss
import cv2
import imutils
import numpy as np
sct = mss()
while True:
	monitor_var = sct.monitors[1]
	monitor = np.array(sct.grab(monitor_var))
	monitor = cv2.UMat(monitor)
	gray = cv2.cvtColor(monitor, cv2.COLOR_RGB2GRAY)	
	cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	biggest_cnt = None
	biggest_cnt_size = 100.0
	for c in cnts:
		# compute the center of the contour
		M = cv2.moments(c)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		# draw the contour and center of the shape on the image
		cv2.drawContours(monitor, [c], -1, (0, 255, 0), 2)
		cv2.circle(monitor, (cX, cY), 7, (255, 255, 255), -1)
		cv2.putText(monitor, "center", (cX - 20, cY - 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		# show the image
	cv2.imshow("Image", monitor)
	cv2.waitKey(0)
