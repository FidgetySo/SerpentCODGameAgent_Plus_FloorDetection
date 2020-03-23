import numpy as np
import cv2
from mss import mss
from PIL import Image
import imutils
#Define im as frame
def convolve(image, kernel):
	# grab the spatial dimensions of the image, along with
	# the spatial dimensions of the kernel
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]

	# allocate memory for the output image, taking care to
	# "pad" the borders of the input image so the spatial
	# size (i.e., width and height) are not reduced
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")

	# loop over the input image, "sliding" the kernel across
	# each (x, y)-coordinate from left-to-right and top to
	# bottom
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			# extract the ROI of the image by extracting the
			# *center* region of the current (x, y)-coordinates
			# dimensions
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

			# perform the actual convolution by taking the
			# element-wise multiplicate between the ROI and
			# the kernel, then summing the matrix
			k = (roi * kernel).sum()

			# store the convolved value in the output (x,y)-
			# coordinate of the output image
			output[y - pad, x - pad] = k

	# rescale the output image to be in the range [0, 255]
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")

	# return the output image
	return output
sharpen = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]), dtype="int")
dilation_size = 5
while True:
	with mss() as sct:
		monitor_var = sct.monitors[1]
		monitor = np.array(sct.grab(monitor_var))
	frame = cv2.resize(monitor, (1920, 1080))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	opencvOutput = cv2.filter2D(gray, -1, sharpen)
	morphingMatrix = np.ones((1, 1), dtype = "float32")
	opening = cv2.morphologyEx(opencvOutput, cv2.MORPH_OPEN, morphingMatrix)
	element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_size, dilation_size))
	dilation_1 = cv2.dilate(opening, element1)
	# create engine
	engine = cv2.hfs.HfsSegment_create(dilation_1.shape[0], dilation_1.shape[1])

	# perform segmentation
	# now "res" is a matrix of indices
	# change the second parameter to "True" to get a rgb image for "res"
	res = engine.performSegmentGpu(dilation_1, True)
	contours, _ = cv2.findContours(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	areaArray = []
	for i, c in enumerate(contours):
		area = cv2.contourArea(c)
		areaArray.append(area)
	sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
	largestcontour = sorteddata[1][1]
	x, y, w, h = cv2.boundingRect(largestcontour)
	img = cv2.drawContours(gray, largestcontour, -1, (255, 255, 255), 6)
	cv2.imshow("Image", img)
	cv2.waitKey(100)
