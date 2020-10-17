#!/usr/bin/env python3

import cv2
import numpy as np


def blobdetect():
	
	path = r'/home/crystal/Cone/Blue_cone.jpg'

	src = cv2.imread(path)

	window_name = 'Image'
	#cv2.imshow(window_name,src)

	hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

	blueLower = (100, 150, 50)
	blueUpper = (150, 255, 255)

	whiteLower = (30, 0, 150)
	#whiteUpper = (180, 30, 255)
	whiteUpper = (196, 216, 255)

	orangeLower = (0, 100, 100)
	orangeUpper = (20, 255, 255)

	#'orange': [((0, 100, 100), (20, 255, 255)), ((160, 100, 100), (180, 255, 255))]
	
	yellowLower = (20, 100, 60)
	yellowUpper = (35, 255, 255)

	blackLower = (0, 0, 0)
	blackUpper = (180, 255, 50)

	colourLowerBounds = [whiteLower, blueLower, orangeLower, yellowLower, blackLower]
	colourUpperBounds = [whiteUpper, blueUpper, orangeUpper, yellowUpper, blackUpper]
	colours = ["white", "blue", "orange", "yellow", "black"]


	masks = []
	for x in range(0,5) :
		mask = cv2.inRange(hsv,colourLowerBounds[x],colourUpperBounds[x])
		mask = cv2.erode(mask, None, iterations = 2)
		mask = cv2.dilate(mask, None, iterations = 2)

		

		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0]

		masks.append(mask)

		print(colours[x])

		for c in cnts:
			M = cv2.moments(c)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
		
			cv2.drawContours(src, [c], -1, (0, 255, 0), 2)
			cv2.circle(src, (cX, cY), 7, (255, 255, 255), -1)
			cv2.putText(src, colours[x], (cX - 20, cY - 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

	cv2.imshow(window_name,src)
	cv2.imshow("Mask", mask)
	Masks = cv2.bitwise_or(cv2.bitwise_or(masks[0], masks[1]), cv2.bitwise_or(masks[2], cv2.bitwise_or(masks[3], masks[4])))
	cv2.imshow("Mask", Masks)
	MasksBytes = Masks.tostring()
	cv2.imwrite('Blue_cone_mask.png', Masks)
	#Masks = Masks.flatten()
	#MasksBytes = Masks.tobytes()
	#bitmap_string = MasksBytes.decode(encoding = 'utf-16')
	#print(bitmap_string)


if __name__ == '__main__':
	blobdetect()
	cv2.waitKey(0)
	cv2.destroyAllWindows
