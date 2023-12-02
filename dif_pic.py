import cv2
import numpy as np

def find_differences(image1_path, image_path2):
	img1=cv2.imread(image1_path)
	img2=cv2.imread(image2_path)

	if img1.shape != img2.shape:
		img2=cv2.resize(img2, (img1.shape[1], img1.shape[0]))

	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	hist1 = cv2.calcHist([gray1], [0], None, [256], [0,256])
	hist2 = cv2.calcHist([gray2], [0], None, [256], [0,256])

	similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

	if similarity<0.95:
		print("image is different")
		diff_image = cv2.absdiff(gray1, gray2)
		_, thresh = cv2.threshold(diff_image, 30, 256, cv2.THRESH_BINARY
