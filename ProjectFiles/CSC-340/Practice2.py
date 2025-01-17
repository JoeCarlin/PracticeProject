import cv2
import numpy as np

# load and image in grayscale
img = cv2.imread("ProjectFiles/CSC-340/Media/cones1.png", -1)

# Get size of the image (height and width)
numRows = img.shape[0]  # Height of the image
numCols = img.shape[1]  # Width of the image

# print the size of the image
print(f"The size of the image is {numRows} rows and {numCols} columns")