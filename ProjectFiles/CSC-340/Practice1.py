import cv2
import numpy as np

# load and image in grayscale
img = cv2.imread("ProjectFiles/CSC-340/Media/cones1.png", -1)

# resize the image
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

# display the image 
cv2.imshow("Image", img)
# wait for a key press
cv2.waitKey(0)
# destroy the window
cv2.destroyAllWindows()