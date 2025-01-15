import cv2
import numpy as np

# read in an image in opencv
image = cv2.imread("/Users/joecarlin30/Desktop/School/CSC-331/Project/PythonProjects/ProjectFiles/Media/cones1.png")

# get size of image
numRows = image.shape[0]
numCols = image.shape[1]
print("Number of rows: ", numRows)
print("Number of columns: ", numCols)

# create a second image of the same size of the first image
emptyIm = np.zeros((numRows, numCols, 3), np.float32)

# itterate over all pixels
for i in range(numRows):
    for j in range(numCols):
        # copy blue channel into empty image
       emptyIm[i][j][0] = image[i][j][0]

# display the image
cv2.imshow("Image", emptyIm / 255.0)

cv2.waitKey(0) # wait for a key press "enter"
cv2.destroyAllWindows() # close the window
