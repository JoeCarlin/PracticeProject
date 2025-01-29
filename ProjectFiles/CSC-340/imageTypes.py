import cv2
import numpy as np

image1 = cv2.imread("cones1.png")

numRows = image1.shape[0]
numCols = image1.shape[1]  

image2 = np.zeros((numRows, numCols, 3), np.uint8)  

for i in range(numRows):
    for j in range(numCols):
        image2[i][j] = image1[i][j]
    

cv2.imshow("Image 2", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()  
