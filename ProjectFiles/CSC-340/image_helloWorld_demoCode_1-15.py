import cv2
import numpy as np

# read in an image in opencv
image = cv2.imread("ProjectFiles/CSC-340/Media/cones1.png")
# if image is in a subfolder, give relative path
#image = cv2.imread("img/cones1.png")

# get size of an image
numRows = image.shape[0] # height of image
numCols = image.shape[1] # width of image
print("size: ", numRows, numCols)

# create a second image, of the same size as the first
emptyIm = np.zeros( (numRows, numCols, 3), np.float32)


# iterate over all the pixels in the image
for i in range(numRows): # height of the image, y coordinates
    for j in range(numCols): # width of the image, x coordinates


        avgInt = (0.082 * float(image[i][j][0]) + 0.609 * float(image[i][j][1]) + 0.309 * float(image[i][j][2]))
        # copy the blue channel into empty image
        emptyIm[i][j][0] = avgInt

        # copy the green channel into empty image
        emptyIm[i][j][1] = avgInt

        # copy the green channel into empty image
        emptyIm[i][j][2] = avgInt

        # image[i][j][0]: blue channel at location row i, col j
        # image[i][j][1]: green channel at location row i, col j
        # image[i][j][2]: red channel at location row i, col j
    



# display an image
cv2.imshow("Displaying an image", emptyIm/255.0)

# save an image
cv2.imwrite("ProjectFiles/CSC-340/Media/savedImg_sca.png", emptyIm)


cv2.waitKey(0) # not going to proceed until you hit "enter"
cv2.destroyAllWindows() # closes all windows opened with "imshow"
