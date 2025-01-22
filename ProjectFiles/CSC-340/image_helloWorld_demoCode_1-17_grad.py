import cv2
import numpy as np

# read in an image in opencv
image = cv2.imread("ProjectFiles/CSC-340/Media/cones1.png")
image_grayscale = cv2.imread("ProjectFiles/CSC-340/Media/cones1.png", 0)
# if image is in a subfolder, give relative path
#image = cv2.imread("img/cones1.png")

# get size of an image
numRows = image.shape[0] # height of image
numCols = image.shape[1] # width of image
print("size: ", numRows, numCols)

# create a second image, of the same size as the first
emptyIm = np.zeros( (numRows, numCols, 3), np.float32)
emptyIm_grayScale = np.zeros( (numRows, numCols), np.float32)


# iterate over all the pixels in the image
for i in range(1,numRows-1): # height of the image, y coordinates
    for j in range(1,numCols-1): # width of the image, x coordinates

            # convert to grayscale, preserve intensity
            ## avgInt =  0.082 * float(image[i][j][0]) + 0.609 *float(image[i][j][1]) + 0.309 * float(image[i][j][2])
##        # copy the blue channel into empty image
##        emptyIm[i][j][0] = avgInt
##
##        # copy the green channel into empty image
##        emptyIm[i][j][1] = avgInt
##
##        # copy the green channel into empty image
##        emptyIm[i][j][2] = avgInt
##
##


        # invert your RGB values
        avgInt =  0.082 * float(image[i][j][0]) + 0.609 *float(image[i][j][1]) + 0.309 * float(image[i][j][2])
        # copy the blue channel into empty image
        emptyIm[i][j][0] = 255 - image[i][j][0]

        # copy the green channel into empty image
        emptyIm[i][j][1] = 255 - image[i][j][1]

        # copy the green channel into empty image
        emptyIm[i][j][2] = 255 - image[i][j][2]

        # image[i][j][0]: blue channel at location row i, col j
        # image[i][j][1]: green channel at location row i, col j
        # image[i][j][2]: red channel at location row i, col j
    
        emptyIm[i][j]=abs(float(image_grayscale[i+1][j])-float(image_grayscale[i-1][j]))


# display an image
cv2.imshow("Displaying an image", emptyIm/255.0)

# save an image
cv2.imwrite("savedImg_gradY.png", emptyIm)


cv2.waitKey(0) # not going to proceed until you hit "enter"
cv2.destroyAllWindows() # closes all windows opened with "imshow"
