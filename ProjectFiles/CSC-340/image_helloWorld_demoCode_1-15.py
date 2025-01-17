import cv2
import numpy as np

# Read in an image using OpenCV
image = cv2.imread("ProjectFiles/CSC-340/Media/cones1.png")
# If image is in a subfolder, give relative path
# image = cv2.imread("img/cones1.png")

# Get size of the image (height and width)
numRows = image.shape[0]  # Height of the image
numCols = image.shape[1]  # Width of the image
print("Original size: ", numRows, numCols)

# Scaling factor based on original image dimensions
scale_factor = 2  # Example scaling factor, change as needed
new_numRows = int(numRows * scale_factor)
new_numCols = int(numCols * scale_factor)

# Create new larger image of the correct size 
# Create a new larger image with a black background (empty image)
emptyIm = np.zeros((new_numRows, new_numCols, 3), np.float32)

# Copy the original image into the center of the new image
for i in range(numRows):
    for j in range(numCols):
        # Place original image at the center of the new image
        emptyIm[i + (new_numRows - numRows) // 2][j + (new_numCols - numCols) // 2] = image[i, j]

# Gradient effect for one color channel (x-gradient for red) 
# Create a gradient effect along the x-direction for the red channel
for i in range(new_numRows):
    for j in range(new_numCols):
        emptyIm[i, j, 2] = (j / new_numCols) * 255  # Red channel (x-gradient)

#  Load, initialize, display, and save images correctly
cv2.imshow("New Image with Original in Center", emptyIm / 255.0)
cv2.imwrite("ProjectFiles/CSC-340/Media/scaled_with_gradient.png", emptyIm)

# Wait for a key press to close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()