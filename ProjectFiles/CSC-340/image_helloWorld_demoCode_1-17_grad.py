import cv2
import numpy as np

# Read in the image (color and grayscale)
image = cv2.imread("ProjectFiles/CSC-340/Media/cones1.png")
image_grayscale = cv2.imread("ProjectFiles/CSC-340/Media/cones1.png", 0)

# Get the size of the original image
numRows = image.shape[0]  # Height of the image
numCols = image.shape[1]  # Width of the image
print("Original size: ", numRows, numCols)

# Create a new larger image (double dimensions)
newRows, newCols = numRows * 2, numCols * 2
largerImage = np.zeros((newRows, newCols, 3), dtype=np.uint8)  # Larger blank image for color

# Copy the original image to the center of the larger image
for i in range(numRows):
    for j in range(numCols):
        largerImage[i + numRows // 2, j + numCols // 2] = image[i, j]

# Gradient along the Y-axis
emptyIm_grayScale = np.zeros((numRows, numCols), np.float32)
for i in range(1, numRows - 1):
    for j in range(1, numCols - 1):
        # Calculate Y-gradient
        emptyIm_grayScale[i, j] = abs(float(image_grayscale[i + 1, j]) - float(image_grayscale[i - 1, j]))

# Normalize gradient for better visibility
emptyIm_grayScale = cv2.normalize(emptyIm_grayScale, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Create a bordered gradient image (place gradient in the center of a larger grayscale image)
largerGradient = np.zeros((newRows, newCols), dtype=np.uint8)  # Larger blank grayscale image
for i in range(numRows):
    for j in range(numCols):
        largerGradient[i + numRows // 2, j + numCols // 2] = emptyIm_grayScale[i, j]

# Save outputs
cv2.imwrite("ProjectFiles/CSC-340/Media/scaled_with_border.png", largerImage)              # Scaled image with border
cv2.imwrite("ProjectFiles/CSC-340/Media/gradient_with_border.png", largerGradient)         # Gradient image with border
cv2.imwrite("ProjectFiles/CSC-340/Media/gradient_only.png", emptyIm_grayScale)             # Gradient image only

# Display all images
cv2.imshow("Original Image", image)
cv2.imshow("Scaled Image with Border", largerImage)
cv2.imshow("Gradient Image", emptyIm_grayScale)
cv2.imshow("Gradient Image with Border", largerGradient)

cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()  # Close all OpenCV windows