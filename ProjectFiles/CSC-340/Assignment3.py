import numpy as np
import cv2

# Load grayscale image
image = cv2.imread("ProjectFiles/CSC-340/Media/checkerboard.png", cv2.IMREAD_GRAYSCALE)
height, width = image.shape

# Initialize empty images for gradients and second moment matrix components
Ix = np.zeros((height, width))
Iy = np.zeros((height, width))
Ixx = np.zeros((height, width))
Iyy = np.zeros((height, width))
Ixy = np.zeros((height, width))

# Compute gradients using Sobel operator
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

for y in range(1, height - 1):
    for x in range(1, width - 1):
        region = image[y - 1:y + 2, x - 1:x + 2]
        Ix[y, x] = np.sum(region * sobel_x)
        Iy[y, x] = np.sum(region * sobel_y)
        Ixx[y, x] = Ix[y, x] ** 2
        Iyy[y, x] = Iy[y, x] ** 2
        Ixy[y, x] = Ix[y, x] * Iy[y, x]

# Create an empty image to hold "cornerness" values
cornerness = np.zeros((height, width))

# Loop through the pixels of the image to compute the "cornerness" value
for y in range(1, height - 1):
    for x in range(1, width - 1):
        Sxx = np.sum(Ixx[y - 1:y + 2, x - 1:x + 2])
        Syy = np.sum(Iyy[y - 1:y + 2, x - 1:x + 2])
        Sxy = np.sum(Ixy[y - 1:y + 2, x - 1:x + 2])
        
        # Compute the determinant and trace of the M matrix
        detM = Sxx * Syy - Sxy ** 2
        traceM = Sxx + Syy
        
        # Compute the "cornerness" value
        cornerness[y, x] = detM - 0.04 * (traceM ** 2)

# Identify corners by thresholding the cornerness values
threshold = 0.01 * cornerness.max()
corners = np.zeros_like(image)
corners[cornerness > threshold] = 255

# Draw red dots on the original image where corners are detected
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for y in range(height):
    for x in range(width):
        if corners[y, x] == 255:
            cv2.circle(output_image, (x, y), 1, (0, 0, 255), -1)

# Visualize the result
cv2.imshow('Corners', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()