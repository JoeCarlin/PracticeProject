import numpy as np
import cv2

# Step 1: Load the original image
image = cv2.imread('ProjectFiles/CSC-340/Media/checkerboardSmall.png', cv2.IMREAD_GRAYSCALE)
height, width = image.shape

# Step 2: Create empty images for Ix, Iy, Ixx, Iyy, and Ixy
Ix = np.zeros((height, width), dtype=np.float32)
Iy = np.zeros((height, width), dtype=np.float32)
Ixx = np.zeros((height, width), dtype=np.float32)
Iyy = np.zeros((height, width), dtype=np.float32)
Ixy = np.zeros((height, width), dtype=np.float32)

# Step 3: Loop through the pixels of the image to compute Ix, Iy, Ixx, Iyy, and Ixy
for y in range(1, height-1):
    for x in range(1, width-1):
        Ix[y, x] = (image[y, x+1] - image[y, x-1]) / 2.0
        Iy[y, x] = (image[y+1, x] - image[y-1, x]) / 2.0
        Ixx[y, x] = Ix[y, x] ** 2
        Iyy[y, x] = Iy[y, x] ** 2
        Ixy[y, x] = Ix[y, x] * Iy[y, x]

# Step 4: Create an empty image to hold "cornerness" values
cornerness = np.zeros((height, width), dtype=np.float32)

# Step 5: Loop through the pixels of the image to compute the "cornerness" value
for y in range(1, height-1):
    for x in range(1, width-1):
        Sxx = np.sum(Ixx[y-1:y+2, x-1:x+2])
        Syy = np.sum(Iyy[y-1:y+2, x-1:x+2])
        Sxy = np.sum(Ixy[y-1:y+2, x-1:x+2])
        
        # Compute the determinant and trace of the M matrix
        detM = Sxx * Syy - Sxy ** 2
        traceM = Sxx + Syy
        
        # Compute the "cornerness" value
        cornerness[y, x] = detM - 0.04 * (traceM ** 2)

# Step 6: Identify corners by thresholding the cornerness values
threshold = 0.01 * cornerness.max()
corners = np.zeros_like(image)
corners[cornerness > threshold] = 255

# Step 7: Draw red dots on the original image where corners are detected
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for y in range(height):
    for x in range(width):
        if corners[y, x] == 255:
            cv2.circle(output_image, (x, y), 1, (0, 0, 255), -1)

# Step 8: Visualize the result
cv2.imshow('Corners', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()