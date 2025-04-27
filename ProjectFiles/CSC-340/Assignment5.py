import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('ProjectFiles/CSC-340/Media/waterfall1.jpg', 0)  # queryImage
img2 = cv2.imread('ProjectFiles/CSC-340/Media/waterfall2.jpg', 0)  # trainImage

# Check if the images are loaded correctly
if img1 is None or img2 is None:
    print("Error: One or both images could not be loaded.")
    exit(1)

# Step 1: Detect keypoints and descriptors using SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default parameters
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test (Lowe's ratio test)
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])

# Draw matches
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
plt.imshow(img3), plt.show()

# Extract points from the good matches
pts1 = np.zeros((len(good), 2), np.float32)
pts2 = np.zeros((len(good), 2), np.float32)

for i in range(len(good)):
    pts1[i] = kp1[good[i][0].queryIdx].pt
    pts2[i] = kp2[good[i][0].trainIdx].pt

# Variables for tracking best homography
bestH = None
best_error = float('inf')

# RANSAC iterations
iterations = 1000
for i in range(iterations):
    # Step 2a: Pick 4 random matches
    indices = np.random.choice(len(good), 4, replace=False)
    pts1_sample = pts1[indices]
    pts2_sample = pts2[indices]

    # Step 2b: Construct the 8x9 matrix A
    A = []
    for j in range(4):
        x1, y1 = pts1_sample[j]
        x2, y2 = pts2_sample[j]
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])

    A = np.array(A)

    # Step 2c: Perform SVD on A
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]  # Last row of Vt is the solution for h

    # Step 2d: Reformat into a 3x3 homography matrix
    H = h.reshape(3, 3)

    # Step 2e: Evaluate quality of H
    total_error = 0
    for j in range(len(good)):
        pt1_homogeneous = np.array([pts1[j][0], pts1[j][1], 1])
        pt2_estimated_homogeneous = np.dot(H, pt1_homogeneous)
        pt2_estimated = pt2_estimated_homogeneous[:2] / pt2_estimated_homogeneous[2]

        # Compute Euclidean distance between pt2_estimated and actual pt2
        pt2_actual = pts2[j]
        distance = np.linalg.norm(pt2_estimated - pt2_actual)
        total_error += distance

    # Step 2f: Check if current homography is the best
    if total_error < best_error:
        best_error = total_error
        bestH = H

    # Optional: print progress every 100 iterations
    if (i + 1) % 100 == 0:
        print(f"Iteration {i + 1}/{iterations}, Error: {total_error}")

# Once RANSAC is complete, we use the best homography to align the images
if bestH is not None:
    print("Best Homography Matrix:\n", bestH)

    # Step 3: Warp the images using the best homography
    img1_aligned = cv2.warpPerspective(img1, bestH, (img2.shape[1], img2.shape[0]))

    # Display the aligned images
    plt.subplot(121), plt.imshow(img1, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(img1_aligned, cmap='gray'), plt.title('Aligned Image')
    plt.show()
else:
    print("Homography calculation failed.")