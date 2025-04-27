import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load images
img1 = cv2.imread('ProjectFiles/CSC-340/Media/waterfall1.jpg', 0)  # queryImage
img2 = cv2.imread('ProjectFiles/CSC-340/Media/waterfall2.jpg', 0)  # trainImage

# Initiate SIFT detector
sift = cv2.SIFT_create()

# Step 1: Find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test to filter good matches
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])

# Draw matches
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
plt.imshow(img3)
plt.title('Good Matches')
plt.show()

# Prepare points arrays for homography computation
pts1 = np.zeros((len(good), 2), np.float32)
pts2 = np.zeros((len(good), 2), np.float32)
for m in range(len(good)):
    pts1[m] = kp1[good[m][0].queryIdx].pt
    pts2[m] = kp2[good[m][0].trainIdx].pt

# Step 2: Run RANSAC to find best homography
def ransac(pts1, pts2, iterations=3000, threshold=5.0):
    best_inliers = 0
    best_H = None
    num_points = len(pts1)
    best_sum_distances = float('inf')  # Initialize best error sum

    for i in range(iterations):
        # Step 2.1: Pick 4 random matches from good list
        indices = np.random.choice(num_points, 4, replace=False)
        pts1_sample = pts1[indices]
        pts2_sample = pts2[indices]

        # Step 2.2: Construct the A matrix and run SVD
        A = []
        for i in range(4):
            x1, y1 = pts1_sample[i]
            x2, y2 = pts2_sample[i]
            A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
            A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
        A = np.array(A)

        # Run SVD on A
        _, _, V = np.linalg.svd(A)
        hCol = V[8, :]
        H_candidate = hCol.reshape(3, 3)

        # Step 2.3: Evaluate the quality of the homography by checking inliers
        sum_distances = 0
        inliers = 0
        for i in range(num_points):
            x1, y1 = pts1[i]
            x2, y2 = pts2[i]

            pt1_homog = np.array([x1, y1, 1.0])
            projected = H_candidate @ pt1_homog
            projected /= projected[2]  # Normalize by the homogeneous coordinate

            u, v = projected[0], projected[1]
            distance = np.sqrt((u - x2)**2 + (v - y2)**2)
            sum_distances += distance

            if distance < threshold:
                inliers += 1

        # Step 2.4: If the current homography has a better sum of distances, keep it
        if sum_distances < best_sum_distances:
            best_sum_distances = sum_distances
            best_inliers = inliers
            best_H = H_candidate

        # Print debug information (can be commented out for performance)
        print(f"Iteration {i+1}/{iterations}, Inliers: {inliers}, Sum of Distances: {sum_distances}")

    return best_H

# Step 3: Run RANSAC to get the best homography
best_H = ransac(pts1, pts2, iterations=3000, threshold=5.0)

# Normalize the homography
best_H /= best_H[2, 2]

# Warp img1 to img2's perspective using the best homography
height, width = img2.shape
img1_warped = cv2.warpPerspective(img1, best_H, (width, height))

# Step 4: Show the original and warped images
plt.subplot(121), plt.imshow(img2, cmap='gray')
plt.title('Image 2 (Train)'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img1_warped, cmap='gray')
plt.title('Warped Image 1'), plt.xticks([]), plt.yticks([])
plt.show()

# Optionally, compare with OpenCV's result for debugging
opencv_H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
print("H matrix estimated by OpenCV (for comparison):\n", opencv_H)