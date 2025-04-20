import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load images
img1 = cv2.imread('ProjectFiles/CSC-340/Media/waterfall1.jpg', 0)  # queryImage
img2 = cv2.imread('ProjectFiles/CSC-340/Media/waterfall2.jpg', 0)  # trainImage

# Initiate SIFT detector
sift = cv2.SIFT_create()

# Find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])

# Draw matches
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
plt.imshow(img3)
plt.show()

# Store points for homography calculation
pts1 = np.zeros((len(good), 2), np.float32)
pts2 = np.zeros((len(good), 2), np.float32)
for m in range(len(good)):
    pts1[m] = kp1[good[m][0].queryIdx].pt
    pts2[m] = kp2[good[m][0].trainIdx].pt

# Implementing RANSAC
def ransac(pts1, pts2, iterations=20000, threshold=5.0):
    best_inliers_count = 0
    best_H = None
    num_points = len(pts1)

    for _ in range(iterations):
        # Randomly select 4 points
        indices = np.random.choice(num_points, 4, replace=False)
        pts1_sample = pts1[indices]
        pts2_sample = pts2[indices]

        # Compute homography using the selected points
        H = compute_homography(pts1_sample, pts2_sample)

        # Reproject all points and calculate inliers
        inliers = 0
        for i in range(num_points):
            pt1 = np.array([pts1[i][0], pts1[i][1], 1])
            pt2 = np.array([pts2[i][0], pts2[i][1], 1])
            pt1_transformed = np.dot(H, pt1)
            pt1_transformed /= pt1_transformed[2]  # Homogeneous coordinates normalization

            # Compute distance between the transformed point and the original point
            distance = np.linalg.norm(pt2[:2] - pt1_transformed[:2])

            if distance < threshold:
                inliers += 1

        # If the number of inliers is higher than the previous best, update the best homography
        if inliers > best_inliers_count:
            best_inliers_count = inliers
            best_H = H

    return best_H

# Function to compute homography from 4 point pairs
def compute_homography(pts1, pts2):
    A = []
    for i in range(4):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])

    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)

    return H

# Run RANSAC
H_ransac = ransac(pts1, pts2)

# Ensure the bottom right element is 1 (homogeneous coordinates)
H_ransac[2, 2] = 1

# Print the custom H matrix computed by RANSAC
print("Custom H matrix (with H[2,2] = 1):")
print(H_ransac)

# Now, use OpenCV to calculate the homography for comparison
opencv_H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
print("OpenCV H matrix:")
print(opencv_H)