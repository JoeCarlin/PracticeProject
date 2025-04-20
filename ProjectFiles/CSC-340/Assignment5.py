import numpy as np
import cv2
import random

def compute_homography(pts1, pts2):
    A = []
    for i in range(pts1.shape[0]):
        x, y = pts1[i][0], pts1[i][1]
        x_p, y_p = pts2[i][0], pts2[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x * x_p, y * x_p, x_p])
        A.append([0, 0, 0, -x, -y, -1, x * y_p, y * y_p, y_p])
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape((3, 3))
    return H / H[2, 2]

def ransac_homography(pts1, pts2, iterations=50000, threshold=5.0):
    max_inliers = 0
    best_H = None

    for _ in range(iterations):
        idx = random.sample(range(len(pts1)), 4)
        sample_pts1 = pts1[idx]
        sample_pts2 = pts2[idx]

        try:
            H = compute_homography(sample_pts1, sample_pts2)
        except np.linalg.LinAlgError:
            continue

        inliers = 0
        for i in range(len(pts1)):
            pt1 = np.append(pts1[i], 1)
            projected = H @ pt1
            projected /= projected[2]
            dist = np.linalg.norm(pts2[i] - projected[:2])
            if dist < threshold:
                inliers += 1

        if inliers > max_inliers:
            max_inliers = inliers
            best_H = H

    return best_H

# Load grayscale images
img1 = cv2.imread('ProjectFiles/CSC-340/Media/waterfall1.jpg', 0)
img2 = cv2.imread('ProjectFiles/CSC-340/Media/waterfall2.jpg', 0)

# SIFT feature detection
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Feature matching
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]

# Extract matched point coordinates
pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

# Custom RANSAC Homography
H_custom = ransac_homography(pts1, pts2, iterations=50000, threshold=5.0)

# OpenCV Homography for comparison
H_opencv, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

print("Custom H matrix (with H[2,2] = 1):\n", H_custom)
print("\nOpenCV H matrix:\n", H_opencv)