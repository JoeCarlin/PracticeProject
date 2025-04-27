import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load images
img1 = cv2.imread('ProjectFiles/CSC-340/Media/mountainLeft.jpg', 0)
img2 = cv2.imread('ProjectFiles/CSC-340/Media/mountainRight.jpg', 0)

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
pts1 = []
pts2 = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.array(pts1)
pts2 = np.array(pts2)

# Draw matches for visualization
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, [[g] for g in good], None, flags=2)
plt.imshow(img3)
plt.title('Good Matches')
plt.show()

# Function to compute homography from 4 point pairs
def compute_homography(pts1, pts2):
    A = []
    for i in range(4):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H

# RANSAC implementation
def ransac(pts1, pts2, iterations=3000):
    best_total_distance = float('inf')
    best_H = None
    num_points = len(pts1)

    for _ in range(iterations):
        # Randomly pick 4 matches
        indices = np.random.choice(num_points, 4, replace=False)
        pts1_sample = pts1[indices]
        pts2_sample = pts2[indices]

        # Compute candidate homography
        H_candidate = compute_homography(pts1_sample, pts2_sample)

        # Calculate sum of distances for all points
        total_distance = 0
        for i in range(num_points):
            x1, y1 = pts1[i]
            x2, y2 = pts2[i]

            pt1_homog = np.array([x1, y1, 1.0])
            projected = H_candidate @ pt1_homog
            projected /= projected[2]  # Normalize homogeneous coordinate

            u, v = projected[0], projected[1]
            distance = np.sqrt((u - x2)**2 + (v - y2)**2)

            total_distance += distance

        # Keep the H with the lowest error
        if total_distance < best_total_distance:
            best_total_distance = total_distance
            best_H = H_candidate

    return best_H

# Run RANSAC to find best homography
H_ransac = ransac(pts1, pts2)

# Normalize H so H[2,2] = 1
H_ransac /= H_ransac[2,2]

print("Custom RANSAC Homography matrix:")
print(H_ransac)

# Now compare with OpenCV's built-in RANSAC
opencv_H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

print("\nOpenCV findHomography matrix:")
print(opencv_H)

# Step 3: Compute panorama dimensions
def compute_panorama_dimensions(image1, image2, bestH):
    # Get image dimensions
    width1, height1 = image1.shape[1], image1.shape[0]
    width2, height2 = image2.shape[1], image2.shape[0]
    
    # Define corners of image1 in homogeneous coordinates
    corners_image1 = np.array([
        [0, 0, 1],
        [width1, 0, 1],
        [0, height1, 1],
        [width1, height1, 1]
    ])
    
    # Project corners of image1 onto image2
    projected_corners = np.dot(bestH, corners_image1.T).T
    projected_corners /= projected_corners[:, 2:3]  # Homogeneous division
    
    # Combine corners of image1 and image2 to find panorama bounds
    minX = min(np.min(projected_corners[:, 0]), 0)
    maxX = max(np.max(projected_corners[:, 0]), width2)
    minY = min(np.min(projected_corners[:, 1]), 0)
    maxY = max(np.max(projected_corners[:, 1]), height2)
    
    # Panorama dimensions
    panorama_width = int(maxX - minX)
    panorama_height = int(maxY - minY)
    
    return panorama_width, panorama_height, minX, minY

# Step 4: Stitch images using bestH
def stitch_images(image1, image2, bestH):
    # Get dimensions of image2
    height2, width2 = image2.shape[:2]

    # Compute the panorama size and offsets
    panorama_width, panorama_height, minX, minY = compute_panorama_dimensions(image1, image2, bestH)
    
    # Create a new empty panorama image
    panorama = np.zeros((panorama_height, panorama_width), dtype=np.uint8)
    
    # Convert minX, minY to integers
    minX, minY = int(minX), int(minY)
    
    # Copy image2 into the panorama (shift by minX, minY if necessary)
    panorama[-minY:height2-minY, -minX:width2-minX] = image2
    
    # Iterate over all pixels in the panorama
    for y in range(panorama_height):
        for x in range(panorama_width):
            # Map panorama pixel (x, y) to image1 using inverse homography
            pt2 = np.array([x + minX, y + minY, 1])  # Shifting by minX, minY
            pt1 = np.dot(np.linalg.inv(bestH), pt2)  # Inverse homography transformation
            pt1 /= pt1[2]  # Homogeneous division
            
            # Check if pt1 is within bounds of image1
            x1, y1 = int(pt1[0]), int(pt1[1])
            if 0 <= x1 < image1.shape[1] and 0 <= y1 < image1.shape[0]:
                panorama[y, x] = image1[y1, x1]  # Copy pixel from image1 to panorama
    
    return panorama

# Stitch the images and show the result
result = stitch_images(img1, img2, H_ransac)

# Show the resulting panorama
plt.imshow(result, cmap='gray')
plt.title('Stitched Panorama')
plt.show()