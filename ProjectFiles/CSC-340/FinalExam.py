import numpy as np
import cv2

# Load images in color
img1 = cv2.imread('ProjectFiles/CSC-340/Media/waterfall1.jpg')
img2 = cv2.imread('ProjectFiles/CSC-340/Media/waterfall2.jpg')

# Initiate SIFT detector
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Match descriptors
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Ratio test
good = []
pts1, pts2 = [], []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.array(pts1)
pts2 = np.array(pts2)

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

def ransac(pts1, pts2, iterations=3000):
    best_total_distance = float('inf')
    best_H = None
    num_points = len(pts1)

    for _ in range(iterations):
        indices = np.random.choice(num_points, 4, replace=False)
        sample1 = pts1[indices]
        sample2 = pts2[indices]
        H = compute_homography(sample1, sample2)

        total_distance = 0
        for i in range(num_points):
            x1, y1 = pts1[i]
            x2, y2 = pts2[i]
            pt1 = np.array([x1, y1, 1.0])
            projected = H @ pt1
            projected /= projected[2]
            u, v = projected[0], projected[1]
            total_distance += np.sqrt((u - x2)**2 + (v - y2)**2)

        if total_distance < best_total_distance:
            best_total_distance = total_distance
            best_H = H

    return best_H

H_ransac = ransac(pts1, pts2)
H_ransac /= H_ransac[2,2]

print("Custom RANSAC Homography matrix:")
print(H_ransac)

opencv_H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
print("\nOpenCV findHomography matrix:")
print(opencv_H)

def compute_panorama_dimensions(image1, image2, bestH):
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    corners = np.array([
        [0, 0, 1],
        [w1, 0, 1],
        [0, h1, 1],
        [w1, h1, 1]
    ])
    proj_corners = (bestH @ corners.T).T
    proj_corners /= proj_corners[:, 2:3]
    minX = min(np.min(proj_corners[:, 0]), 0)
    maxX = max(np.max(proj_corners[:, 0]), w2)
    minY = min(np.min(proj_corners[:, 1]), 0)
    maxY = max(np.max(proj_corners[:, 1]), h2)
    return int(maxX - minX), int(maxY - minY), int(minX), int(minY)

def stitch_images(image1, image2, bestH):
    h2, w2 = image2.shape[:2]
    pano_w, pano_h, minX, minY = compute_panorama_dimensions(image1, image2, bestH)
    panorama = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)

    # Paste image2 into panorama
    panorama[-minY:h2 - minY, -minX:w2 - minX] = image2

    # Initialize blending statistics
    total_red_diff = 0
    total_green_diff = 0
    total_blue_diff = 0
    total_pixels = 0

    invH = np.linalg.inv(bestH)

    for y in range(pano_h):
        for x in range(pano_w):
            pt2 = np.array([x + minX, y + minY, 1])
            pt1 = invH @ pt2
            pt1 /= pt1[2]
            x1, y1 = int(pt1[0]), int(pt1[1])

            if 0 <= x1 < image1.shape[1] and 0 <= y1 < image1.shape[0]:
                pixel1 = image1[y1, x1]
                pixel2 = panorama[y, x]

                if np.any(pixel2 > 0):
                    blended = ((pixel1.astype(np.uint16) + pixel2.astype(np.uint16)) // 2).astype(np.uint8)
                    panorama[y, x] = blended

                    # Preserve sign in difference
                    total_red_diff += int(pixel1[0]) - int(pixel2[0])
                    total_green_diff += int(pixel1[1]) - int(pixel2[1])
                    total_blue_diff += int(pixel1[2]) - int(pixel2[2])
                    total_pixels += 1
                else:
                    panorama[y, x] = pixel1

    print("\nBlending statistics:")
    print(f"Total pixels blended: {total_pixels}")
    print(f"Total red diff: {total_red_diff}")
    print(f"Total green diff: {total_green_diff}")
    print(f"Total blue diff: {total_blue_diff}")

    return panorama

result = stitch_images(img1, img2, H_ransac)
cv2.imwrite('ProjectFiles/CSC-340/Media/stitched_panorama.jpg', result)