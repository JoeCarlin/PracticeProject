import numpy as np
import cv2

# Compute the image gradients (Ix, Iy) using a simple finite difference method.
def compute_gradients(img):
    height, width = img.shape
    Ix = np.zeros_like(img, dtype=np.float32)
    Iy = np.zeros_like(img, dtype=np.float32)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            Ix[y, x] = (float(img[y, x + 1]) - float(img[y, x - 1])) / 2.0
            Iy[y, x] = (float(img[y + 1, x]) - float(img[y - 1, x])) / 2.0

    return Ix, Iy

# Compute the gradient products: Ixx, Iyy, Ixy
def compute_gradient_products(Ix, Iy):
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix * Iy
    return Ixx, Iyy, Ixy

# Harris corner response function calculation
def compute_harris_response(Ixx, Iyy, Ixy, k=0.04):
    height, width = Ixx.shape
    cornerness = np.zeros_like(Ixx, dtype=np.float32)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            Ixx_sum = np.sum(Ixx[y-1:y+2, x-1:x+2])
            Iyy_sum = np.sum(Iyy[y-1:y+2, x-1:x+2])
            Ixy_sum = np.sum(Ixy[y-1:y+2, x-1:x+2])

            det_M = Ixx_sum * Iyy_sum - Ixy_sum**2
            trace_M = Ixx_sum + Iyy_sum

            R = det_M - k * (trace_M**2)
            cornerness[y, x] = R

    return cornerness

# Function to compute the Harris cornerness from the image
def harris_corner_detection(img_path, k=0.04):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found!")

    Ix, Iy = compute_gradients(img)
    Ixx, Iyy, Ixy = compute_gradient_products(Ix, Iy)
    cornerness = compute_harris_response(Ixx, Iyy, Ixy, k)

    return cornerness, img

# Threshold-based corner detection and visualization (original method)
def save_corners_overlay(img_path, cornerness, threshold_percentage=0.01):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found for overlay.")

    threshold = threshold_percentage * np.max(cornerness)
    corners = np.where(cornerness > threshold)

    for y, x in zip(*corners):
        cv2.circle(img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

    output_path = "ProjectFiles/CSC-340/Media/corners.jpg"
    cv2.imwrite(output_path, img)
    print(f"Image with corners saved as {output_path}")

# Custom implementation for selecting the top n corners per block
def select_blockwise_top_corners(cornerness, m=4, n=10, min_response_ratio=0.01):
    height, width = cornerness.shape
    corner_mask = np.zeros_like(cornerness, dtype=np.uint8)
    block_height = height // m
    block_width = width // m

    max_response = np.max(cornerness)
    min_response_threshold = min_response_ratio * max_response

    for i in range(m):
        for j in range(m):
            y_start = i * block_height
            y_end = (i + 1) * block_height if i != m - 1 else height
            x_start = j * block_width
            x_end = (j + 1) * block_width if j != m - 1 else width

            block = cornerness[y_start:y_end, x_start:x_end]
            flat = block.flatten()

            if flat.size == 0:
                continue

            valid_indices = np.where(flat > min_response_threshold)[0]
            if valid_indices.size == 0:
                continue

            # Rank and select top n cornerness values (without using built-in sort)
            top_indices = []
            top_values = []

            for idx in valid_indices:
                y, x = np.unravel_index(idx, block.shape)
                corner_value = flat[idx]
                if len(top_values) < n:
                    top_values.append(corner_value)
                    top_indices.append((y, x))
                else:
                    min_val = min(top_values)
                    if corner_value > min_val:
                        min_idx = top_values.index(min_val)
                        top_values[min_idx] = corner_value
                        top_indices[min_idx] = (y, x)

            # Mark the top corners in the mask
            for y, x in top_indices:
                corner_mask[y_start + y, x_start + x] = 255

    return corner_mask

# Save blockwise corners overlay with custom ranking
def save_blockwise_corners_overlay(img_path, cornerness, m=4, n=10):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found for overlay.")

    corner_mask = select_blockwise_top_corners(cornerness, m, n)
    ys, xs = np.where(corner_mask == 255)

    for y, x in zip(ys, xs):
        cv2.circle(img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

    output_path = "ProjectFiles/CSC-340/Media/corners_blockwise_overlayed.jpg"
    cv2.imwrite(output_path, img)
    print(f"Blockwise corners saved as {output_path}")

# Bonus 2: grayscale cornerness visualization
def visualize_cornerness_grayscale_manual(cornerness, output_path="ProjectFiles/CSC-340/Media/cornerness_grayscale.jpg"):
    height, width = cornerness.shape
    min_val = float('inf')
    max_val = float('-inf')

    # Find min and max manually
    for y in range(height):
        for x in range(width):
            val = cornerness[y][x]
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val

    # Avoid division by zero
    if max_val - min_val == 0:
        max_val += 1e-5

    # Create a grayscale image manually
    grayscale_img = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            norm_val = (cornerness[y][x] - min_val) / (max_val - min_val)
            gray_val = int(norm_val * 255)
            grayscale_img[y][x] = gray_val

    cv2.imwrite(output_path, grayscale_img)
    print(f"Grayscale cornerness visualization saved at {output_path}")

def main():
    img_path = 'ProjectFiles/CSC-340/Media/checkerboard.png'

    # Harris corner detection
    cornerness, img = harris_corner_detection(img_path)

    # Original method: thresholded corners
    save_corners_overlay(img_path, cornerness)

    # Bonus 1: blockwise top corners (using custom ranking)
    save_blockwise_corners_overlay(img_path, cornerness, m=4, n=10)

    # Bonus 2: grayscale cornerness visualization (fixed function name here)
    visualize_cornerness_grayscale_manual(cornerness)

if __name__ == "__main__":
    main()