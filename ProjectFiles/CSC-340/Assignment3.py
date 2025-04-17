import numpy as np
import cv2

# Compute gradients in x and y directions using central difference
def compute_gradients(img):
    height, width = img.shape
    Ix = np.zeros_like(img, dtype=np.float32)  # Gradient in x-direction
    Iy = np.zeros_like(img, dtype=np.float32)  # Gradient in y-direction

    # Loop over all pixels except the border pixels
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Compute central difference gradient along x
            Ix[y, x] = (float(img[y, x + 1]) - float(img[y, x - 1])) / 2.0
            # Compute central difference gradient along y
            Iy[y, x] = (float(img[y + 1, x]) - float(img[y - 1, x])) / 2.0

    return Ix, Iy

# Compute products of gradients needed for Harris matrix
def compute_gradient_products(Ix, Iy):
    height, width = Ix.shape
    Ixx = np.zeros((height, width), dtype=np.float32)  # Ix^2
    Iyy = np.zeros((height, width), dtype=np.float32)  # Iy^2
    Ixy = np.zeros((height, width), dtype=np.float32)  # Ix * Iy

    for y in range(height):
        for x in range(width):
            Ixx[y, x] = Ix[y, x] * Ix[y, x]
            Iyy[y, x] = Iy[y, x] * Iy[y, x]
            Ixy[y, x] = Ix[y, x] * Iy[y, x]

    return Ixx, Iyy, Ixy

# Compute Harris response using cornerness formula
def compute_harris_response(Ixx, Iyy, Ixy, k=0.04):
    height, width = Ixx.shape
    cornerness = np.zeros_like(Ixx, dtype=np.float32)

    # Loop over interior pixels to compute Harris response
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            Ixx_sum = 0
            Iyy_sum = 0
            Ixy_sum = 0

            # Sum gradient products in a 3x3 neighborhood
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    Ixx_sum += Ixx[y + dy, x + dx]
                    Iyy_sum += Iyy[y + dy, x + dx]
                    Ixy_sum += Ixy[y + dy, x + dx]

            # Compute determinant and trace of matrix M
            det_M = Ixx_sum * Iyy_sum - Ixy_sum * Ixy_sum
            trace_M = Ixx_sum + Iyy_sum
            # Harris cornerness measure
            R = det_M - k * (trace_M * trace_M)
            cornerness[y, x] = R

    return cornerness

# Run full Harris detection pipeline
def harris_corner_detection(img_path, k=0.04):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load grayscale image
    if img is None:
        raise ValueError("Image not found!")

    Ix, Iy = compute_gradients(img)
    Ixx, Iyy, Ixy = compute_gradient_products(Ix, Iy)
    cornerness = compute_harris_response(Ixx, Iyy, Ixy, k)

    return cornerness, img

# Overlay corners on original image using global threshold
def save_corners_overlay(img_path, cornerness, threshold_percentage=0.01):
    img = cv2.imread(img_path)  # Load original (color) image
    if img is None:
        raise ValueError("Image not found for overlay.")

    max_corner = -1e9  # Track maximum cornerness response
    height, width = cornerness.shape

    # Find maximum cornerness value
    for y in range(height):
        for x in range(width):
            if cornerness[y, x] > max_corner:
                max_corner = cornerness[y, x]

    threshold = threshold_percentage * max_corner  # Threshold for filtering

    # Draw circles for corners above threshold
    for y in range(height):
        for x in range(width):
            if cornerness[y, x] > threshold:
                cv2.circle(img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

    output_path = "ProjectFiles/CSC-340/Media/corners.jpg"
    cv2.imwrite(output_path, img)
    print(f"Image with corners saved as {output_path}")

# Select top N corners in each block of the image
def select_blockwise_top_corners(cornerness, m=4, n=10, min_response_ratio=0.01):
    height, width = cornerness.shape
    corner_mask = np.zeros((height, width), dtype=np.uint8)
    block_height = height // m
    block_width = width // m

    # Find global max response
    max_response = -1e9
    for y in range(height):
        for x in range(width):
            if cornerness[y, x] > max_response:
                max_response = cornerness[y, x]

    min_response_threshold = min_response_ratio * max_response

    # Divide image into m x m blocks
    for i in range(m):
        for j in range(m):
            y_start = i * block_height
            y_end = (i + 1) * block_height if i != m - 1 else height
            x_start = j * block_width
            x_end = (j + 1) * block_width if j != m - 1 else width

            # Initialize top N lists for each block
            top_vals = [-1e9] * n
            top_coords = [(-1, -1)] * n

            # Find top N corners in the block manually
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    val = cornerness[y, x]
                    if val > min_response_threshold:
                        min_idx = 0
                        min_val = top_vals[0]
                        # Find smallest value in top_vals
                        for k in range(1, n):
                            if top_vals[k] < min_val:
                                min_idx = k
                                min_val = top_vals[k]
                        # Replace smallest if current value is higher
                        if val > min_val:
                            top_vals[min_idx] = val
                            top_coords[min_idx] = (y, x)

            # Set mask at top corner positions
            for yx in top_coords:
                y, x = yx
                if y >= 0 and x >= 0:
                    corner_mask[y, x] = 255

    return corner_mask

# Overlay blockwise top corners on original image
def save_blockwise_corners_overlay(img_path, cornerness, m=4, n=10):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found for overlay.")

    corner_mask = select_blockwise_top_corners(cornerness, m, n)
    height, width = corner_mask.shape

    # Draw red dots on corner locations
    for y in range(height):
        for x in range(width):
            if corner_mask[y, x] == 255:
                cv2.circle(img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

    output_path = "ProjectFiles/CSC-340/Media/corners_blockwise_overlayed.jpg"
    cv2.imwrite(output_path, img)
    print(f"Blockwise corners saved as {output_path}")

# Convert cornerness matrix to grayscale image for visualization
def visualize_cornerness_grayscale_manual(cornerness, output_path="ProjectFiles/CSC-340/Media/cornerness_grayscale.jpg"):
    height, width = cornerness.shape
    min_val = 1e9
    max_val = -1e9

    # Find min and max values in cornerness for normalization
    for y in range(height):
        for x in range(width):
            val = cornerness[y][x]
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val

    # Prevent division by zero
    if max_val - min_val == 0:
        max_val += 1e-5

    grayscale_img = np.zeros((height, width), dtype=np.uint8)

    # Normalize and scale each value to 0-255 range
    for y in range(height):
        for x in range(width):
            norm_val = (cornerness[y][x] - min_val) / (max_val - min_val)
            gray_val = int(norm_val * 255)
            grayscale_img[y][x] = gray_val

    cv2.imwrite(output_path, grayscale_img)
    print(f"Grayscale cornerness visualization saved at {output_path}")

# Main function to run full pipeline
def main():
    img_path = 'ProjectFiles/CSC-340/Media/cones1.png'

    cornerness, img = harris_corner_detection(img_path)
    save_corners_overlay(img_path, cornerness)
    save_blockwise_corners_overlay(img_path, cornerness, m=4, n=10)
    visualize_cornerness_grayscale_manual(cornerness)

# Entry point
if __name__ == "__main__":
    main()