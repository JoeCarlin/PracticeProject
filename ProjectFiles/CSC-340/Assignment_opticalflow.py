import cv2
import numpy as np
from math import atan2, pi, sqrt

def compute_gradients(img1, img2):
    """
    Compute the gradients of the two images (Ix, Iy, and It).

    Args:
        img1: First image (grayscale).
        img2: Second image (grayscale).

    Returns:
        Ix: Gradient in the x direction.
        Iy: Gradient in the y direction.
        It: Temporal gradient (difference between img1 and img2).
    """
    height, width = img1.shape
    Ix = [[0.0 for _ in range(width)] for _ in range(height)]
    Iy = [[0.0 for _ in range(width)] for _ in range(height)]
    It = [[0.0 for _ in range(width)] for _ in range(height)]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Compute gradients using central difference for spatial (Ix, Iy) and temporal (It) gradients
            Ix[y][x] = (float(img1[y][x + 1]) - float(img1[y][x - 1])) / 2.0
            Iy[y][x] = (float(img1[y + 1][x]) - float(img1[y - 1][x])) / 2.0
            It[y][x] = float(img2[y][x]) - float(img1[y][x])

    return Ix, Iy, It

def lucas_kanade_flow(img1, img2, window_size=21):
    """
    Compute the optical flow using the Lucas-Kanade method.

    Args:
        img1: First image (grayscale).
        img2: Second image (grayscale).
        window_size: Size of the window for calculating the flow.

    Returns:
        u: Horizontal optical flow.
        v: Vertical optical flow.
        Ix: Gradient in the x direction.
        Iy: Gradient in the y direction.
        It: Temporal gradient (difference between img1 and img2).
    """
    # Compute image gradients (Ix, Iy, It)
    Ix, Iy, It = compute_gradients(img1, img2)
    height, width = img1.shape
    u = [[0.0 for _ in range(width)] for _ in range(height)]
    v = [[0.0 for _ in range(width)] for _ in range(height)]

    half_win = window_size // 2

    # Loop over every pixel in the image, skipping the edges due to window size
    for y in range(half_win, height - half_win):
        for x in range(half_win, width - half_win):
            A = []
            b = []

            # Construct A matrix and b vector for each pixel in the window
            for j in range(-half_win, half_win + 1):
                for i in range(-half_win, half_win + 1):
                    ix = Ix[y + j][x + i]
                    iy = Iy[y + j][x + i]
                    it = It[y + j][x + i]
                    A.append([ix, iy])
                    b.append(-it)

            # Solve the system of equations: A * [u, v] = b
            a11 = sum([a[0] * a[0] for a in A])
            a12 = sum([a[0] * a[1] for a in A])
            a22 = sum([a[1] * a[1] for a in A])
            b1 = sum([A[i][0] * b[i] for i in range(len(A))])
            b2 = sum([A[i][1] * b[i] for i in range(len(A))])

            det = a11 * a22 - a12 * a12
            if det != 0:
                # Inverse of A matrix
                inv_a11 = a22 / det
                inv_a12 = -a12 / det
                inv_a21 = -a12 / det
                inv_a22 = a11 / det

                # Calculate u and v (optical flow components)
                u_val = inv_a11 * b1 + inv_a12 * b2
                v_val = inv_a21 * b1 + inv_a22 * b2

                u[y][x] = u_val
                v[y][x] = v_val

    return u, v, Ix, Iy, It

def normalize_and_convert_to_uint8(data):
    """
    Normalize the flow data to the range [0, 255] and convert to uint8.

    Args:
        data: 2D list of flow data (u, v, or gradients).

    Returns:
        Normalized 2D numpy array with uint8 values.
    """
    flat = [val for row in data for val in row]
    min_val = min(flat)
    max_val = max(flat)
    if max_val == min_val:
        max_val = min_val + 1  # avoid division by zero

    height = len(data)
    width = len(data[0])
    result = np.zeros((height, width), dtype=np.uint8)

    # Normalize and convert to uint8
    for y in range(height):
        for x in range(width):
            norm = (data[y][x] - min_val) / (max_val - min_val)
            result[y][x] = int(norm * 255)

    return result

def multiply_and_normalize(a, b):
    """
    Multiply two 2D arrays element-wise and normalize the result.

    Args:
        a: First 2D array.
        b: Second 2D array.

    Returns:
        Normalized product of the two arrays.
    """
    height = len(a)
    width = len(a[0])
    product = [[a[y][x] * b[y][x] for x in range(width)] for y in range(height)]
    return normalize_and_convert_to_uint8(product)

def save_intermediate_images(Ix, Iy, It, u, v, out_prefix='flow'):
    """
    Save intermediate images for visualizing the gradients, flow, and magnitude.

    Args:
        Ix, Iy, It: Image gradients in x, y, and time directions.
        u, v: Horizontal and vertical optical flow components.
        out_prefix: Prefix for output image filenames.
    """
    cv2.imwrite(f'{out_prefix}_Ix.png', normalize_and_convert_to_uint8(Ix))
    cv2.imwrite(f'{out_prefix}_Iy.png', normalize_and_convert_to_uint8(Iy))
    cv2.imwrite(f'{out_prefix}_It.png', normalize_and_convert_to_uint8(It))
    cv2.imwrite(f'{out_prefix}_ItIx.png', multiply_and_normalize(It, Ix))
    cv2.imwrite(f'{out_prefix}_ItIy.png', multiply_and_normalize(It, Iy))
    cv2.imwrite(f'{out_prefix}_u.png', normalize_and_convert_to_uint8(u))
    cv2.imwrite(f'{out_prefix}_v.png', normalize_and_convert_to_uint8(v))

    # Calculate magnitude of flow and save it
    mag = [[sqrt(u[y][x] ** 2 + v[y][x] ** 2) for x in range(len(u[0]))] for y in range(len(u))]
    cv2.imwrite(f'{out_prefix}_magnitude.png', normalize_and_convert_to_uint8(mag))

def color_flow_image(u, v):
    """
    Create a color-coded image representing the optical flow.

    Args:
        u: Horizontal optical flow.
        v: Vertical optical flow.

    Returns:
        Color-coded flow image.
    """
    height = len(u)
    width = len(u[0])
    color_img = [[[0, 0, 0] for _ in range(width)] for _ in range(height)]

    # Step 1: Calculate magnitude and find max
    mag = [[sqrt(u[y][x]**2 + v[y][x]**2) for x in range(width)] for y in range(height)]
    max_mag = max([max(row) for row in mag])
    if max_mag == 0:
        max_mag = 1.0  # avoid divide by zero

    # Step 2: Map angle to color and scale by normalized magnitude
    for y in range(height):
        for x in range(width):
            fx = u[y][x]
            fy = v[y][x]
            angle = atan2(fy, fx)  # [-pi, pi]
            magnitude = mag[y][x]

            # Normalize angle to [0, 1]
            norm_angle = (angle + pi) / (2 * pi)
            blue = int(255 * norm_angle)
            green = int(255 * (1 - norm_angle))
            red = 0

            # Normalize magnitude
            intensity = magnitude / max_mag
            color_img[y][x] = [
                int(blue * intensity),
                int(green * intensity),
                int(red)
            ]

    # Convert to image format
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            img[y, x] = color_img[y][x]

    return img

def draw_arrow_overlay(gray_img, u, v, step=10):
    """
    Draw an arrow overlay on the grayscale image to represent optical flow.

    Args:
        gray_img: Grayscale image.
        u: Horizontal optical flow.
        v: Vertical optical flow.
        step: Step size for arrow placement.

    Returns:
        Image with arrows drawn over the grayscale image.
    """
    height = len(u)
    width = len(u[0])
    overlay = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    # Draw arrows on the image
    for y in range(0, height, step):
        for x in range(0, width, step):
            dx = u[y][x]
            dy = v[y][x]
            if dx**2 + dy**2 > 1:
                pt1 = (x, y)
                pt2 = (int(x + dx), int(y + dy))
                cv2.arrowedLine(overlay, pt1, pt2, (0, 0, 255), 1, tipLength=0.3)

    return overlay

def show_img(title, img_path):
    """
    Display an image using OpenCV.

    Args:
        title: Title of the window.
        img_path: Path to the image file.
    """
    img = cv2.imread(img_path)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """
    Main function to run the optical flow computation, save intermediate results,
    and display the color-coded flow and arrow overlays.
    """
    # Paths to input images and output
    img1_path = 'ProjectFiles/CSC-340/Media/sphere1.jpg'
    img2_path = 'ProjectFiles/CSC-340/Media/sphere2.jpg'
    out_prefix = 'ProjectFiles/CSC-340/Media/sphere'

    # Load and convert to grayscale
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute optical flow
    u, v, Ix, Iy, It = lucas_kanade_flow(gray1, gray2)

    # Save and show intermediate results
    save_intermediate_images(Ix, Iy, It, u, v, out_prefix=out_prefix)
    show_img("Ix", f'{out_prefix}_Ix.png')
    show_img("Iy", f'{out_prefix}_Iy.png')
    show_img("It", f'{out_prefix}_It.png')
    show_img("It*Ix", f'{out_prefix}_ItIx.png')
    show_img("It*Iy", f'{out_prefix}_ItIy.png')
    show_img("u", f'{out_prefix}_u.png')
    show_img("v", f'{out_prefix}_v.png')
    show_img("Magnitude", f'{out_prefix}_magnitude.png')

    # Color-coded flow and arrows
    color_img = color_flow_image(u, v)
    arrow_img = draw_arrow_overlay(gray1, u, v)

    cv2.imwrite(f'{out_prefix}_color.png', color_img)
    cv2.imwrite(f'{out_prefix}_arrows.png', arrow_img)

    cv2.imshow("Color-coded Flow", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("Arrow Overlay", arrow_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("All images saved and displayed.")

if __name__ == "__main__":
    main()