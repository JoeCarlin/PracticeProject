import numpy as np
import cv2
import math

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Create a larger image with a black border
    diagonal = int(math.sqrt(h**2 + w**2))
    bordered_image = np.zeros((diagonal, diagonal, 3), dtype=image.dtype)
    offset_y = (diagonal - h) // 2
    offset_x = (diagonal - w) // 2
    bordered_image[offset_y:offset_y+h, offset_x:offset_x+w] = image

    # Compute the rotation matrix components
    theta = np.radians(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Create an empty image for the rotated result
    rotated_image = np.zeros_like(bordered_image)

    # Perform the rotation
    for i in range(diagonal):
        for j in range(diagonal):
            # Step 1: Translate (center the origin)
            Ty, Tx = i - diagonal // 2, j - diagonal // 2

            # Step 2: Rotate (using 2D rotation matrix formula)
            Rx = Tx * cos_theta - Ty * sin_theta
            Ry = Tx * sin_theta + Ty * cos_theta

            # Step 3: Translate back
            Fy, Fx = Ry + diagonal // 2, Rx + diagonal // 2

            # Bilinear interpolation
            if 0 <= Fy < diagonal - 1 and 0 <= Fx < diagonal - 1:
                x1, y1 = int(Fx), int(Fy)
                x2, y2 = x1 + 1, y1 + 1

                a = Fx - x1
                b = Fy - y1

                if x2 < diagonal and y2 < diagonal:
                    rotated_image[i, j] = (
                        (1 - a) * (1 - b) * bordered_image[y1, x1] +
                        a * (1 - b) * bordered_image[y1, x2] +
                        (1 - a) * b * bordered_image[y2, x1] +
                        a * b * bordered_image[y2, x2]
                    )

    return rotated_image

# Load the image
image1 = cv2.imread("ProjectFiles/CSC-340/Media/cones1.png")

# Check if image was loaded correctly
if image1 is None:
    print("Error: Image not found or could not be loaded.")
else:
    # Define the number of rotations to perform
    rotate_number = 1
    # Rotate image
    rotated_image = rotate_image(image1, 45 * rotate_number)

    # Save the rotated image to the specified path
    save_path = "ProjectFiles/CSC-340/Media/rotated_image.jpg"
    cv2.imwrite(save_path, rotated_image)
    print(f"Rotated image saved to {save_path}")

    # Display the image
    cv2.imshow('Rotated Image', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()