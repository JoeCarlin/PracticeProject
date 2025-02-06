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

    total_color_error = 0
    total_pixel_rounding_error = 0
    total_pixel_displacement = 0
    pixel_count = 0

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
                    interpolated_pixel = (
                        (1 - a) * (1 - b) * bordered_image[y1, x1] +
                        a * (1 - b) * bordered_image[y1, x2] +
                        (1 - a) * b * bordered_image[y2, x1] +
                        a * b * bordered_image[y2, x2]
                    )
                    rotated_image[i, j] = interpolated_pixel

                    # Calculate color error
                    original_pixel = bordered_image[y1, x1]
                    color_error = np.abs(interpolated_pixel - original_pixel).sum()
                    total_color_error += color_error

                    # Calculate pixel rounding error
                    rounding_error = np.abs(Fx - x1) + np.abs(Fy - y1)
                    total_pixel_rounding_error += rounding_error

                    # Calculate pixel displacement
                    pixel_displacement = np.sqrt((Fx - j)**2 + (Fy - i)**2)
                    total_pixel_displacement += pixel_displacement

                    pixel_count += 1

    # Calculate average errors
    avg_color_error = total_color_error / pixel_count
    avg_pixel_rounding_error = total_pixel_rounding_error / pixel_count

    print(f"Average Color Error: {avg_color_error}")
    print(f"Average Pixel Rounding Error: {avg_pixel_rounding_error}")

    return rotated_image, total_pixel_displacement

# Load the image
image1 = cv2.imread("ProjectFiles/CSC-340/Media/cones1.png")

# Check if image was loaded correctly
if image1 is None:
    print("Error: Image not found or could not be loaded.")
else:
    # Rotation parameters
    step_size = 45         # How much to rotate each time (degrees)
    rotations = 8          # Number of rotations to apply

    total_displacement = 0

    for i in range(rotations):
        angle = step_size * (i + 1)  # Incrementing angle with each rotation
        rotated_image, pixel_displacement = rotate_image(image1, angle)
        total_displacement += pixel_displacement

        # Save the rotated image
        save_path = f"ProjectFiles/CSC-340/Media/rotated_image_{angle}.jpg"
        cv2.imwrite(save_path, rotated_image)
        print(f"Rotated image saved to {save_path} - {angle}° rotation")

        # Display the rotated image
        cv2.imshow(f'Rotated Image ({angle}°)', rotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Print rotation details
        print(f"Image rotated {angle}° with steps of {step_size}°, {i+1} rotation operation(s) applied so far.")

    # Calculate and print the total displacement
    total_rotation_displacement = total_displacement
    print(f"Total Rotation Displacement: {total_rotation_displacement}")