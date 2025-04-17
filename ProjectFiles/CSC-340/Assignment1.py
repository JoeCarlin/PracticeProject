import cv2
import numpy as np
import math

def multiply_matrices(A, B):
    result = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i, j] += A[i, k] * B[k, j]
    return result

def get_rotation_matrix(angle):
    theta = math.radians(angle)
    return np.array([[math.cos(theta), -math.sin(theta)], 
                     [math.sin(theta), math.cos(theta)]])

def rotate_image(image, angle_step):
    rows, cols = image.shape[:2]

    # Add a black border to prevent clipping during rotation
    border = max(rows, cols)
    new_rows, new_cols = rows + 2 * border, cols + 2 * border
    padded_image = np.zeros((new_rows, new_cols, 3), dtype=np.uint8)
    for y in range(rows):
        for x in range(cols):
            padded_image[y + border][x + border] = image[y][x]

    rotated_image = padded_image.copy()
    center_x = new_cols / 2
    center_y = new_rows / 2

    num_rotations = 360 // angle_step

    for step in range(num_rotations):
        rotation_matrix = get_rotation_matrix(angle_step)
        temp_image = np.zeros_like(rotated_image)

        for y in range(new_rows):
            for x in range(new_cols):
                # Shift origin to center
                dx = x - center_x
                dy = y - center_y

                # Rotate point using custom matrix multiplication
                new_x = rotation_matrix[0][0] * dx + rotation_matrix[0][1] * dy
                new_y = rotation_matrix[1][0] * dx + rotation_matrix[1][1] * dy

                # Shift back
                final_x = int(round(new_x + center_x))
                final_y = int(round(new_y + center_y))

                # Copy pixel if within bounds
                if 0 <= final_x < new_cols and 0 <= final_y < new_rows:
                    temp_image[final_y][final_x] = rotated_image[y][x]

        rotated_image = temp_image

        cv2.imshow(f'After {angle_step * (step + 1)}° rotation', rotated_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    return rotated_image

def calculate_absolute_error(original, rotated, border):
    height = original.shape[0]
    width = original.shape[1]

    # Crop rotated image to original size (remove border)
    cropped_rotated = rotated[border:border+height, border:border+width]

    total_error = 0
    for y in range(height):
        for x in range(width):
            orig_pixel = original[y][x]
            rot_pixel = cropped_rotated[y][x]

            # Manually compute absolute color error for R, G, B
            for c in range(3):  # RGB channels
                diff = int(orig_pixel[c]) - int(rot_pixel[c])
                total_error += abs(diff)

    total_pixels = height * width
    avg_error = total_error / total_pixels  # NOT dividing by 3
    return avg_error

def calculate_rounding_error(original, rotated, angle_step, num_pixels):
    error = 0
    center_x, center_y = original.shape[1] // 2, original.shape[0] // 2
    current_rotation_matrix = np.eye(2)  # Start with the identity matrix

    for _ in range(360 // angle_step):
        rotation_matrix = get_rotation_matrix(angle_step)
        current_rotation_matrix = multiply_matrices(rotation_matrix, current_rotation_matrix)
        
        for x in range(original.shape[1]):
            for y in range(original.shape[0]):
                # Center the origin
                original_coords = np.array([[x - center_x], [y - center_y]])
                rotated_coords = multiply_matrices(current_rotation_matrix, original_coords)
                
                # Translate back to original space
                rotated_coords[0] += center_x
                rotated_coords[1] += center_y
                rounded_coords_x = round(rotated_coords[0, 0])
                rounded_coords_y = round(rotated_coords[1, 0])
                
                # Calculate rounding error for valid pixel coordinates
                if 0 <= rounded_coords_x < original.shape[1] and \
                   0 <= rounded_coords_y < original.shape[0]:
                    dx = rotated_coords[0, 0] - rounded_coords_x
                    dy = rotated_coords[1, 0] - rounded_coords_y
                    error += math.sqrt(dx * dx + dy * dy)

    return error / num_pixels

def main():
    image = cv2.imread('ProjectFiles/CSC-340/Media/cones1.png')
    
    if image is None:
        print("Error: Unable to read the image file. Please check the file path and integrity.")
        return

    # Rotate the image
    angle_step = 45
    rotated_image = rotate_image(image, angle_step)
    
    # Calculate the errors
    border = max(image.shape[:2])
    absolute_error = calculate_absolute_error(image, rotated_image, border)
    rounding_error = calculate_rounding_error(image, rotated_image, angle_step, image.shape[0] * image.shape[1])
    
    # Print the results
    print(f"Angle Step Size: {angle_step} degrees")
    print(f"# Rotations: {360 // angle_step}")
    print(f"Absolute Color Error: {absolute_error:.3f}")
    print(f"Pixel Rounding Error: {rounding_error:.3f}")
    print(f"(# Rotations) * (Pixel Displacement): {(360 // angle_step) * rounding_error:.3f}")
    
    cv2.imwrite('ProjectFiles/CSC-340/Media/cones_rotate.png', rotated_image)

if __name__ == "__main__":
    main()