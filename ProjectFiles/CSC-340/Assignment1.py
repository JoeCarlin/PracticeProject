import cv2
import numpy as np
import math

def multiply_matrices(A, B):
    return np.dot(A, B)

def get_rotation_matrix(angle):
    theta = math.radians(angle)
    return np.array([[math.cos(theta), -math.sin(theta)], 
                     [math.sin(theta), math.cos(theta)]])

def rotate_image(image, angle_step):
    rows, cols = image.shape[:2]
    center = (cols / 2, rows / 2)
    
    # Create an empty canvas with black border around the original image
    border = max(rows, cols)
    new_rows, new_cols = rows + 2 * border, cols + 2 * border
    new_image = np.zeros((new_rows, new_cols, 3), dtype=np.uint8)
    new_image[border:border+rows, border:border+cols] = image
    
    rotated_image = new_image.copy()
    for angle in range(0, 360, angle_step):
        rotation_matrix = get_rotation_matrix(angle)
        temp_image = np.zeros_like(new_image)
        
        for x in range(new_cols):
            for y in range(new_rows):
                old_x = x - new_cols / 2
                old_y = y - new_rows / 2
                
                new_coords = multiply_matrices(rotation_matrix, np.array([old_x, old_y]))
                new_x = int(new_coords[0] + new_cols / 2)
                new_y = int(new_coords[1] + new_rows / 2)
                
                if 0 <= new_x < new_cols and 0 <= new_y < new_rows:
                    temp_image[y, x] = new_image[new_y, new_x]
        
        rotated_image = temp_image.copy()
        cv2.imshow(f'Rotated by {angle} degrees', rotated_image)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    return rotated_image

def calculate_absolute_error(original, rotated, border):
    cropped_rotated = rotated[border:border+original.shape[0], border:border+original.shape[1]]
    return np.sum(np.abs(original.astype(int) - cropped_rotated.astype(int))) / (original.shape[0] * original.shape[1] * original.shape[2])

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
                original_coords = np.array([x - center_x, y - center_y])
                rotated_coords = multiply_matrices(current_rotation_matrix, original_coords)
                
                # Translate back to original space
                rotated_coords[0] += center_x
                rotated_coords[1] += center_y
                rounded_coords = np.round(rotated_coords).astype(int)
                
                # Calculate rounding error for valid pixel coordinates
                if 0 <= rounded_coords[0] < original.shape[1] and \
                   0 <= rounded_coords[1] < original.shape[0]:
                    error += np.linalg.norm(rotated_coords - rounded_coords)

    return error / num_pixels

def main():
    image = cv2.imread('ProjectFiles/CSC-340/Media/cones1.png')
    
    if image is None:
        print("Error: Unable to read the image file. Please check the file path and integrity.")
        return
    
    angle_step = 120
    rotated_image = rotate_image(image, angle_step)
    
    border = max(image.shape[:2])
    absolute_error = calculate_absolute_error(image, rotated_image, border)
    rounding_error = calculate_rounding_error(image, rotated_image, angle_step, image.shape[0] * image.shape[1])
    
    print(f"Angle Step Size: {angle_step} degrees")
    print(f"# Rotations: {360 // angle_step}")
    print(f"Absolute Color Error: {absolute_error:.3f}")
    print(f"Pixel Rounding Error: {rounding_error:.3f}")
    print(f"(# Rotations) * (Pixel Displacement): {(360 // angle_step) * rounding_error:.3f}")
    
    cv2.imwrite('ProjectFiles/CSC-340/Media/rotated_image.png', rotated_image)

if __name__ == "__main__":
    main()