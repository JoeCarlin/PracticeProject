import cv2
import numpy as np
import math

# Matrix multiplication function
def multiply_matrices(A, B):
    # A and B are 2D NumPy arrays (matrices)
    return np.dot(A, B)

# Rotation matrix function
def get_rotation_matrix(angle):
    theta = math.radians(angle)
    return np.array([[math.cos(theta), -math.sin(theta)], 
                     [math.sin(theta), math.cos(theta)]])

# Function to rotate image by a given angle using matrix multiplication
def rotate_image(image, angle, step_size):
    rows, cols = image.shape[:2]
    center = (cols / 2, rows / 2)
    
    # Create an empty canvas with black border around the original image
    new_rows, new_cols = rows + 2 * (rows // 10), cols + 2 * (cols // 10)
    new_image = np.zeros((new_rows, new_cols, 3), dtype=np.uint8)
    new_image[rows//10:rows//10+rows, cols//10:cols//10+cols] = image
    
    # Apply rotation
    for i in range(0, 360, step_size):
        rotation_matrix = get_rotation_matrix(i)
        rotated_image = np.copy(new_image)
        
        # Perform the rotation (matrix multiplication with each pixel coordinate)
        for x in range(new_cols):
            for y in range(new_rows):
                old_x = (x - new_cols / 2)
                old_y = (y - new_rows / 2)
                
                # Rotate the pixel
                new_coords = multiply_matrices(rotation_matrix, np.array([old_x, old_y]))
                new_x = int(new_coords[0] + new_cols / 2)
                new_y = int(new_coords[1] + new_rows / 2)
                
                # Check bounds and set pixel
                if 0 <= new_x < new_cols and 0 <= new_y < new_rows:
                    rotated_image[y, x] = new_image[new_y, new_x]
        
        # Display intermediate rotated image
        cv2.imshow(f'Rotated by {i} degrees', rotated_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    return rotated_image

# Color Error Calculation (Absolute Color Error)
def calculate_absolute_error(original_image, rotated_image):
    error = np.abs(original_image.astype(np.int32) - rotated_image.astype(np.int32))
    return np.sum(error)

# Pixel Rounding Error Calculation
def calculate_rounding_error(original_image, rotated_image, num_rotations, num_pixels):
    rounding_error = 0
    for i in range(rotated_image.shape[0]):
        for j in range(rotated_image.shape[1]):
            rounding_error += np.linalg.norm([i - j, j - i])
    
    return rounding_error / (num_pixels * num_rotations)

def main():
    image = cv2.imread('ProjectFiles/CSC-340/Media/cones1.png')
    
    if image is None:
        print("Error: Unable to read the image file. Please check the file path and integrity.")
        return
    
    # Add black border around the image
    rows, cols = image.shape[:2]
    border = 20
    bordered_image = np.zeros((rows + 2*border, cols + 2*border, 3), dtype=np.uint8)
    bordered_image[border:border+rows, border:border+cols] = image
    
    step_sizes = [45, 60, 90, 120, 180, 360]  # Step sizes
    for step in step_sizes:
        rotated_image = rotate_image(bordered_image, 360, step)
        
        # Color error
        color_error = calculate_absolute_error(image, rotated_image)
        
        # Pixel rounding error
        rounding_error = calculate_rounding_error(image, rotated_image, len(step_sizes), rows * cols)
        
        # Display the results
        print(f"Rotation Step Size: {step}°")
        print(f"Absolute Color Error: {color_error}")
        print(f"Pixel Rounding Error: {rounding_error}")
        # Save rotated image
        cv2.imwrite(f"ProjectFiles/CSC-340/Media/rotated_image_{step}.png", rotated_image)

if __name__ == "__main__":
    main()