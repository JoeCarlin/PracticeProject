import numpy as np
import cv2

def compute_gradients(img):
    """
    Compute the gradients of the image (Ix, Iy).

    Args:
        img: Input grayscale image.

    Returns:
        Ix: Gradient in the x direction.
        Iy: Gradient in the y direction.
    """
    height, width = img.shape
    Ix = np.zeros_like(img, dtype=np.float32)
    Iy = np.zeros_like(img, dtype=np.float32)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Compute gradients using central difference for spatial gradients
            Ix[y, x] = (float(img[y, x + 1]) - float(img[y, x - 1])) / 2.0
            Iy[y, x] = (float(img[y + 1, x]) - float(img[y - 1, x])) / 2.0

    return Ix, Iy

def compute_gradient_products(Ix, Iy):
    """Compute the products of gradients: Ixx, Iyy, Ixy"""
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix * Iy
    return Ixx, Iyy, Ixy

def compute_harris_response(Ixx, Iyy, Ixy, k=0.04):
    """Compute the Harris corner response for each pixel"""
    height, width = Ixx.shape
    cornerness = np.zeros_like(Ixx, dtype=np.float32)
    
    for y in range(1, height-1):
        for x in range(1, width-1):
            # Sum up values in the 3x3 neighborhood
            Ixx_sum = np.sum(Ixx[y-1:y+2, x-1:x+2])
            Iyy_sum = np.sum(Iyy[y-1:y+2, x-1:x+2])
            Ixy_sum = np.sum(Ixy[y-1:y+2, x-1:x+2])
            
            # Compute the determinant and trace of the matrix M
            det_M = Ixx_sum * Iyy_sum - Ixy_sum**2
            trace_M = Ixx_sum + Iyy_sum
            
            # Compute the corner response value
            R = det_M - k * (trace_M**2)
            cornerness[y, x] = R
    
    return cornerness

def harris_corner_detection(img_path, k=0.04):
    """Main function to perform Harris Corner Detection"""
    # Load the image and convert to grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Image not found!")
    
    # Step 1: Compute the gradients
    Ix, Iy = compute_gradients(img)
    
    # Step 2: Compute the gradient products
    Ixx, Iyy, Ixy = compute_gradient_products(Ix, Iy)
    
    # Step 3: Compute the Harris corner response
    cornerness = compute_harris_response(Ixx, Iyy, Ixy, k)
    
    return cornerness, img

def save_corners_overlay(img_path, cornerness, threshold_percentage=0.01):
    """Save the image with corners overlaid on it"""
    # Load the original image in color to overlay corners
    img = cv2.imread(img_path)
    
    # Threshold cornerness values to detect corners
    threshold = threshold_percentage * np.max(cornerness)
    corners = (cornerness > threshold).astype(np.uint8) * 255  # Create a binary mask for corners
    
    # Overlay red on detected corners
    img[corners == 255] = [0, 0, 255]  # Set detected corner pixels to red

    # Save the result to a file
    output_path = "ProjectFiles/CSC-340/Media/corners_overlayed.jpg"
    cv2.imwrite(output_path, img)
    print(f"Image with corners saved as {output_path}")

def main():
    # Define the path for the image
    img_path = 'ProjectFiles/CSC-340/Media/checkerboard.png'  # Replace with your image path
    
    # Perform Harris corner detection
    cornerness, img = harris_corner_detection(img_path)
    
    # Save the image with corners overlaid
    save_corners_overlay(img_path, cornerness)

# Call the main function if this file is executed as a script
if __name__ == "__main__":
    main()