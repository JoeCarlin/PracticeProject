import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def convolve2d(image, kernel):
    """Performs a 2D convolution manually."""
    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image)

    for i in range(img_h):
        for j in range(img_w):
            region = padded_image[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(region * kernel)

    return output

def sobel_filters():
    """Define Sobel-X and Sobel-Y filters."""
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    return sobel_x, sobel_y

def main():
    # Load an image (convert to grayscale)
    image_path = 'ProjectFiles/CSC-340/Media/cones1.png'  # Replace with your image path
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = np.array(image, dtype=np.float32)

    # Get Sobel filters
    sobel_x, sobel_y = sobel_filters()

    # Apply Sobel filters
    sobel_x_result = convolve2d(image, sobel_x)
    sobel_y_result = convolve2d(image, sobel_y)

    # Display the original and filtered images
    plt.figure(figsize=(18, 6))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    # Sobel-X Filtered Image
    plt.subplot(1, 3, 2)
    plt.title('Sobel-X Filtered Image')
    plt.imshow(sobel_x_result, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    # Sobel-Y Filtered Image
    plt.subplot(1, 3, 3)
    plt.title('Sobel-Y Filtered Image')
    plt.imshow(sobel_y_result, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()