import numpy as np
import matplotlib.pyplot as plt

def create_gaussian_filter(size, sigma):
    """Creates a manually computed 2D Gaussian filter."""
    print(f"Creating Gaussian filter with size={size} and sigma={sigma}...")
    k = size // 2
    x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian /= np.sum(gaussian)  # Normalize

    # Print the filter rounded to 4 decimal places
    print("Gaussian filter (rounded to 4 decimals):")
    for row in gaussian:
        print(["{:.4f}".format(val) for val in row])

    return gaussian

def convolve2d(image, kernel):
    """Manually performs a 2D convolution."""
    print("Starting convolution...")
    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    # Pad the original image with zeros on all sides
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    output = np.zeros_like(image)

    for i in range(img_h):
        for j in range(img_w):
            region = padded_image[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(region * kernel)

    print("Convolution complete.")
    return output

def main():
    # Parameters (you can change these to test different filters)
    filter_size = 5
    sigma = 1.5

    # Test image (5x5) - easy to see effects
    image = np.array([
        [100, 100, 100,   0,   0],
        [100, 100, 100,   0,   0],
        [100, 100, 100,   0,   0],
        [  0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0]
    ], dtype=np.float32)

    # Step 1: Create the Gaussian filter
    gaussian_filter = create_gaussian_filter(filter_size, sigma)

    # Step 2: Apply convolution
    blurred_image = convolve2d(image, gaussian_filter)

    # Step 3: Display results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'Blurred Image\n(Filter Size: {filter_size}x{filter_size}, Sigma: {sigma})')
    plt.imshow(blurred_image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()