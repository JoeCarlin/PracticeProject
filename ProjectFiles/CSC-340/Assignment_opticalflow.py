from PIL import Image
import numpy as np

def load_image(path):
    return Image.open(path).convert('L')

def image_to_array(image):
    return np.array(image, dtype=np.float32)

def compute_gradients(img1, img2):
    Ix = np.zeros_like(img1)
    Iy = np.zeros_like(img1)
    It = img2 - img1

    for i in range(1, img1.shape[0] - 1):
        for j in range(1, img1.shape[1] - 1):
            Ix[i, j] = (img1[i, j+1] - img1[i, j-1]) / 2.0
            Iy[i, j] = (img1[i+1, j] - img1[i-1, j]) / 2.0

    return Ix, Iy, It

def lucas_kanade(img1, img2, window_size=5):
    Ix, Iy, It = compute_gradients(img1, img2)
    u = np.zeros_like(img1)
    v = np.zeros_like(img1)
    half_window = window_size // 2

    for i in range(half_window, img1.shape[0] - half_window):
        for j in range(half_window, img1.shape[1] - half_window):
            Ix_window = Ix[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()
            Iy_window = Iy[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()
            It_window = It[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()

            A = np.vstack((Ix_window, Iy_window)).T
            b = -It_window

            nu = np.linalg.pinv(A.T @ A) @ A.T @ b
            u[i, j] = nu[0]
            v[i, j] = nu[1]

    return u, v

def main():
    img1 = load_image('ProjectFiles/CSC-340/Media/sphere1.jpg')
    img2 = load_image('ProjectFiles/CSC-340/Media/sphere2.jpg')

    img1_array = image_to_array(img1)
    img2_array = image_to_array(img2)

    u, v = lucas_kanade(img1_array, img2_array)

    print("Optical flow (u):", u)
    print("Optical flow (v):", v)

if __name__ == "__main__":
    main()