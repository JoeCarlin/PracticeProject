import cv2
import numpy as np
import math

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def compute_gradients(img1, img2):
    Ix = np.zeros_like(img1)
    Iy = np.zeros_like(img1)
    It = img2 - img1

    for i in range(1, img1.shape[0] - 1):
        for j in range(1, img1.shape[1] - 1):
            Ix[i, j] = (img1[i, j+1] - img1[i, j-1]) / 2.0
            Iy[i, j] = (img1[i+1, j] - img1[i-1, j]) / 2.0

    return Ix, Iy, It

def lucas_kanade(img1, img2, window_size=21):
    Ix, Iy, It = compute_gradients(img1, img2)
    height, width = img1.shape
    u = np.zeros_like(img1)
    v = np.zeros_like(img1)
    half_window = window_size // 2

    for y in range(half_window, height - half_window):
        for x in range(half_window, width - half_window):
            Ix_window = Ix[y-half_window:y+half_window+1, x-half_window:x+half_window+1].flatten()
            Iy_window = Iy[y-half_window:y+half_window+1, x-half_window:x+half_window+1].flatten()
            It_window = It[y-half_window:y+half_window+1, x-half_window:x+half_window+1].flatten()

            A = np.vstack((Ix_window, Iy_window)).T
            b = -It_window

            AtA = A.T @ A
            Atb = A.T @ b

            if np.linalg.det(AtA) != 0:
                nu = np.linalg.inv(AtA) @ Atb
                u[y, x] = nu[0]
                v[y, x] = nu[1]

    return u, v

def visualize_flow(u, v):
    height, width = u.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    max_magnitude = np.max(np.sqrt(u**2 + v**2))

    for y in range(height):
        for x in range(width):
            angle = math.atan2(v[y, x], u[y, x])
            magnitude = math.sqrt(u[y, x]**2 + v[y, x]**2)
            blue = int(((angle + math.pi) / (2 * math.pi)) * 255)
            green = int((1 - ((angle + math.pi) / (2 * math.pi))) * 255)
            intensity = int((magnitude / max_magnitude) * 255)
            rgb[y, x] = [0, green * intensity // 255, blue * intensity // 255]

    return rgb

def main():
    img1 = load_image('ProjectFiles/CSC-340/Media/sphere1.jpg')
    img2 = load_image('ProjectFiles/CSC-340/Media/sphere2.jpg')

    u, v = lucas_kanade(img1, img2)

    flow_image = visualize_flow(u, v)
    cv2.imshow("Optical Flow", flow_image)
    cv2.imwrite("optical_flow.png", flow_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()