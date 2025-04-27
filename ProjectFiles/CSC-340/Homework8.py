import math
import cv2
import numpy as np

# blockSize
bkS = 8
# scale up blocks for easier visualization
mag = 10
    
for u in range(bkS):
    for v in range(bkS):
        # store computed DCT values in this variable
        image = np.zeros((bkS*mag, bkS*mag), np.float32)
        # update these values with the smallest and largest DCT value found for this block
        maxV = -100000.0
        minV = 100000.0
        
        # Calculate the DCT filter for the (u, v) index
        # Generate an 8x8 block (f(x, y)) where the pixel values are all initially set to 1
        block = np.ones((bkS, bkS), np.float32)

        # DCT scaling factors for u, v
        def alpha(n):
            return 1/np.sqrt(8) if n == 0 else np.sqrt(2)/np.sqrt(8)
        
        # Calculate the DCT filter for the (u, v) coefficient
        for x in range(bkS):
            for y in range(bkS):
                sum_val = 0.0
                for i in range(bkS):
                    for j in range(bkS):
                        sum_val += block[i][j] * np.cos(((2*i+1)*u*np.pi) / 16) * np.cos(((2*j+1)*v*np.pi) / 16)
                
                # Apply the scaling factors
                image[x][y] = alpha(u) * alpha(v) * sum_val
        
        # Update max and min values for the DCT
        maxV = np.max(image)
        minV = np.min(image)
        
# Map DCT values to [0-255] for visualization        
imageN = np.zeros((bkS*mag, bkS*mag), np.float32)
if maxV == minV:
    for x in range(bkS*mag):
        for y in range(bkS*mag):
            imageN[y][x] = 255.0
else:
    for x in range(bkS):
        for m in range(mag):
            for y in range(bkS):
                val = 255.0 * (image[y][x] - minV) / (maxV - minV)
                for n in range(mag):
                    imageN[mag * y + n][mag * x + m] = val

# Convert imageN to 8-bit unsigned integers
imageN = np.clip(imageN, 0, 255).astype(np.uint8)

# Save the magnified image of the DCT filter
name = "dctPatches/dctPatch_" + str(u) + "_" + str(v) + ".png"
cv2.imwrite(name, imageN)