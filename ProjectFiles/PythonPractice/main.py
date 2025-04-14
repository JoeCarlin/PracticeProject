import numpy as np

# Given points
x = np.array([0, 2, 4, 6])
y = np.array([-3, 5, -1, 18])

# Use numpy to fit a cubic polynomial (degree = 3)
coeffs = np.polyfit(x, y, 3)

# Display the polynomial coefficients: ax^3 + bx^2 + cx + d
a, b, c, d = coeffs
print(f"The cubic polynomial is:\ny = {a:.4f}x^3 + {b:.4f}x^2 + {c:.4f}x + {d:.4f}")