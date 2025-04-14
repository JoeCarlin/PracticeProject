import numpy as np
import matplotlib.pyplot as plt

# Define the points
points = [(0, -3), (2, 5), (4, -1), (6, 18)]
x_points = np.array([p[0] for p in points])
y_points = np.array([p[1] for p in points])

# Function to calculate the Lagrange polynomial
def lagrange_polynomial(x, x_points, y_points):
    n = len(x_points)
    result = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

# Generate x values for plotting
x_fit = np.linspace(min(x_points) - 1, max(x_points) + 1, 500)
y_fit = [lagrange_polynomial(x, x_points, y_points) for x in x_fit]

# Print the Lagrange polynomial equation step by step
print("Step-by-Step Lagrange Polynomial Calculation:")
equation = ""
for i in range(len(x_points)):
    print(f"\nStep {i + 1}: Calculating term for point ({x_points[i]}, {y_points[i]})")
    term = f"{y_points[i]:.4f}"
    term_steps = [f"{y_points[i]:.4f}"]
    for j in range(len(x_points)):
        if j != i:
            factor = f"(x - {x_points[j]}) / ({x_points[i]} - {x_points[j]})"
            term_steps.append(factor)
            term += f" * {factor}"
    print(f"  Term {i + 1}: {' * '.join(term_steps)}")
    if i == 0:
        equation += f"({term})"
    else:
        equation += f" + ({term})"

print("\nFinal Lagrange Polynomial:")
print(f"P(x) = {equation}")

# Plot the points and the Lagrange polynomial
plt.scatter(x_points, y_points, color='red', label='Given Points')
plt.plot(x_fit, y_fit, color='blue', label='Lagrange Polynomial')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Lagrange Polynomial Fit')
plt.legend()
plt.grid()

# Save the plot to a file
plt.savefig("lagrange_polynomial_plot.png")
plt.show()