# Initialize values for x, y, z
x_k = [0, 0, 0]  # Initial guess [x, y, z] = [0, 0, 0]
numIts = 5  # Number of iterations

# Perform iterations
for i in range(numIts):
    x_k1 = [0, 0, 0]  # New estimates for x, y, z
    
    # estimates for x, y, z using the Jacobi method
    x_k1[0] = (5 - 2*x_k[1]) / 3  # x^{(i)} = (5 - 2y) / 3
    x_k1[1] = (-x_k[0] - 2*x_k[2] - 18) / -5  # y^{(i)} = (-x - 2z - 18) / -5
    x_k1[2] = (-2*x_k[0] + x_k[1] + 7) / 8  # z^{(i)} = (-2x + y + 7) / 8
    
    # Output the results for this iteration
    print(f"Iteration {i+1}:")
    print(f"x^{i+1} = {x_k1[0]:.3f}, y^{i+1} = {x_k1[1]:.3f}, z^{i+1} = {x_k1[2]:.3f}")
    
    # Update x_k for the next iteration
    x_k = x_k1

# Final result after 5 iterations
print("\nFinal Result after 5 iterations:")
print(f"x ≈ {x_k[0]:.3f}, y ≈ {x_k[1]:.3f}, z ≈ {x_k[2]:.3f}")