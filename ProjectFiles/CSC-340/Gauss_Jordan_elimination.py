def gauss_jordan_elimination(A, b):
    n = len(A)
    
    # Step 1: Create the augmented matrix
    augmented_matrix = [A[i] + [b[i]] for i in range(n)]
    
    # Step 2: Perform the elimination process
    for i in range(n):
        # Step 2a: Make the diagonal element 1
        diag = augmented_matrix[i][i]
        for j in range(len(augmented_matrix[i])):
            augmented_matrix[i][j] = augmented_matrix[i][j] / diag
        
        # Step 2b: Make the other elements in the current column 0
        for k in range(n):
            if k != i:
                factor = augmented_matrix[k][i]
                for j in range(len(augmented_matrix[i])):
                    augmented_matrix[k][j] = augmented_matrix[k][j] - factor * augmented_matrix[i][j]
    
    # Step 3: Extract the solution
    solution = [augmented_matrix[i][-1] for i in range(n)]
    return solution

# Define the matrix and the constants vector
A = [
    [1, 0, 2],  # Coefficients of the first equation
    [2, -1, 3], # Coefficients of the second equation
    [4, 1, 8]   # Coefficients of the third equation
]

b = [1, -1, 2] # Constants vector

# Test the function with the given matrix
solution = gauss_jordan_elimination(A, b)

# Print the solution
print(f"Solution: x = {solution[0]:.4f}, y = {solution[1]:.4f}, z = {solution[2]:.4f}")