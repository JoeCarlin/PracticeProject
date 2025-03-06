# Define the matrix and the constants vector
A = [
    [1, 0, 2],
    [2, -1, 3],
    [4, 1, 8]
]

b = [1, -1, 2]

# Function to perform Gauss-Jordan elimination
def gauss_jordan_elimination(A, b):
    n = len(A)
    
    # Augment the matrix A with the vector b
    for i in range(n):
        A[i].append(b[i])
    
    # Perform the elimination process
    for i in range(n):
        # Make the diagonal element 1
        diag = A[i][i]
        for j in range(len(A[i])):
            A[i][j] = A[i][j] / diag
        
        # Make the other elements in the current column 0
        for k in range(n):
            if k != i:
                factor = A[k][i]
                for j in range(len(A[i])):
                    A[k][j] = A[k][j] - factor * A[i][j]
    
    # Extract the solution
    solution = [A[i][-1] for i in range(n)]
    return solution

# Test the function with the given matrix
solution = gauss_jordan_elimination(A, b)

# Print the solution
print(f"Solution: x = {solution[0]:.4f}, y = {solution[1]:.4f}, z = {solution[2]:.4f}")