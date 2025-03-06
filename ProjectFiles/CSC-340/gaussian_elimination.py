def gaussian_elimination(matrix, vector):
    n = len(matrix)
    
    # Step 1: Create the augmented matrix by appending the vector of constants
    augmented_matrix = [matrix[i] + [vector[i]] for i in range(n)]
    
    # Step 2: Perform the elimination process to get an upper triangular matrix
    for i in range(n):
        # Step 2a: Make the diagonal element non-zero by swapping rows if necessary
        if augmented_matrix[i][i] == 0:
            for k in range(i + 1, n):
                if augmented_matrix[k][i] != 0:
                    augmented_matrix[i], augmented_matrix[k] = augmented_matrix[k], augmented_matrix[i]
                    break
            else:
                raise ValueError("Matrix is singular or nearly singular.")
        
        # Step 2b: Eliminate the elements below the diagonal
        for k in range(i + 1, n):
            factor = augmented_matrix[k][i] / augmented_matrix[i][i]
            for j in range(i, n + 1):
                augmented_matrix[k][j] -= factor * augmented_matrix[i][j]
    
    # Step 3: Perform back substitution to find the solution vector
    solution = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        solution[i] = augmented_matrix[i][n] / augmented_matrix[i][i]
        for k in range(i - 1, -1, -1):
            augmented_matrix[k][n] -= augmented_matrix[k][i] * solution[i]
    
    return solution

# Example usage
matrix = [
    [1, 0, 2],
    [2, -1, 3],
    [4, 1, 8]
]
vector = [1, -1, 2]

try:
    solution = gaussian_elimination(matrix, vector)

    # Print the solution vector
    print("Solution Vector using Gaussian Elimination:")
    print(solution)
except ValueError as e:
    print(e)