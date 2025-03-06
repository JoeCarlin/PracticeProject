def determinant_using_gaussian_elimination(matrix):
    n = len(matrix)
    # Step 1: Create a copy of the matrix to avoid modifying the original matrix
    mat = [row[:] for row in matrix]
    determinant = 1
    sign = 1
    
    for i in range(n):
        # Step 2: Find the pivot element
        if mat[i][i] == 0:
            for k in range(i + 1, n):
                if mat[k][i] != 0:
                    mat[i], mat[k] = mat[k], mat[i]
                    sign *= -1  # Swapping rows changes the sign of the determinant
                    break
            else:
                return 0  # Singular matrix, determinant is 0
        
        # Step 3: Perform elimination below the pivot
        for k in range(i + 1, n):
            factor = mat[k][i] / mat[i][i]
            for j in range(i, n):
                mat[k][j] -= factor * mat[i][j]
    
    # Step 4: Multiply the diagonal elements
    for i in range(n):
        determinant *= mat[i][i]
    
    # Step 5: Return the determinant, adjusted for any row swaps
    return determinant * sign

# Example usage
matrix = [
    [1, -1, 0],
    [-2, 2, -1],
    [0, 1, -2]
]

det = determinant_using_gaussian_elimination(matrix)
print("Determinant:", det)