def gauss_jordan_inverse_and_determinant(matrix):
    n = len(matrix)
    
    # Step 1: Create the augmented matrix by appending the identity matrix
    augmented_matrix = [matrix[i] + [1 if i == j else 0 for j in range(n)] for i in range(n)]
    
    determinant = 1
    
    # Step 2: Perform the elimination process
    for i in range(n):
        # Step 2a: Make the diagonal element 1
        diag = augmented_matrix[i][i]
        if diag == 0:
            # Try to swap with a non-zero row below
            for k in range(i + 1, n):
                if augmented_matrix[k][i] != 0:
                    augmented_matrix[i], augmented_matrix[k] = augmented_matrix[k], augmented_matrix[i]
                    diag = augmented_matrix[i][i]
                    determinant *= -1  # Swapping rows changes the sign of the determinant
                    break
            else:
                raise ValueError("Matrix is singular and cannot be inverted.")
        
        determinant *= diag
        
        for j in range(len(augmented_matrix[i])):
            augmented_matrix[i][j] = augmented_matrix[i][j] / diag
        
        # Step 2b: Make the other elements in the current column 0
        for k in range(n):
            if k != i:
                factor = augmented_matrix[k][i]
                for j in range(len(augmented_matrix[i])):
                    augmented_matrix[k][j] = augmented_matrix[k][j] - factor * augmented_matrix[i][j]
    
    # Step 3: Extract the inverse matrix
    inverse_matrix = [row[n:] for row in augmented_matrix]
    
    return inverse_matrix, determinant

# Example usage
matrix = [
    [1, -1, 0],
    [-2, 2, -1],
    [0, 1, -2]
]

try:
    inverse, determinant = gauss_jordan_inverse_and_determinant(matrix)

    # Print the inverse matrix
    print("Inverse Matrix:")
    for row in inverse:
        print(row)

    # Print the determinant
    print(f"Determinant: {determinant:.4f}")
except ValueError as e:
    print(e)