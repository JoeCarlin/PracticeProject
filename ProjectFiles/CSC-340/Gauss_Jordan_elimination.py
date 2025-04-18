def gauss_jordan_elimination(A, b):
    n = len(A)

    # Step 1: Create the augmented matrix
    augmented_matrix = [A[i] + [b[i]] for i in range(n)]

    # Step 2: Perform the elimination process
    for i in range(n):
        # Step 2a: Find the row with the largest absolute value in column i
        max_row = max(range(i, n), key=lambda r: abs(augmented_matrix[r][i]))
        if augmented_matrix[max_row][i] == 0:
            raise ValueError("No unique solution exists.")

        # Step 2b: Swap the current row with the max_row
        if max_row != i:
            augmented_matrix[i], augmented_matrix[max_row] = augmented_matrix[max_row], augmented_matrix[i]

        # Step 2c: Normalize the pivot row
        diag = augmented_matrix[i][i]
        augmented_matrix[i] = [x / diag for x in augmented_matrix[i]]

        # Step 2d: Eliminate the current column in other rows
        for k in range(n):
            if k != i:
                factor = augmented_matrix[k][i]
                augmented_matrix[k] = [
                    augmented_matrix[k][j] - factor * augmented_matrix[i][j]
                    for j in range(len(augmented_matrix[i]))
                ]

    # Step 3: Extract the solution
    solution = [augmented_matrix[i][-1] for i in range(n)]
    return solution

# Define the matrix and the constants vector
A = [
    [1, 0, 2],
    [2, -1, 3],
    [4, 1, 8]
]

b = [1, -1, 2]

# Test the function
solution = gauss_jordan_elimination(A, b)
print(f"Solution: x = {solution[0]:.4f}, y = {solution[1]:.4f}, z = {solution[2]:.4f}")