def gauss_jordan_elimination(A, b):
    n = len(A)

    # Step 1: Create the augmented matrix
    augmented_matrix = [A[i] + [b[i]] for i in range(n)]

    # Step 2: Perform the elimination process with partial pivoting
    for i in range(n):
        # Partial pivoting: find row with max abs value in current column
        max_row = max(range(i, n), key=lambda r: abs(augmented_matrix[r][i]))
        if augmented_matrix[max_row][i] == 0:
            raise ValueError("System has no unique solution.")

        # Swap rows if needed
        if max_row != i:
            augmented_matrix[i], augmented_matrix[max_row] = augmented_matrix[max_row], augmented_matrix[i]

        # Normalize pivot row
        pivot = augmented_matrix[i][i]
        augmented_matrix[i] = [x / pivot for x in augmented_matrix[i]]

        # Eliminate other rows
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


def gaussian_elimination(matrix, vector):
    n = len(matrix)

    # Step 1: Create the augmented matrix
    augmented_matrix = [matrix[i] + [vector[i]] for i in range(n)]

    # Step 2: Forward elimination with partial pivoting
    for i in range(n):
        # Partial pivoting
        max_row = max(range(i, n), key=lambda r: abs(augmented_matrix[r][i]))
        if augmented_matrix[max_row][i] == 0:
            raise ValueError("Matrix is singular or nearly singular.")

        if max_row != i:
            augmented_matrix[i], augmented_matrix[max_row] = augmented_matrix[max_row], augmented_matrix[i]

        # Eliminate elements below the pivot
        for k in range(i + 1, n):
            factor = augmented_matrix[k][i] / augmented_matrix[i][i]
            for j in range(i, n + 1):
                augmented_matrix[k][j] -= factor * augmented_matrix[i][j]

    # Step 3: Back substitution
    solution = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        sum_ax = sum(augmented_matrix[i][j] * solution[j] for j in range(i + 1, n))
        solution[i] = (augmented_matrix[i][n] - sum_ax) / augmented_matrix[i][i]

    return solution


def gauss_jordan_inverse_and_determinant(matrix):
    n = len(matrix)

    # Step 1: Create the augmented matrix by appending the identity matrix
    augmented_matrix = [matrix[i] + [1 if i == j else 0 for j in range(n)] for i in range(n)]

    determinant = 1

    # Step 2: Perform the elimination process with partial pivoting
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(augmented_matrix[r][i]))
        if augmented_matrix[max_row][i] == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")

        if max_row != i:
            augmented_matrix[i], augmented_matrix[max_row] = augmented_matrix[max_row], augmented_matrix[i]
            determinant *= -1

        pivot = augmented_matrix[i][i]
        determinant *= pivot

        augmented_matrix[i] = [x / pivot for x in augmented_matrix[i]]

        for k in range(n):
            if k != i:
                factor = augmented_matrix[k][i]
                augmented_matrix[k] = [
                    augmented_matrix[k][j] - factor * augmented_matrix[i][j]
                    for j in range(len(augmented_matrix[i]))
                ]

    inverse_matrix = [row[n:] for row in augmented_matrix]
    return inverse_matrix, determinant


def determinant_gaussian_elimination(matrix):
    n = len(matrix)
    augmented_matrix = [row[:] for row in matrix]  # Create a copy of the matrix
    determinant = 1

    # Forward elimination with partial pivoting
    for i in range(n):
        # Partial pivoting: find row with the max abs value in the current column
        max_row = max(range(i, n), key=lambda r: abs(augmented_matrix[r][i]))
        if augmented_matrix[max_row][i] == 0:
            return 0  # The matrix is singular, determinant is 0

        if max_row != i:
            augmented_matrix[i], augmented_matrix[max_row] = augmented_matrix[max_row], augmented_matrix[i]
            determinant *= -1  # Each row swap changes the sign of the determinant

        pivot = augmented_matrix[i][i]
        determinant *= pivot

        # Eliminate elements below the pivot
        for k in range(i + 1, n):
            factor = augmented_matrix[k][i] / augmented_matrix[i][i]
            for j in range(i, n):
                augmented_matrix[k][j] -= factor * augmented_matrix[i][j]

    # The determinant is the product of the pivots
    return determinant


if __name__ == "__main__":
    # Solve system using Gauss-Jordan Elimination
    A = [
        [1, 0, 2],
        [2, -1, 3],
        [4, 1, 8]
    ]
    b = [1, -1, 2]

    print("Using Gauss-Jordan Elimination:")
    try:
        sol = gauss_jordan_elimination(A, b)
        print(f"Solution: x = {sol[0]:.4f}, y = {sol[1]:.4f}, z = {sol[2]:.4f}")
    except ValueError as e:
        print("Error:", e)

    print("\nUsing Gaussian Elimination:")
    try:
        solution = gaussian_elimination(A, b)
        print("Solution Vector:", solution)
    except ValueError as e:
        print("Error:", e)

    # Inverse and determinant using Gauss-Jordan
    matrix = [
        [1, -1, 0],
        [-2, 2, -1],
        [0, 1, -2]
    ]

    print("\nInverse and Determinant using Gauss-Jordan:")
    try:
        inverse, determinant = gauss_jordan_inverse_and_determinant(matrix)
        print("Inverse Matrix:")
        for row in inverse:
            print(row)
        print(f"Determinant: {determinant:.4f}")
    except ValueError as e:
        print("Error:", e)

    print("\nDeterminant using Gaussian Elimination:")
    try:
        det = determinant_gaussian_elimination(matrix)
        print(f"Determinant: {det:.4f}")
    except ValueError as e:
        print("Error:", e)