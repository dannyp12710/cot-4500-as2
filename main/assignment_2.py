import numpy as np

def neville_interpolation(x, val, w):

    n = len(x)
    neville = [[0.0 for _ in range(n)] for _ in range(n)]

    # Assign initial values
    for i in range(n):
        neville[i][0] = val[i]

    # Calculate interpolated values
    for i in range(1, n):
        for j in range(1, i + 1):
            term1 = (w - x[i - j]) * neville[i][j - 1]
            term2 = (w - x[i]) * neville[i - 1][j - 1]
            neville[i][j] = (term1 - term2) / (x[i] - x[i - j])

    return neville

# Define the data points
x = [3.6, 3.8, 3.9]
val = [1.675, 1.436, 1.318]

# Define the point at which to interpolate
w = 3.7

# Calculate the interpolated value
neville_values = neville_interpolation(x, val, w)

# Print the interpolated value
print("Neville's method of f(3.7)=", neville_values[-1][-1])


import numpy as np

def newton_forward_difference(xi, fxi):
    
    n = len(xi)
    diffs = [[0.0 for _ in range(n)] for _ in range(n)]

    # Assign initial values
    for i in range(n):
        diffs[i][0] = fxi[i]

    # Calculate forward differences
    for j in range(1, n):  
        for i in range(n - j):  
            diffs[i][j] = diffs[i + 1][j - 1] - diffs[i][j - 1]

    return diffs

def newton_forward_interpolation(xi, diffs, x):
    
    n = len(xi)
    h = xi[1] - xi[0]  
    u = (x - xi[0]) / h  

    # Compute interpolation using forward difference formula
    fx = diffs[0][0]  
    term = 1  
    factorial = 1 

    for degree in range(1, n):
        term *= (u - (degree - 1))  
        factorial *= degree  
        fx += (diffs[0][degree] * term) / factorial  

    return fx

def print_newton_forward_coefficients(diffs):
    
    print("Newton's Forward Coefficients:")
    for degree in range(1, 4):
        print(diffs[0][degree])  

# Define the data points
xi = [7.2, 7.4, 7.5, 7.6]
fxi = [23.5492, 25.3913, 26.8224, 27.4589] 

# Calculate the forward differences
diffs = newton_forward_difference(xi, fxi)

# Print the Newton's Forward Coefficients
print_newton_forward_coefficients(diffs)

# Approximate f(7.3)
x_target = 7.3
f_approx = newton_forward_interpolation(xi, diffs, x_target)
print(f"\nf({x_target})= {f_approx}")

import numpy as np

def hermite_polynomial_approximation(xi, fxi, dfxi):
    n = len(xi)
    m = 2 * n
    coeffs = np.zeros((m, 5))  
    z = np.zeros(m)
    fz = np.zeros(m)
    
    for i in range(n):
        z[2 * i] = z[2 * i + 1] = xi[i]
        fz[2 * i] = fz[2 * i + 1] = fxi[i]
    
    coeffs[:, 0] = z
    coeffs[:, 1] = fz
    
    for i in range(n):
        coeffs[2 * i + 1, 2] = dfxi[i]
        if i > 0:
            coeffs[2 * i, 2] = (fz[2 * i] - fz[2 * i - 1]) / (z[2 * i] - z[2 * i - 1])
    
    for j in range(3, 5):  
        for i in range(j, m):
            coeffs[i, j] = (coeffs[i, j - 1] - coeffs[i - 1, j - 1]) / (z[i] - z[i - j + 1])
    
    return coeffs

def print_hermite_polynomial_approximation(coeffs):
    
    print("Hermite Polynomial Approximation Matrix:")
    for i, row in enumerate(coeffs):
        formatted_row = []
        for j, x in enumerate(row):
            if j == 0: 
                formatted_row.append(f"{x:.8e}")
            elif j == 1:  
                formatted_row.append(f"{x:.8e}")
            elif j == 2:  
                formatted_row.append(f"{x:.8e}")
            elif j == 3:  
                formatted_row.append(f"{x:.8e}")
            else:  
                formatted_row.append(f"{x:.8e}")
        print(formatted_row)

# Define the data points
xi = [3.6, 3.8, 3.9]
fxi = [1.675, 1.436, 1.318]
dfxi = [-1.195, -1.188, -1.182]

# Compute Hermite divided difference table
coeffs = hermite_polynomial_approximation(xi, fxi, dfxi)

# Print the Hermite polynomial approximation matrix
print_hermite_polynomial_approximation(coeffs)

import numpy as np

def cubic_spline_matrix(x, y):
    n = len(x) - 1  
    h = np.diff(x)  
    
    # Construct matrix A
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    
    A[0, 0] = A[n, n] = 1  
    
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    
    return A, b

# Given data points
x = np.array([2, 5, 8, 10])
y = np.array([3, 5, 7, 9])

# Compute matrix A and vector b
A, b = cubic_spline_matrix(x, y)

# Solve for vector x (second derivatives of the spline)
x_sol = np.linalg.solve(A, b)

print("Matrix A:")
print(A)
print("\nVector b:")
print(b)
print("\nVector x:")
print(x_sol)


