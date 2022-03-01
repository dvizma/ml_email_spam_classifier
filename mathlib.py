#######################################################
#                 ML Assignment 2                     #
#              Dvizma Sinha, CS20M504                 #
#                CS5011W, 25/05/2021                  #
#######################################################

"""This module performs math operations. Although it uses np.array as 
base type, only below functions are used fro numpy:
    1. Zeros
    2. shape attribute
"""

import numpy as np


def transpose(a: np.array) -> np.array:
    """This function performs the transpose of a matrix

    Args:
        a (np.array): input matrix

    Returns:
        np.array: output matrix
    """
    m, n = a.shape

    result = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            result[i][j] = a[j][i]

    return result


def matmul(a: np.array, b: np.array) -> np.array:
    """Perform matrix multiplication

    Args:
        a (np.array): input matrix 1
        b (np.array): input matrix 2

    Returns:
        np.array: output matrix
    """
    # Matrix 1 Dimensions : M X K
    # Matrix 2 Dimensions : K X N
    M, K1 = a.shape
    K2, N = b.shape

    assert K1 == K2, f"Shapes {a.shape}, {b.shape} not compatible with transpose"
    K = K1

    result = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            for k in range(K):
                result[i][j] += a[i][k] * b[k][j]

    return result


def sub(x: np.array, y: np.array) -> np.array:
    """Perform elementwise subtraction on matrices, vectors

    Args:
        x (np.array): input array 1
        y (np.array): input array 2

    Raises:
        ValueError: input has more than 2 dimensions

    Returns:
        np.array: x - y array
    """
    assert x.shape == y.shape

    # If x and y are vectors
    if len(x.shape) == 1:
        z = np.zeros(x.shape)
        n = x.shape[0]
        for i in range(n):
            z[i] = x[i] - y[i]

        return z

    # if x and y are matrices
    if len(x.shape) == 2:
        m, n = x.shape
        z = np.zeros(x.shape)
        for i in range(m):
            for j in range(n):
                z[i][j] = x[i][j] - y[i][j]

        return z

    raise ValueError(f"{len(x.shape)}D array not supported")


def div(x: np.array, y: np.array) -> np.array:
    """Perform elementwise division on matrices, vectors

    Args:
        x (np.array): input array 1
        y (np.array): input array 2

    Raises:
        ValueError: input has more than 2 dimensions

    Returns:
        np.array: x / y array
    """
    assert x.shape == y.shape

    # If x and y are vectors
    if len(x.shape) == 1:
        z = np.zeros(x.shape)
        n = x.shape[0]
        for i in range(n):
            z[i] = x[i] / y[i]

        return z

    # if x and y are matrices
    if len(x.shape) == 2:
        m, n = x.shape
        z = np.zeros(x.shape)
        for i in range(m):
            for j in range(n):
                z[i][j] = x[i][j] / y[i][j]

        return z

    raise ValueError(f"{len(x.shape)}D array not supported")


def add(x: np.array, y: np.array) -> np.array:
    """Perform elementwise addition on matrices, vectors

    Args:
        x (np.array): input array 1
        y (np.array): input array 2

    Raises:
        ValueError: input has more than 2 dimensions

    Returns:
        np.array: x + y array
    """
    assert x.shape == y.shape

    # If x and y are vectors
    if len(x.shape) == 1:
        z = np.zeros(x.shape)
        n = x.shape[0]
        for i in range(n):
            z[i] = x[i] + y[i]

        return z

    # if x and y are matrices
    if len(x.shape) == 2:
        m, n = x.shape
        z = np.zeros(x.shape)
        for i in range(m):
            for j in range(n):
                z[i][j] = x[i][j] + y[i][j]

        return z

    raise ValueError(f"{len(x.shape)}D array not supported")


def mul(x: np.array, y: np.array) -> np.array:
    """Perform elementwise multiplication on matrices, vectors

    Args:
        x (np.array): input array 1
        y (np.array): input array 2

    Raises:
        ValueError: input has more than 2 dimensions

    Returns:
        np.array: x * y array
    """
    assert x.shape == y.shape

    # If x and y are vectors
    if len(x.shape) == 1:
        z = np.zeros(x.shape)
        n = x.shape[0]
        for i in range(n):
            z[i] = x[i] * y[i]

        return z

    # if x and y are matrices
    if len(x.shape) == 2:
        m, n = x.shape
        z = np.zeros(x.shape)
        for i in range(m):
            for j in range(n):
                z[i][j] = x[i][j] * y[i][j]

        return z

    raise ValueError(f"{len(x.shape)}D array not supported")


def l2_norm(x):
    """Calculate L2 norm of a given vector

    Args:
        x (np.array): input vector

    Returns:
        float: L2 norm of vector
    """
    x_f = x.reshape(-1,)
    sum_sq = 0
    for i in x_f:
        sum_sq += i * i

    return sum_sq ** 0.5


def matrix_copy(x):
    """Create a copy of given matrix"""
    n, m = x.shape
    x_copy = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            x_copy[i][j] = x[i][j]

    return x_copy


def identity_matrix(n):
    """Create an identity matrix of size NXN"""
    x = np.zeros((n, n))
    for i in range(n):
        x[i][i] = 1

    return x


def matrix_inverse(x):
    """Calculate inverse of a matrix, matrix must be non singular"""

    m, n = x.shape
    assert m == n, f"X should be a square matrix"

    # All operations to be performed on copy matrix
    x_copy = matrix_copy(x)
    i_matrix = identity_matrix(m)

    index = list(range(n))

    # For every diagonal
    for d in range(n):
        d_val = x_copy[d][d]

        # scale each element of the row by d_scale so that
        # the diagonal element becomes 1
        # perform same operation on the identity matrix as well
        for j in range(n):
            x_copy[d][j] = x_copy[d][j] / d_val
            i_matrix[d][j] = i_matrix[d][j] / d_val

        # set all other elements of the column d to be 0
        for i in index[:d] + index[d+1:]:
            scale = x_copy[i][d]

            for j in range(n):
                x_copy[i][j] = x_copy[i][j] - scale * x_copy[d][j]
                i_matrix[i][j] = i_matrix[i][j] - scale * i_matrix[d][j]

    # i_matrix now contains the inverse
    return i_matrix

def exp(x: np.array) -> np.array:
    """Calculate e ^ x for an array x

    Args:
        x (np.array): Input array

    Returns:
        np.array: output array
    """
    e = 2.718281828459045
    y = np.zeros(x.shape)

    n,_ = x.shape
    for i in range(n):
        y[i] = e ** x[i]
    
    return y