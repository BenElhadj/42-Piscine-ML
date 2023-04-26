import numpy as np

def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray,
    with two for-loop. The three arrays must have compatible shapes.
    Args:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
    Return:
    A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """

    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)\
            or not isinstance(y, np.ndarray) or x.size == 0 or theta.size == 0\
            or not np.issubdtype(x.dtype, np.number) or not np.issubdtype(theta.dtype, np.number)\
            or not np.issubdtype(y.dtype, np.number) or x.shape[0] != y.shape[0]\
            or (len(x.shape) < 2 or len(theta.shape) < 2 or (x.shape[1] != theta.shape[0] - 1)):
        return None

    y_hat = []

    for i in range(x.shape[0]):
        sum_line = sum(x[i][j] * theta[j + 1] for j in range(x.shape[1]))
        y_hat.append([float(sum_line + theta[0])])

    X = np.insert(x, 0, 1, axis=1)
    gradient = [
        (
            sum(((y_hat[i] - y[i]) * X[i][j]) for i in range(X.shape[0]))
            + (lambda_ * (theta[j] if j > 0 else 0))
        )
        / y.shape[0]
        for j in range(X.shape[1])
    ]
    gradient = np.array(gradient).reshape(-1, 1)

    return gradient

def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have compatible shapes.
    Args:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
    Return:
    A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """

    if not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray)\
            or not isinstance(x, np.ndarray) or not np.issubdtype(y.dtype, np.number)\
            or not np.issubdtype(x.dtype, np.number) or not np.issubdtype(theta.dtype, np.number)\
            or not isinstance(lambda_, (int, float)) or x.shape[0] != y.shape[0]\
            or x.shape[1] != theta.shape[0] - 1 or y.size == 0 or theta.size == 0:
        return None

    X = np.insert(x, 0, 1, axis=1)
    
    theta_with_0 = np.copy(theta)
    theta_with_0[0, 0] = 0
    
    return (X.T @ ((X @ theta) - y) + (lambda_ * theta_with_0)) / y.shape[0]

if __name__ == '__main__':

    x = np.array([
    [ -6, -7, -9],
    [ 13, -2, 14],
    [ -7, 14, -1],
    [ -8, -4, 6],
    [ -5, -9, 6],
    [ 1, -5, 11],
    [ 9, -11, 8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])
    
    # Example 1.1:
    print('\n********************* Example 1.1 ***** reg_linear_grad *****')
    print(reg_linear_grad(y, x, theta, 1))
    # Output:
    # array([[ -60.99 ],
    # [-195.64714286],
    # [ 863.46571429],
    # [-644.52142857]])
    
    # Example 1.2:
    print('\n--------------------- Example 1.2 --- vec_reg_linear_grad ---')
    print(vec_reg_linear_grad(y, x, theta, 1))
    # Output:
    # array([[ -60.99 ],
    # [-195.64714286],
    # [ 863.46571429],
    # [-644.52142857]])
    
    # Example 2.1:
    print('\n********************* Example 2.1 ***** reg_linear_grad *****')
    print(reg_linear_grad(y, x, theta, 0.5))
    # Output:
    # array([[ -60.99 ],
    # [-195.86142857],
    # [ 862.71571429],
    # [-644.09285714]])
    
    # Example 2.2:
    print('\n--------------------- Example 2.2 --- vec_reg_linear_grad ---')
    print(vec_reg_linear_grad(y, x, theta, 0.5))
    # Output:
    # array([[ -60.99 ],
    # [-195.86142857],
    # [ 862.71571429],
    # [-644.09285714]])
    
    # Example 3.1:
    print('\n********************* Example 3.1 ***** reg_linear_grad *****')
    print(reg_linear_grad(y, x, theta, 0.0))
    # Output:
    # array([[ -60.99 ],
    # [-196.07571429],
    # [ 861.96571429],
    # [-643.66428571]])
    
    # Example 3.2:
    print('\n--------------------- Example 3.2 --- vec_reg_linear_grad ---')
    print(vec_reg_linear_grad(y, x, theta, 0.0), '\n')
    # Output:
    # array([[ -60.99 ],
    # [-196.07571429],
    # [ 861.96571429],
    # [-643.66428571]]