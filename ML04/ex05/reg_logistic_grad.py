import numpy as np

def reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray, with two for-loops. The three array
    Args:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    Raises:
    This function should not raise any Exception.
    """

    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)\
            or not isinstance(y, np.ndarray) or x.size == 0 or theta.size == 0\
            or not np.issubdtype(x.dtype, np.number) or not np.issubdtype(theta.dtype, np.number)\
            or not np.issubdtype(y.dtype, np.number) or x.shape[0] != y.shape[0]\
            or (len(x.shape) < 2 or len(theta.shape) < 2 or (x.shape[1] != theta.shape[0] - 1)):
        return None

    X = np.insert(x, 0, 1, axis=1)

    y_hat = 1 / (1 + np.exp(-(X @ theta)))
    reg_log_grad = [(sum(((y_hat[i] - y[i]) * X[i][j]) for i in range(X.shape[0]))
            + (lambda_ * (theta[j] if j > 0 else 0))) / y.shape[0] for j in range(X.shape[1])]
    return np.array(reg_log_grad).reshape(-1, 1)

def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray, without any for-loop. The three arr
    Args:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of shape m * n.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    Raises:
    This function should not raise any Exception.
    """

    if not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray)\
            or not isinstance(x, np.ndarray) or not np.issubdtype(y.dtype, np.number)\
            or not np.issubdtype(x.dtype, np.number) or not np.issubdtype(theta.dtype, np.number)\
            or not isinstance(lambda_, (int, float)) or x.shape[0] != y.shape[0]\
            or x.shape[1] != theta.shape[0] - 1 or y.size == 0 or theta.size == 0:
        return None

    theta2 = np.copy(theta)
    theta2[0, 0] = 0
    X = np.insert(x, 0, 1, axis=1)
    
    return (X.T @ ((1 / (1 + np.exp(-(X @ theta)))) - y) + (lambda_ * theta2)) / y.shape[0]

if __name__ == '__main__':
    x = np.array([[0, 2, 3, 4],
    [2, 4, 5, 5],
    [1, 3, 2, 7]])
    y = np.array([[0], [1], [1]])
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    
    # Example 1.1:
    print('\n********************* Example 1.1 ***** reg_logistic_grad *****')
    print(repr(reg_logistic_grad(y, x, theta, 1)))
    # Output:
    # array([[-0.55711039],
    # [-1.40334809],
    # [-1.91756886],
    # [-2.56737958],
    # [-3.03924017]])
    
    # Example 1.2:
    print('\n--------------------- Example 1.2 --- vec_reg_logistic_grad ---')
    print(repr(vec_reg_logistic_grad(y, x, theta, 1)), "\n")
    # Output:
    # array([[-0.55711039],
    # [-1.40334809],
    # [-1.91756886],
    # [-2.56737958],
    # [-3.03924017]])
    
    # Example 2.1:
    print('\n********************* Example 2.1 ***** reg_logistic_grad *****')
    print(repr(reg_logistic_grad(y, x, theta, 0.5)))
    # Output:
    # array([[-0.55711039],
    # [-1.15334809],
    # [-1.96756886],
    # [-2.33404624],
    # [-3.15590684]])
    
    # Example 2.2:
    print('\n--------------------- Example 2.2 --- vec_reg_logistic_grad ---')
    print(repr(vec_reg_logistic_grad(y, x, theta, 0.5)), "\n")
    # Output:
    # array([[-0.55711039],
    # [-1.15334809],
    # [-1.96756886],
    # [-2.33404624],
    # [-3.15590684]])
    
    # Example 3.1:
    print('\n********************* Example 3.1 ***** reg_logistic_grad *****')
    print(repr(reg_logistic_grad(y, x, theta, 0.0)))
    # Output:
    # array([[-0.55711039],
    # [-0.90334809],
    # [-2.01756886],
    # [-2.10071291],
    # [-3.27257351]])
    
    # Example 3.2:
    print('\n--------------------- Example 3.2 --- vec_reg_logistic_grad ---')
    print(repr(vec_reg_logistic_grad(y, x, theta, 0.0)))
    # Output:
    # array([[-0.55711039],
    # [-0.90334809],
    # [-2.01756886],
    # [-2.10071291],
    # [-3.27257351]])