import numpy as np


def vec_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be a numpy.array, a matrix of shape m * 1.
    y: has to be a numpy.array, a vector of shape m * 1.
    theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
    None if x, y, or theta is an empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """

    if not isinstance(x, np.ndarray) or x.size == 0\
        or not isinstance(y, np.ndarray) or y.size == 0\
        or not isinstance(theta, np.ndarray) or theta.size == 0\
        or x.shape[0] != y.shape[0] or theta.shape[0] - 1 != x.shape[1]:
        return None
    
    X = np.insert(x, 0, 1, axis=1)

    return (X.T @ ((X @ theta) - y)) / y.shape[0]

if __name__ == '__main__':
    import numpy as np
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))
    # Example 0:
    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    print(vec_gradient(x, y, theta1), '\n')
    
    # Output:
    # array([[-19.0342...], [-586.6687...]])
    # Example 1:
    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    print(vec_gradient(x, y, theta2), '\n')
    
    # Output:
    # array([[-57.8682...], [-2230.1229...]])