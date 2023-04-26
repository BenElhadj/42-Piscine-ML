import numpy as np

def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, with a for-loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.array, a vector of shape 2 * 1.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """

    if not isinstance(x, np.ndarray) or x.size == 0\
        or not isinstance(y, np.ndarray) or y.size == 0\
        or not isinstance(theta, np.ndarray) or theta.size == 0\
        or x.shape[0] != y.shape[0] or theta.shape[0] - 1 != x.shape[1]:
        return None

    X = np.insert(x, 0, 1, axis=1)
    y_pred = []
    sum1_line = 0
    sum2_line = 0

    for i in range(x.shape[0]):
        sum1_line += ((x[i] * theta[1] + theta[0]) - y[i]) / x.shape[0]
        sum2_line += ((x[i] * theta[1] + theta[0]) - y[i]) * x[i] / x.shape[0]
    y_pred.extend([float(sum1_line), float(sum2_line)])

    return np.array(y_pred).reshape(-1, 1)

if __name__ == '__main__':
    import numpy as np
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))
    # Example 0:
    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    print(simple_gradient(x, y, theta1), '\n')
    # Output:
    # array([[-19.0342574], [-586.66875564]])
    # Example 1:
    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    print(simple_gradient(x, y, theta2).shape, '\n')
    # Output:
    # array([[-57.86823748], [-2230.12297889]])