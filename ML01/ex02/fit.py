import numpy as np

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
    new_theta: numpy.ndarray, a vector of dimension 2 * 1.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exception.
    """

    if not isinstance(x, np.ndarray) or x.size == 0\
        or not isinstance(y, np.ndarray) or y.size == 0\
        or not isinstance(theta, np.ndarray) or theta.size == 0\
        or x.shape[0] != y.shape[0] or theta.shape[0] - 1 != x.shape[1]:
        return None
    
    X = np.insert(x, 0, 1, axis=1)
    for _ in range(max_iter):
        theta = theta - alpha * (X.T @ ((X @ theta) - y)) / y.shape[0]

    return theta

def predict_(x, theta):
    
    X = np.insert(x, 0, 1, axis=1)
    return X @ theta


if __name__ == '__main__':
    import numpy as np
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    theta= np.array([1, 1]).reshape((-1, 1))
    # Example 0:
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print(theta1, '\n')
    # Output:
    # # array([[1.40709365],
    # # [1.1150909 ]])
    # Example 1:
    print(predict_(x,theta1), '\n')
    # Output:
    # # array([[15.3408728 ],
    # # [25.38243697],
    # # [36.59126492],
    # # [55.95130097],
    # # [65.53471499]])