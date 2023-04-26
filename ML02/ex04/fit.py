import numpy as np
import sys
sys.path.insert(0, "../ex01")
from prediction import predict_ #type: ignore

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.array, a matrix of dimension m * n:
    (number of training examples, number of features).
    y: has to be a numpy.array, a vector of dimension m * 1:
    (number of training examples, 1).
    theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
    (number of features + 1, 1).
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
    new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
    None if there is a matching dimension problem.
    None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
    This function should not raise any Exception.
    """

    if not np.issubdtype(type(max_iter), np.number) or not np.issubdtype(type(alpha), np.number)\
            or not np.issubdtype(x.dtype, np.number) \
            or not np.issubdtype(y.dtype, np.number) \
            or not np.issubdtype(theta.dtype, np.number) \
            or not isinstance(x, np.ndarray) or x.size == 0\
            or not isinstance(y, np.ndarray) or y.size == 0\
            or not isinstance(theta, np.ndarray) or theta.size == 0\
            or x.shape[0] != y.shape[0] or theta.shape[0] - 1 != x.shape[1]:
        return None

    X = np.insert(x, 0, 1, axis=1)
    
    for _ in range(max_iter):
        grad = (X.T @ ((X @ theta) - y)) / y.shape[0]
        theta -= alpha * grad

    return theta

if __name__ == '__main__':
    x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.], [1.], [1.], [1.]])
    # Example 0:
    theta2 = fit_(x, y, theta, alpha = 0.0005, max_iter = 42000)
    print(theta2,'\n')
    # Output:
    # # array([[41.99..],[0.97..], [0.77..], [-1.20..]])
    # Example 1:
    print(predict_(x, theta2),'\n')
    # Output:
    # # array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])