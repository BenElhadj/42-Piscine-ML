import numpy as np
import sys
sys.path.insert(0, '../ex01')
from log_pred import logistic_predict_ # type: ignore

def vec_log_loss_(y, y_hat, eps=1e-15):
    """
    Compute the logistic loss value.
    Args:
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
    eps: epsilon (default=1e-15)
    Returns:
    The logistic loss value as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """

    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)\
            or not np.issubdtype(y.dtype, np.number) or not np.issubdtype(y_hat.dtype, np.number)\
            or y.shape != y_hat.shape or y.size == 0:
        return None

    return float((-1 / y.shape[0]) * (y.T @ np.log(y_hat + eps) + (1 - y).T @ np.log(1 - y_hat + eps)))

if __name__ == '__main__':

    # Example 1:
    print('\n******************** Example 1 ********************\n')
    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    y_hat1 = logistic_predict_(x1, theta1)
    print(vec_log_loss_(y1, y_hat1))
    # Output:
    # 0.018149927917808714

    # Example 2:
    print('\n******************** Example 2 ********************\n')
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    y_hat2 = logistic_predict_(x2, theta2)
    print(vec_log_loss_(y2, y_hat2))
    # Output:
    # 2.4825011602472347

    # Example 3:
    print('\n******************** Example 3 ********************\n')
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    y_hat3 = logistic_predict_(x3, theta3)
    print(vec_log_loss_(y3, y_hat3),'\n')
    # Output:
    # 2.993853310859968