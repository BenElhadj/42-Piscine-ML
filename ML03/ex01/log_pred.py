import numpy as np
# import math

def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * n.
    theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """

    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)\
        or not np.issubdtype(x.dtype, np.number) or not np.issubdtype(theta.dtype, np.number)\
        or x.size == 0 or theta.size == 0\
        or (len(x.shape) < 2 or len(theta.shape) < 2\
        or (x.shape[1] + 1 != theta.shape[0])):
        return None
    
    X = np.insert(x, 0, 1, axis=1)
    return 1 / (1 + np.exp(-X @ theta))
    #     z = -X @ theta
    # return [[1 / (1 + math.exp(item))] for item in z]

if __name__ == '__main__':

    # Example 1
    print('\n******************** Example 1 ********************\n')
    x = np.array([4]).reshape((-1, 1))
    theta = np.array([[2], [0.5]])
    print(logistic_predict_(x, theta))
    # Output:
    # array([[0.98201379]])
    
    # Example 2
    print('\n******************** Example 2 ********************\n')
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(logistic_predict_(x2, theta2))
    # Output:
    # array([[0.98201379],
    # [0.99624161],
    # [0.97340301],
    # [0.99875204],
    # [0.90720705]])
    
    # Example 3
    print('\n******************** Example 3 ********************\n')
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(logistic_predict_(x3, theta3), '\n')
    # Output:
    # array([[0.03916572],
    # [0.00045262],
    # [0.2890505 ]]