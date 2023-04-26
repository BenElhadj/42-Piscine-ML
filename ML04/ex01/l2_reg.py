import numpy as np

def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
    Args:
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
    The L2 regularization as a float.
    None if theta in an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """

    if not isinstance(theta, np.ndarray) or theta.size == 0:
        return None
    return sum(float(theta[j] * theta[j]) for j in range(1, len(theta)))

def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
    Args:
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
    The L2 regularization as a float.
    None if theta in an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    
    if not isinstance(theta, np.ndarray) or theta.size == 0:
        return None
    return float(np.sum(theta[1:].T @ theta[1:]))

if __name__ == '__main__':

    x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    
    # Example 1:
    print('\n********************* Example 1 *** iterative_l2 ***')
    print(iterative_l2(x))
    # Output:
    # 911.0
    
    # Example 2:
    print('\n--------------------- Example 2 -------- l2 --------')
    print(l2(x))
    # Output:
    # 911.0

    y = np.array([3,0.5,-6]).reshape((-1, 1))
    # Example 3:
    print('\n********************* Example 3 *** iterative_l2 ***')
    print(iterative_l2(y))
    # Output:
    # 36.25

    # Example 4:
    print('\n--------------------- Example 4 -------- l2 --------')
    print(l2(y), '\n')
    # Output:
    # 36.25