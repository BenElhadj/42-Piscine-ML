import numpy as np

def reg_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop.
    Args:
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    The regularized loss as a float.
    None if y, y_hat, or theta are empty numpy.ndarray.
    None if y and y_hat do not share the same shapes.
    Raises:
    This function should not raise any Exception.
    """
    
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)\
        or not isinstance(theta, np.ndarray)\
        or y.shape != y_hat.shape or y.size == 0 or theta.size == 0:
        return None
    
    return float((((y_hat - y).T @ (y_hat - y)) +\
        lambda_ * float(np.sum(theta[1:] ** 2))) / (2 * y.shape[0]))

if __name__ == '__main__':

    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
    
    # Example 1:
    print('\n********************* Example 1 *********************')
    print(reg_loss_(y, y_hat, theta, .5))
    # Output:
    # 0.8503571428571429
    
    # Example 2:
    print('\n********************* Example 2 *********************')
    print(reg_loss_(y, y_hat, theta, .05))
    # Output:
    # 0.5511071428571429
    
    # Example 3:
    print('\n********************* Example 3 *********************')
    print(reg_loss_(y, y_hat, theta, .9), '\n')
    # Output:
    # 1.116357142857143