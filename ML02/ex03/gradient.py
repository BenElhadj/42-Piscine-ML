import numpy as np

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have the compatible dimensions.
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    y: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
    The gradient as a numpy.array, a vector of dimensions n * 1,
    containg the result of the formula for all j.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible dimensions.
    None if x, y or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """

    if not np.issubdtype(x.dtype, np.number) \
            or not np.issubdtype(y.dtype, np.number) \
            or not np.issubdtype(theta.dtype, np.number) \
            or not isinstance(x, np.ndarray) or x.size == 0\
            or not isinstance(y, np.ndarray) or y.size == 0\
            or not isinstance(theta, np.ndarray) or theta.size == 0\
            or x.shape[0] != y.shape[0] or theta.shape[0] != x.shape[1]:
        return None

    X = np.insert(x, 0, 1, axis=1)
    theta = np.insert(theta, 0, 0, axis=0)

    return (X.T @ ((X @ theta) - y)) / y.shape[0]

if __name__ == '__main__':

    x = np.array([
    [ -6, -7, -9],
    [ 13, -2, 14],
    [ -7, 14, -1],
    [ -8, -4, 6],
    [ -5, -9, 6],
    [ 1, -5, 11],
    [ 9, -11, 8]])
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    theta1 = np.array([3,0.5,-6]).reshape((-1, 1))
    # Example :
    print(gradient(x, y, theta1), '\n')
    # Output:
    # # array([[ -33.71428571], [ -37.35714286], [183.14285714], [-393.]])
    # Example :
    theta2 = np.array([0,0,0]).reshape((-1, 1))
    print(gradient(x, y, theta2), '\n')
    # Output:
    # # array([[ -0.71428571], [ 0.85714286], [23.28571429], [-26.42857143]])