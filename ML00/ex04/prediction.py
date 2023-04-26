import numpy as np


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.array, a vector of dimension m * 1.
    None if x and/or theta are not numpy.array.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exceptions.
    """
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)\
            or x.size == 0 or theta.size == 0:
        return None

    x = x.reshape(-1, 1) if len(x.shape) < 2 else x
    x = np.insert(x, 0, 1, axis=1)

    theta = theta.reshape(-1, 1) if len(theta.shape) < 2 else theta

    if x.shape[1] != theta.shape[0] and theta.shape != (2, 1):
        return None

    return x @ theta


# if __name__ == '__main__':
#     import numpy as np
#     x = np.arange(1,6)
#     # Example 1:
#     theta1 = np.array([[5], [0]])
#     print(predict_(x, theta1), '\n')
#     # Ouput:
#     # array([[5.], [5.], [5.], [5.], [5.]])
#     # Do you remember why y_hat contains only 5’s here?
#     # Example 2:
#     theta2 = np.array([[0], [1]])
#     print(predict_(x, theta2), '\n')
#     # Output:
#     # array([[1.], [2.], [3.], [4.], [5.]])
#     # Do you remember why y_hat == x here?
#     # Example 3:
#     theta3 = np.array([[5], [3]])
#     print(predict_(x, theta3), '\n')
#     # Output:
#     # array([[ 8.], [11.], [14.], [17.], [20.]])
#     # Example 4:
#     theta4 = np.array([[-3], [1]])
#     print(predict_(x, theta4), '\n')
#     # Output:
#     # array([[-2.], [-1.], [ 0.], [ 1.], [ 2.]])