import numpy as np


def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.array x.
    Args:
    x: has to be a numpy.array of dimension m * n.
    Returns:
    X, a numpy.array of dimension m * (n + 1).
    None if x is not a numpy.array.
    None if x is an empty numpy.array.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None

    if len(x.shape) < 2:
        x = x.reshape(-1, 1)
        X = np.insert(x, 0, 1, axis=1)

    X = np.insert(x, 0, 1, axis=1)
    return X


# if __name__ == '__main__':
#     import numpy as np
#     # Example 1:
#     x = np.arange(1, 6)
#     print(add_intercept(x), '\n')
#     # Output:
#     # array([[1., 1.],
#     # [1., 2.],
#     # [1., 3.],
#     # [1., 4.],
#     # [1., 5.]])
#     # Example 2:
#     y = np.arange(1, 10).reshape((3, 3))
#     print(add_intercept(y), '\n')
#     # Output:
#     # array([[1., 1., 2., 3.],
#     # [1., 4., 5., 6.],
#     # [1., 7., 8., 9.]])
