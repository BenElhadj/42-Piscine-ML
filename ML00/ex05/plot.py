import matplotlib.pyplot as plt
import numpy as np


def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    y: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exceptions.
    """
    x = x.reshape(-1, 1)
    X = np.insert(x, 0, 1, axis=1)

    theta = theta.reshape(-1, 1)

    plt.figure("42")
    plt.scatter(x, y)
    plt.plot(x, X @ theta, c='r')
    plt.show()


# if __name__ == '__main__':
#     import numpy as np
#     x = np.arange(1, 6)
#     y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
#     # y = np.arange(1, 13, 4)
#     # Example 1:
#     theta1 = np.array([[4.5], [-0.2]])
#     plot(x, y, theta1)
#     # Output:

#     # Example 2:
#     theta2 = np.array([[-1.5], [2]])
#     plot(x, y, theta2)
#     # Output:

#     # Example 3:
#     theta3 = np.array([[3], [0.3]])
#     plot(x, y, theta3)
#     # Output:
