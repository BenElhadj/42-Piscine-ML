import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../ex04')
sys.path.insert(0, '../ex07')
from vec_loss import loss_  # type: ignore
from prediction import predict_  # type: ignore

def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    y_hat = predict_(x, theta)

    plt.scatter(x, y)
    plt.plot(x, y_hat, 'r')

    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y_hat[i], y[i]], 'r--')

    plt.title(f"Cost {loss_(y, y_hat)}")
    plt.show()


if __name__ == '__main__':
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')

    x = np.arange(1, 6)
    y = np.array([11.52434424, 10.62589482, 13.14755699,
                 18.60682298, 14.14329568]).reshape(-1, 1)
    # Example 1:
    theta1 = np.array([18, -1])
    plot_with_loss(x, y, theta1)
    # Output:

    # Example 2:
    theta2 = np.array([14, 0])
    plot_with_loss(x, y, theta2) 
    # Output:

    # Example 3:
    theta3 = np.array([12, 0.8])
    plot_with_loss(x, y, theta3)
