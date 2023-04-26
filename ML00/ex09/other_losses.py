import math
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


def mse_(y, y_hat):
    """
    Description:
    Calculate the MSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    mse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if (y.shape != y_hat.shape or y.size == 0):
        return None

    return sum((y_hat - y)**2)/y.shape[0]


def rmse_(y, y_hat):
    """
    Description:
    Calculate the RMSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    rmse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if (y.shape != y_hat.shape or y.size == 0):
        return None

    return math.sqrt(sum((y_hat - y)**2) / y.shape[0])


def mae_(y, y_hat):
    """
    Description:
    Calculate the MAE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    mae: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if (y.shape != y_hat.shape or y.size == 0):
        return None

    return sum(abs(y_hat - y)) / y.shape[0]


def r2score_(y, y_hat):
    """
    Description:
    Calculate the R2score between the predicted output and the output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    r2score: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if (y.shape != y_hat.shape or y.size == 1 or np.mean(y) == 0):
        return None
    
    sum_ = sum((y - np.mean(y))**2)
    
    if not sum_:
        return None
    
    return 1 - sum((y_hat - y)**2) / sum_


# if __name__ == '__main__':
#     import numpy as np
#     from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#     from math import sqrt
#     # Example 1:
#     x = np.array([0, 15, -9, 7, 12, 3, -21])
#     y = np.array([2, 14, -13, 5, 12, 4, -19])
#     # Mean squared error
#     ## your implementation
#     print(mse_(x,y), '\n')
#     ## Output:
#     ## 4.285714285714286
#     ## sklearn implementation
#     print(mean_squared_error(x,y), '\n')
#     ## Output:
#     ## 4.285714285714286
#     # Root mean squared error
#     ## your implementation
#     print(rmse_(x,y), '\n')
#     ## Output:
#     ## 2.0701966780270626
#     ## sklearn implementation not available: take the square root of MSE
#     print(sqrt(mean_squared_error(x,y)), '\n')
#     ## Output:
#     ## 2.0701966780270626
#     # Mean absolute error
#     ## your implementation
#     print(mae_(x,y), '\n')
#     # Output:
#     ## 1.7142857142857142
#     ## sklearn implementation
#     print(mean_absolute_error(x,y), '\n')
#     # Output:
#     ## 1.7142857142857142
#     # R2-score
#     ## your implementation
#     print(r2score_(x,y), '\n')
#     ## Output:
#     ## 0.9681721733858745
#     ## sklearn implementation
#     print(r2_score(x,y), '\n')
#     ## Output:
#     ## 0.9681721733858745
