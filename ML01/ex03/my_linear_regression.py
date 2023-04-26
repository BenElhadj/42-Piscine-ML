import numpy as np

class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        if not isinstance(thetas, np.ndarray) or thetas.size == 0:
            return None
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def DataChecker(func):
        def wrapper(self, *args, **kwargs):
            for item in args:
                if not isinstance(item, np.ndarray)\
                    or not np.issubdtype(item.dtype, np.number):
                    return None
            res = func(self, *args, **kwargs)
            return res
        return wrapper

    @DataChecker
    def fit_(self, x, y):

        X = np.insert(x, 0, 1, axis=1)
        for _ in range(self.max_iter):
            self.thetas = self.thetas - self.alpha\
                * (X.T @ ((X @ self.thetas) - y)) / y.shape[0]

        return self.thetas

    @DataChecker
    def predict_(self, x):

        x = x.reshape(-1, 1) if len(x.shape) < 2 else x
        X = np.insert(x, 0, 1, axis=1)

        theta = self.thetas.reshape(-1, 1) if len(self.thetas.shape) < 2 else self.thetas

        if X.shape[1] != theta.shape[0] and theta.shape != (2, 1):
            return None

        return X @ theta

    @DataChecker
    def loss_elem_(self, y, y_hat):

        return (y - y_hat)**2

    @DataChecker
    def loss_(self, y, y_hat):

        return float(sum((y - y_hat)**2)/(2 * y.shape[0]))

    @staticmethod
    def mse_(y, y_hat):

        return float(sum((y - y_hat)**2)/(y.shape[0]))

if __name__ == '__main__':
    import math
    import numpy as np
    from my_linear_regression import MyLinearRegression as MyLR
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    lr1 = MyLR(np.array([[2], [0.7]]))

    # Example 0.0:
    y_hat = lr1.predict_(x)
    print(y_hat, '\n')
    # Output:
    ## array([[10.74695094],
    ## [17.05055804],
    ## [24.08691674],
    ## [36.24020866],
    ## [42.25621131]])

    # Example 0.1:
    print(lr1.loss_elem_(y, y_hat), '\n')
    # Output:
    ## array([[710.45867381],
    ## [364.68645485],
    ## [469.96221651],
    ## [108.97553412],
    ## [299.37111101]])

    # Example 0.2:
    print(lr1.loss_(y, y_hat), '\n')
    # Output:
    ## 195.34539903032385

    # Example 1.0:
    lr2 = MyLR(np.array([[1], [1]]), 5e-8, 1500000)
    lr2.fit_(x, y)
    print(lr2.thetas, '\n')
    # Output:
    #array([[1.40709365],
    #[1.1150909 ]])

    # Example 1.1:
    y_hat = lr2.predict_(x)
    print(y_hat, '\n')
    # Output:
    ## array([[15.3408728 ],
    ## [25.38243697],
    ## [36.59126492],
    ## [55.95130097],
    ## [65.53471499]])

    # Example 1.2:
    print(lr2.loss_elem_(y, y_hat), '\n')
    # Output:
    ## array([[486.66604863],
    ## [115.88278416],
    ## [ 84.16711596],
    ## [ 85.96919719],
    ## [ 35.71448348]])

    # Example 1.3:
    print(lr2.loss_(y, y_hat), '\n')
    # Output:
    ## 80.83996294128525
