import numpy as np
from tqdm import tqdm

np.seterr(all='ignore')

class MyLinearRegression:
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        if not isinstance(thetas, np.ndarray) or thetas.size == 0:
            raise('thetas is not a array or size == 0')
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def DataChecker(func):
        def wrapper(self, *args, **kwargs):
            for item in args:
                if not isinstance(item, np.ndarray)\
                    or not np.issubdtype(item.dtype, np.number):
                        raise('{item} is not a array or not a number')
            res = func(self, *args, **kwargs)
            return res
        return wrapper

    @DataChecker
    def fit_(self, x, y):

        X = np.insert(x, 0, 1, axis=1)
        cost = []
        for _ in tqdm(range(self.max_iter)):
            self.thetas = self.thetas - self.alpha\
                * (X.T @ ((X @ self.thetas) - y)) / y.shape[0]
            cost.append(self.loss_(y, self.predict_(x)))
        return self.thetas

    @DataChecker
    def predict_(self, x):

        x = x.reshape(x.shape[0], 1) if len(list(x.shape)) < 2 else x
        X = np.insert(x, 0, 1, axis=1)

        theta = self.thetas.reshape(-1, 1) if len(self.thetas.shape) < 2 else self.thetas

        if X.shape[1] != theta.shape[0] :
            return None

        return X @ theta

    @DataChecker
    def loss_elem_(self, y, y_hat):

        return (y - y_hat)**2

    @DataChecker
    def loss_(self, y, y_hat):

        return float((y_hat - y).T @ (y_hat - y) / (2 * y.shape[0]))

    @staticmethod
    def mse_(y, y_hat):

        return float(sum((y - y_hat)**2)/(y.shape[0]))

if __name__ == '__main__':

    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    Y = np.array([[23.], [48.], [218.]])
    mylr = MyLinearRegression(np.array([[1.], [1.], [1.], [1.], [1]]))

    # Example 0:
    y_hat = mylr.predict_(X)
    print(y_hat, '\n')
    # Output:
    # # array([[8.], [48.], [323.]])

    # Example 1:
    print(mylr.loss_elem_(Y, y_hat),'\n')
    # Output:
    # # array([[225.], [0.], [11025.]])

    # Example 2:
    print(mylr.loss_(Y, y_hat),'\n')
    # Output:
    # # 1875.0

    # Example 3:
    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    mylr.fit_(X, Y)
    print(mylr.thetas,'\n')
    # Output:
    # # array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])

    # Example 4:
    y_hat = mylr.predict_(X)
    print(y_hat, '\n')
    # Output:
    # # array([[23.417..], [47.489..], [218.065...]])

    # Example 5:
    print(mylr.loss_elem_(Y, y_hat),'\n')
    # Output:
    # # array([[0.174..], [0.260..], [0.004..]])

    # Example 6:
    print(mylr.loss_(Y, y_hat),'\n')
    # Output:
    # # 0.0732..