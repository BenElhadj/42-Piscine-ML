import numpy as np
from tqdm import tqdm

np.seterr(all='ignore')

class Color:
    BLUE = '\x1b[96m'
    WARNING = '\x1b[93m'
    GREEN = '\x1b[92m'
    END = '\x1b[0m'

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
            self.thetas -= self.alpha * (X.T @ ((X @ self.thetas) - y)) / y.shape[0]
            cost.append(self.loss_(y, self.predict_(x)))
        
        return self.thetas

    @DataChecker
    def predict_(self, x):

        x = x.reshape(x.shape[0], 1) if len(list(x.shape)) < 2 else x
        X = np.insert(x, 0, 1, axis=1)

        theta = self.thetas.reshape(-1, 1) if len(self.thetas.shape) < 2 else self.thetas

        return None if X.shape[1] != theta.shape[0] else X @ theta

    @DataChecker
    def loss_elem_(self, y, y_hat):

        return (y - y_hat)**2

    @DataChecker
    def loss_(self, y, y_hat):

        return float((y_hat - y).T @ (y_hat - y) / (2 * y.shape[0]))
    
    @staticmethod
    def mse_(y, y_hat):

        return float(sum((y - y_hat)**2)/(y.shape[0]))    
        
    def cost_(self, x, y):
        
        return 1/(2*x.shape[0]) * np.sum((self.predict_(x) - y)**2)

class MyRidge(MyLinearRegression):
    """
    Description: My personal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        super().__init__(thetas, alpha, max_iter)
        self.lambda_ = lambda_
        self.polynomial = 1

    def get_params(self):
        
        return vars(self)

    def set_params(self, **params):
        for key, value in params.items():
            if key in vars(self):
                setattr(self, key, value)

        return self

    def l2(self):
        theta_with_0 = self.thetas.copy()
        theta_with_0[0][0] = 0
        
        return theta_with_0.T @ theta_with_0

    def loss_(self, y, y_hat):
        
        return (((y_hat - y).T @ (y_hat - y)) + self.lambda_ * self.l2()) / (2 * y.shape[0])

    def loss_elem_(self, y, y_hat):
        
        return (y_hat - y)**2 + self.lambda_ * self.l2()

    def predict_(self, x):

        x = x.reshape(x.shape[0], 1) if len(list(x.shape)) < 2 else x
        X = np.insert(x, 0, 1, axis=1)

        theta = self.thetas.reshape(-1, 1) if len(self.thetas.shape) < 2 else self.thetas

        return None if X.shape[1] != theta.shape[0] else X @ theta

    def gradient_(self, x, y):
        theta_with_0 = self.thetas.copy()
        theta_with_0[0][0] = 0
        y_hat = x @ self.thetas
        
        return (x.T @ (y_hat - y)) + (self.lambda_ * theta_with_0) / y.shape[0]

    def fit_(self, x, y):
        X = np.insert(x, 0, 1, axis=1)
        for j in range(self.max_iter):
            print(f'Gradient descent: {Color.WARNING}{int((j / self.max_iter) * 100)}%{Color.END}\r', end="", flush=True)
            self.thetas = self.thetas - (self.alpha * self.gradient_(X, y))
        
        return self.thetas

    def mse_(self, y, y_hat):
        
        return (((y_hat - y).T @ (y_hat - y)) + self.lambda_ * self.l2()) / y.shape[0]

    def data_spliter(self, y, proportion):
        data = np.hstack((self, y))
        p = int(self.shape[0] * proportion)
        np.random.shuffle(data)

        x_train, y_train = data[:p, :-1], data[:p, -1:]
        x_test, y_test = data[p:, :-1], data[p:, -1:]

        return x_train, x_test, y_train, y_test

    def add_polynomial_features(self, power):
        
        return np.hstack([self**i for i in range(1, power + 1)])

if __name__ == '__main__':

    pass