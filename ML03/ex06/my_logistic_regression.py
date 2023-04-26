import numpy as np

class Color:
    BLUE = '\x1b[96m'
    WARNING = '\x1b[93m'
    GREEN = '\x1b[92m'
    END = '\x1b[0m'

class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta

    def predict_(self, x):  
        return 1 / (1 + np.exp(-(np.insert(x, 0, 1, axis=1)) @ self.theta))

    def loss_elem_(self, y, y_hat, eps = 1e-15):
        return y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)

    def loss_(self, y, y_hat, eps=1e-15): 
        return -np.sum(self.loss_elem_(y, y_hat)) / y.shape[0]

    def fit_(self, x, y): 
        for j in range(self.max_iter):
            print(f'Gradient descent: {Color.WARNING}\
                {int((j / self.max_iter) * 100)}%{Color.END}\r', end="", flush=True)
            self.theta -= self.alpha * self.gradaint_(x, y)

        return self.theta
    
    def gradaint_(self, x, y):
        X = np.insert(x, 0, 1, axis=1)
        return X.T @ (self.sigmoid_(X @ self.theta) - y) / x.shape[0] 

    def sigmoid_(self, x):
        return np.array(1 / (1 + np.exp(-x)))

if __name__ == '__main__':

    from my_logistic_regression import MyLogisticRegression as MyLR
    
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
    mylr = MyLR(thetas)
    
    # Example 0:
    print('\n******************** Example 0 ********************\n')
    print(mylr.predict_(X))
    # Output:
    # array([[0.99930437],
    # [1. ],
    # [1. ]])
    
    # Example 1:
    print('\n******************** Example 1 ********************\n')
    print(mylr.loss_(Y, mylr.predict_(X)))
    # Output:
    # 11.513157421577004
    
    # Example 2:
    print('\n******************** Example 2 ********************\n')
    mylr.fit_(X, Y)
    print(mylr.theta)
    # Output:
    # array([[ 2.11826435]
    # [ 0.10154334]
    # [ 6.43942899]
    # [-5.10817488]
    # [ 0.6212541 ]])
    
    # Example 3:
    print('\n******************** Example 3 ********************\n')
    print(mylr.predict_(X))
    # Output:
    # array([[0.57606717]
    # [0.68599807]
    # [0.06562156]])
    
    # Example 4:
    print('\n******************** Example 4 ********************\n')
    print(mylr.loss_(Y, mylr.predict_(X)), '\n')
    # Output:
    # 1.4779126923052268