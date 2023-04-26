import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.seterr(all='ignore')

class Color:
    BLUE = '\x1b[96m'
    WARNING = '\x1b[93m'
    GREEN = '\x1b[92m'
    END = '\x1b[0m'

class LogisticRegressionTraining(object):

    def __init__(self, alpha=0.001, max_iter=1000):  
        self.alpha = alpha                            
        self.max_iter = max_iter

    def dataSpliter_(x, y, proportion):
        data = np.hstack((x, y))
        p = int(x.shape[0] * proportion)
        np.random.shuffle(data)

        return data[:p, :-1], data[p:, :-1], data[:p, -1:], data[p:, -1:]

    def predict_(self, X):
        return np.array([max((self.sigmoid_(i @ theta), c)\
            for theta, c in self.theta)[1] for i in np.insert(X, 0, 1, axis=1) ]).reshape(-1, 1)

    def sigmoid_(self, x):
        return 1 / (1 + np.exp(-x))

    def loss_(self, h, y):
        return (1 / len(y)) * (np.sum(-y.T @ np.log(h) - (1 - y).T @ np.log(1 - h)))

    def gradient_descent_(self, X, h, theta, y, m): 
        return theta - self.alpha * ((X.T @ (h - y)) / m)

    def fit_(self, X, y):
        self.theta = []
        self.cost = []
        X = np.insert(X, 0, 1, axis=1)

        for i in np.unique(y):
            y_onevsall = np.where(y == i, 1, 0)
            theta = np.zeros(X.shape[1]).reshape(-1, 1)
            cost = []
            for j in range(self.max_iter):
                z = X @ theta
                h = self.sigmoid_(z)
                theta = self.gradient_descent_(X, h, theta, y_onevsall, len(y))
                cost.append(self.loss_(h, y_onevsall))
                print(f'Gradient descent: {Color.WARNING}\
                    {int((j / self.max_iter) * 100)}%{Color.END}\r', end="", flush=True)
            print(f"Trainning class: {Color.GREEN} {i:{' '}<2} ✔ {Color.END}")
            self.theta.append((theta, i))
            self.cost.append((cost,i))

        return self.theta

if __name__ == '__main__':

    from multi_log import LogisticRegressionTraining as LRT # type: ignore

    data_x = pd.read_csv('./solar_system_census.csv')
    data_y = pd.read_csv('./solar_system_census_planets.csv')
    x = np.array(data_x[['weight', 'height', 'bone_density']])
    y = np.array(data_y[['Origin']])
    x_train, x_test, y_train, y_test = LRT.dataSpliter_(x, y, proportion=0.8)
    features = ['weight', 'height', 'bone_density']

    MyLR = LRT(alpha=0.0001, max_iter=100000)
    MyLR.fit_(x, y)
    
    y_pred = MyLR.predict_(x_test)
    score_ = np.sum(y_pred == y_test)

    print(f'Correctly predicted values\t{score_} / {y_test.shape[0]} = \
        {Color.GREEN}{score_ / y_test.shape[0] * 100}%{Color.END}')

    for idx, feature in enumerate(features):
        plt.figure(idx + 1)
        plt.title(f'Zipcode predicted based on {feature}')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Zipcode')
        x_ = x_test[:, idx]
        plt.scatter(x_, y_test, label='Real values', s=150)
        plt.scatter(x_, y_pred, label='My prediction', s=30)
        plt.legend(loc='best')
    plt.show()