import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '../ex06')
from my_logistic_regression import MyLogisticRegression as MLR# type: ignore
np.seterr(all='ignore')

class Color:
    BLUE = '\x1b[96m'
    WARNING = '\x1b[93m'
    GREEN = '\x1b[92m'
    END = '\x1b[0m'

ZIPCODE_FAVORITE = 1

def data_spliter(x, y, proportion):
    data = np.hstack((x, y))
    p = int(x.shape[0] * proportion)
    np.random.shuffle(data)
    x_train, y_train = data[:p, :-1], data[:p, -1:]
    x_test, y_test = data[p:, :-1], data[p:, -1:]

    return x_train, x_test, y_train, y_test

if __name__ == '__main__':

    zip = {0:'Venus', 1: 'Earth', 2: 'Mars Republic', 3: 'The Asteroids'}
    data_x = pd.read_csv('./solar_system_census.csv')
    data_y = pd.read_csv('./solar_system_census_planets.csv')
    x = np.array(data_x[['weight', 'height', 'bone_density']])
    y = np.array(data_y[['Origin']])
    y = np.where(y != ZIPCODE_FAVORITE, 0, 1)
    MyLR = MLR(theta = np.ones((3 + 1, 1)), alpha=0.0001, max_iter=100000)
    x_train, x_test, y_train, y_test = data_spliter(x, y, proportion = 0.8)
    MyLR.fit_(x, y)
    y_pred = MyLR.predict_(x_test)
    score_ = np.sum(np.round(y_pred) == y_test)
    
    print(f"Trainning hypothesis Zipcode{Color.GREEN} {zip[ZIPCODE_FAVORITE]:{' '}<2} \
        ✔ {Color.END}")
    print(f'Correctly predicted values {score_} / {y_test.shape[0]} = {Color.GREEN}\
        {score_ / y_test.shape[0] * 100}%{Color.END}')
    
    features = ['weight', 'height', 'bone_density']
    for idx, feature in enumerate(features):
        plt.figure(idx + 1)
        plt.title(f'Zipcode predicted based on {feature}')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Zipcode')
        x_ = x_test[:, idx]
        plt.scatter(x_, y_test, label='Real values', s=150)
        plt.scatter(x_, y_pred, label='My prediction', s=30)
        plt.scatter(x_, np.round(y_pred), label='My rounded prediction')
        plt.legend(loc='best')
    plt.show()