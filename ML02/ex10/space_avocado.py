import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
np.seterr(all='ignore', invalid='ignore')
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, '../ex09')
from data_spliter import data_spliter as DataSplit# type: ignore
sys.path.insert(0, '../ex07')
from polynomial_model import add_polynomial_features as Pf# type: ignore
sys.path.insert(0, '../ex05')
from mylinearregression import MyLinearRegression as MyLR # type: ignore

class Color:
    BLUE = '\x1b[96m'
    WARNING = '\x1b[93m'
    GREEN = '\x1b[92m'
    END = '\x1b[0m'

def benchmark_test(x, y):

    with open('models.pickle', 'rb') as handle:
        thetas = pickle.load(handle)
    x_scaler = StandardScaler()

    iter = [20000, 30000, 40000, 50000]
    alpha = [0.3, 0.2, 0.1, 0.01] 

    models = [MyLR(thetas[i], alpha=alpha[i], max_iter=iter[i]) for i in range(np.size(thetas[0]))]

    loss = []
    y_hat = []

    for index, model in enumerate(models):
        x_ = Pf(x, index + 1)
        x_ = x_scaler.fit_transform(x_)
        y_pred = model.predict_(x_)
        loss.append(model.mse_(y, y_pred))
        y_hat.append(y_pred)

    bestModel = loss.index(min(loss))
    y_pred = y_hat[bestModel]

    features = ['Weight', 'Prod_distance', 'Time_delivery']
    for index, feature in enumerate(features):
        plt.figure(index+1)
        plt.title(feature)
        plt.scatter(x[:, index], y, label='Target value')
        plt.scatter(x[:, index], y_pred, label='Predictions Target value')
        plt.legend()
    plt.show()


if __name__ == '__main__':
    
    data = pd.read_csv("./space_avocado.csv")
    
    x = np.array(data[['weight', 'prod_distance', 'time_delivery']])
    y = np.array(data[['target']])


    x_train, x_test, y_train, y_test = DataSplit(x, y, 0.8)
    benchmark_test(x_train, y_train)