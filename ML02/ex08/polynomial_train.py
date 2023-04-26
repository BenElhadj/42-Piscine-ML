import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../ex07')
from polynomial_model import add_polynomial_features as Pf# type: ignore
sys.path.insert(0, '../ex05')
from mylinearregression import MyLinearRegression as MyLR # type: ignore

def LRTrain():
    data = pd.read_csv("./are_blue_pills_magics.csv")
    X = np.array(data[['Micrograms']])
    Y = np.array(data[['Score']])

    thetas = [np.random.rand(1 + 1, 1),
        np.random.rand(1 + 2, 1), np.random.rand(1 + 3, 1),
        np.array([[-20],[ 160],[ -80],[ 10],[ -1]]).reshape(-1,1),
        np.array([[1140],[ -1850],[ 1110],[ -305],[ 40],[ -2]]).reshape(-1,1),
        np.array([[9110],[ -18015],[ 13400],[ -4935],[ 966],[ -96.4],[ 3.86]]).reshape(-1,1)]

    iter = [1000, 20000, 240000, 300000, 380000, 500000]
    alpha = [0.01, 2e-3, 7e-5, 1e-7, 5e-8, 1e-9]

    mseTab = []
    x_ = np.linspace(1, 7, 100).reshape(-1,1)
    
    for i in range(6):
        x = Pf(X, i+1)
        mylr = MyLR(thetas[i], alpha[i], iter[i])
        mylr.fit_(x, Y)
        mse =  mylr.mse_(Y, mylr.predict_(x))
        mseTab.append(mse)
        y_pred = mylr.predict_(Pf(x_, (i + 1)))
    
        print(f'\n\t({i+1})\tMSE {i+1} =\t{mse}\n\talpha = {alpha[i]}\titer = {iter[i]}\n')
        plt.figure("Train Polynomial Models")
        plt.plot(x_, y_pred)
        plt.plot(x_, y_pred, label=f'$pred_{i+1}\tmse_{i+1}:{round(mse, 3)}$')
    plt.scatter(X, Y, label="data points", c='black', s=60)
    plt.ylabel('X_')
    plt.xlabel('Y_pred')
    plt.legend()
    plt.grid(linestyle='-', linewidth=0.2)
    plt.show()
    
    plt.figure("MSE score based on polynomial hypothesis")
    plt.grid(linestyle='-', linewidth=0.2)
    plt.bar(range(1,7), mseTab)
    plt.xlabel('polynomial degree')
    plt.ylabel('MSE')
    plt.show()


if __name__ == '__main__':

    LRTrain()
