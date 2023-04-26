import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error
sys.path.insert(0, '../ex03')
from my_linear_regression import MyLinearRegression as MyLR # type: ignore
import matplotlib.pyplot as plt
import numpy as np

class Linear_model():

    def LinearModel(x, y, y_hat):

        plt.scatter(x, y_hat, marker='X', label='S$_{predict}$(pills)', c='lime')
        plt.scatter(x, y, label='S$_{true}$(pills)', c='cyan')
        plt.plot(x, y_hat, c='lime', linestyle = '--')
        plt.xlabel("Quantitiy of blue pill (in micrograms)")
        plt.ylabel("Space driving score")
        plt.legend(bbox_to_anchor=(0.0, 1.15), frameon=False, ncol=2, loc='upper left')
        plt.grid()
        plt.show()

    def cost_theta(x, y):

        theta0 = np.linspace(78, 97, 6)
        theta1 = np.linspace(-14, -3, 700)
        for th0 in theta0:
            cost = []
            for th1 in theta1:
                model = MyLR(np.array([th0, th1]))
                y_pred = model.predict_(x)
                loss = model.loss_(y, y_pred)
                cost.append(loss)
            plt.plot(theta1, cost, label=r'J($\theta_0={},\theta_1$)'.format(th0))

        plt.legend(loc="lower right")
        plt.xlabel(r'$\theta_1$')
        plt.ylabel(r'cost function J($\theta_0, \theta_1$)')
        plt.ylim([10, 140])
        plt.grid()
        plt.show()


if __name__ == '__main__':
    data = pd.read_csv("./are_blue_pills_magics.csv")
    Xpill = np.array(data['Micrograms']).reshape(-1,1)
    Yscore = np.array(data['Score']).reshape(-1,1)

    linear_model1 = MyLR(np.array([[89.0], [-8]]))
    linear_model2 = MyLR(np.array([[89.0], [-6]]))
    Y_model1 = linear_model1.predict_(Xpill)
    Y_model2 = linear_model2.predict_(Xpill)
    
    x = np.arange(1, 7)

    Linear_model.LinearModel(Xpill, Yscore, Y_model1)
    
    Linear_model.cost_theta(Xpill, Yscore)

    print(MyLR.mse_(Yscore, Y_model1))
    # 57.60304285714282
    print(mean_squared_error(Yscore, Y_model1), '\n')
    # 57.603042857142825
    print(MyLR.mse_(Yscore, Y_model2))
    # 232.16344285714285
    print(mean_squared_error(Yscore, Y_model2), '\n')
    # 232.16344285714285

