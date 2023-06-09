import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '../ex05') 
from mylinearregression import MyLinearRegression as MyLR # type: ignore
np.seterr(all='ignore')

def myPlot(x, y, y_pred, loss, tabData):
    plt.figure(tabData[0])
    plt.scatter(x, y, label=tabData[1], marker='o', c= tabData[2])
    plt.scatter(x, y_pred, label=tabData[3], marker='.', c= tabData[4])
    plt.title(str(loss))
    plt.ylabel(tabData[5])
    plt.xlabel(tabData[6])
    plt.legend()
    plt.grid()

if __name__ == '__main__':
    data = pd.read_csv("./spacecraft_data.csv")
    X = np.array(data[['Age','Thrust_power','Terameters']])
    Y = np.array(data[['Sell_price']])

    tabData = np.array([['Age' , 'Sell Price', 'navy', 'Predected sell price', 'dodgerblue', 
            'y: sell price (in keuros)', '$X_1$: age (in years)', 'left-bottom', 681.0, 390.0],
                
            ['Thrust power' , 'Sell Price', 'green', 'Predected sell price', 'lime', 
            'y: sell price (in keuros)', '$X_2$: thrust power (in 10Km/s)', 'left-top', 25, 897],
                
            ['Terameters', 'Sell Price', 'blueviolet', 'Predected sell price', 'lightpink',
            'y: sell price (in keuros)','$X_3$: distance totalizer value of spacecraft (in Tmeters)',
            'right-top', 723, 211,]])

    for i in range(tabData.shape[0]):
        myLR = MyLR(thetas = np.array([[float(tabData[i][8])],
            [float(tabData[i][9])]]), alpha = 2.5e-5, max_iter = 100000)
        myLR.fit_(X[:, i].reshape(-1, 1), Y)
        y_pred = myLR.predict_(X[:,i].reshape(-1, 1))
        loss = myLR.mse_(y_pred, Y)
        myPlot(X[:, i], Y, y_pred, loss, tabData[i])
    plt.show()

    my_lreg = MyLR(thetas = np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1), alpha = 5e-5, max_iter = 600000)
    
    # Example 0:
    y_pred = my_lreg.predict_(X)
    print(my_lreg.mse_(y_pred, Y),'\n')
    # Output:
    # # 144044.877...

    # Example 1:
    my_lreg.fit_(X,Y)
    print(my_lreg.thetas,'\n')
    # Output:
    # # array([[334.994...],[-22.535...],[5.857...],[-2.586...]])

    # Example 2:
    y_pred = my_lreg.predict_(X)
    loss = my_lreg.mse_(y_pred,Y)
    print(loss,'\n')
    # Output:
    # # 586.896999...

    for i in range(tabData.shape[0]):
        myPlot(X[:, i], Y, y_pred, loss, tabData[i])
    plt.show()