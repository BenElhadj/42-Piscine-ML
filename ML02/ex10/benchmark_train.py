import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
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

def plot_mse(loss):
    plt.figure(" ")
    plt.title("Trainnng Cost")
    plt.plot(range(1, len(loss)+1), loss)
    plt.xlabel("Polynomial Hypothesis")
    plt.ylabel("MSE Score")
    plt.show()

def benchmark_train(x, y):
    

    iter = [20000, 30000, 40000, 50000]
    alpha = [0.3, 0.2, 0.1, 0.01]    

    models = [MyLR(thetas = np.random.rand((i+1)*3+1, 1), alpha=alpha[i], max_iter=iter[i]) for i in range(4)]
    loss = []
    for index, model in enumerate(models):
        x_scaler = StandardScaler()
        x_ = Pf(x, index + 1)
        x_ = x_scaler.fit_transform(x_)
        model.fit_(x_, y)
        loss.append(model.mse_(y, model.predict_(x_)))

        print(f"Trainning: {Color.GREEN} {f'model_number_{index + 1}':{' '}<15} ✔ {Color.END}")

    thetas = [model.thetas for model in models]
    with open('models.pickle', 'wb') as handle:
        pickle.dump(thetas, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return loss

if __name__ == '__main__':
    
    data = pd.read_csv("./space_avocado.csv")
    
    x = np.array(data[['weight', 'prod_distance', 'time_delivery']])
    y = np.array(data[['target']])

    x_train, x_test, y_train, y_test = DataSplit(x, y, 0.8)
    loss = benchmark_train(x_train, y_train)

    plot_mse(loss)