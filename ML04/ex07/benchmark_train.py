import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.insert(0, '../ex06')
from ridge import MyRidge # type:ignore

np.seterr(all='ignore', invalid='ignore')

class Color:
    BLUE = '\x1b[96m'
    WARNING = '\x1b[93m'
    GREEN = '\x1b[92m'
    END = '\x1b[0m'

def plot_evaluation_curve(models):
    xticks = [f'Pol {model.polynomial} λ = {model.lambda_:.1f}' for model in models]
    losses = [model.loss for model in models]
    plt.title('Evaluation metrics of models')
    plt.xticks(range(len(models)), xticks, rotation=270)
    plt.ylabel('Loss score')
    plt.xlabel('Polynomials + Lambda (λ) value')
    plt.plot(range(len(losses)), losses)
    plt.show()

def benchmark_train(x, y):
    models = []
    for i in range(1, 5):
        lambda_range = np.arange(0.0, 1.2, step=0.2)
        for lambda_ in lambda_range:
            cross_validation_loss = []
            for _ in range(4):
                model = MyRidge(thetas=np.random.rand(3*i+1, 1), alpha=0.000005, max_iter=1000, lambda_=lambda_)
                model.set_params(polynomial = i)
                x_ = MyRidge.add_polynomial_features(x, i) 
                mean, std = x_.mean(), x_.std()
                x_train, x_test, y_train, y_test = MyRidge.data_spliter(x_, y, 0.5)
                x_train = (x_train - mean) / std
                x_test = (x_test - mean) / std
                model.fit_(x_train, y_train)
                y_hat = model.predict_(x_test)
                loss = model.loss_(y_test, y_hat)
                cross_validation_loss.append(loss)
            avg_loss = sum(cross_validation_loss) / len(cross_validation_loss)
            print(f'Train Model {i} ==> {Color.BLUE}polynomial = {i}{Color.END}\t{Color.GREEN} λ {lambda_=:.1f}{Color.END}\
            Average loss ==> {Color.WARNING}{float(avg_loss)}{Color.END}\n')

            new_x = MyRidge.add_polynomial_features(x, i)
            mean, std = new_x.mean(), new_x.std()
            new_x = (new_x - mean) / std
            model.set_params(thetas=np.random.rand(3*i+1, 1))
            model.fit_(new_x, y)
            y_hat = model.predict_(new_x)
            model.loss = float(model.loss_(y, y_hat))
            models.append(model)
    with open('models.pickle', 'wb') as handle:
        pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return models

if __name__ == '__main__':
    
    data = pd.read_csv('./space_avocado.csv')
    x = np.array(data[['weight', 'prod_distance', 'time_delivery']])
    y = np.array(data[['target']])
    x_train, _, y_train, _ = MyRidge.data_spliter(x, y, 0.8)
    models = benchmark_train(x_train, y_train)
    plot_evaluation_curve(models)