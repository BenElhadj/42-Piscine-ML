import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.insert(0, '../ex08')
from my_logistic_regression import MyLogisticRegression as MyLogReg # type:ignore
np.seterr(all='ignore', invalid='ignore')

class Color:
    BLUE = '\x1b[96m'
    WARNING = '\x1b[93m'
    GREEN = '\x1b[92m'
    END = '\x1b[0m'

def accuracy_score_(y, y_hat):

    return np.sum(y == y_hat) / y.shape[0]

def f1_score_(y, y_hat, pos_label = 1):

    tp = np.sum((y == pos_label) & (y_hat == pos_label))
    fp = np.sum((y != pos_label) & (y_hat == pos_label))
    fn = np.sum((y == pos_label) & (y_hat != pos_label))
    
    return (2 * (tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn)))

def score_models(models, x_test, y_test):

    pred = np.hstack([model.predict_(x_test) for model in models])
    y_hat = pred.argmax(axis=1).reshape(-1, 1)
    f1 = f1_score_(y_test, y_hat)
    accuracy = accuracy_score_(y_test, y_hat)
    print(f'Correctly predicted {Color.WARNING}{accuracy * 100:.1f}%{Color.END}, f1_score = {Color.WARNING}{f1:.2f}{Color.END}')
    
    return f1

def train_one_vs_all(x_train, y_train, lambda_):
    
    models = []
    for zipcode in range(4):
        model = MyLogReg(thetas=np.ones((3 * 3 + 1, 1)), alpha=1e-2, max_iter=10000, lambda_=lambda_)
        y_train_zipcode = np.where(y_train == zipcode, 1, 0)
        model.fit_(x_train, y_train_zipcode)
        models.append(model)
    
    return models

def benchmark_train(x, y):
    
    lambdas = np.arange(0.0, 1.2, step=0.2)
    models, f1scores = [], []
    for lambda_ in lambdas:
        print(f'Lambda = {Color.BLUE}{lambda_:.1f}{Color.END}')
        valid_cross = []
        for i in range(5):
            print(f'Cross validation {Color.GREEN}set {i} ✔{Color.END} ', end='')
            x_train, x_test, y_train, y_test = MyLogReg.data_spliter(x, y, 0.4)
            models_ = train_one_vs_all(x_train, y_train, lambda_)
            f1 = score_models(models_, x_test, y_test)
        valid_cross.append(f1)
        average_f1_score = sum(valid_cross) / len(valid_cross)
        print(f'Average cross validation score with lambda_= {Color.WARNING}{lambda_:.1f}{Color.END} is {Color.WARNING}{average_f1_score:.2f}{Color.END}')
        f1scores.append(average_f1_score)
        models_ = train_one_vs_all(x, y, lambda_)
        models.append(models_)
    with open('models.pickle', 'wb') as handle:
        pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return f1scores

def plot_f1_scores(f1_scores):

	plt.title('F1 scores')
	plt.ylim(0.0, 1.0)
	plt.bar([f'{lambd:.1f}' for lambd in np.arange(0.0, 1.2, step=0.2)], f1_scores)
	plt.xlabel('Lambda values')
	plt.ylabel('F1 score')
	plt.show()


if __name__ == '__main__':
    
    data_x = pd.read_csv('./solar_system_census.csv')
    data_y = pd.read_csv('./solar_system_census_planets.csv')
    x = np.array(data_x[['weight', 'height', 'bone_density']])
    y = np.array(data_y[['Origin']])
    x = MyLogReg.add_polynomial_features(x, 3)
    
    mean, std = x.mean(), x.std()
    x = (x - mean) / std
    x_train, _, y_train, _ = MyLogReg.data_spliter(x, y, proportion=0.8)
    f1scores = benchmark_train(x_train, y_train)

    plot_f1_scores(f1scores)