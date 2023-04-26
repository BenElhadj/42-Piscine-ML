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

def plot_f1_scores(f1_scores):

	plt.title('F1 scores on the test set')
	plt.ylim(0.0, 1.0)
	plt.bar([f'{lambd:.1f}' for lambd in np.arange(0.0, 1.2, step=0.2)], f1_scores)
	plt.xlabel('Lambda values')
	plt.ylabel('F1 score')
	plt.show()

def score_models(models, x_test):

	pred = np.hstack([model.predict_(x_test) for model in models])
	return pred.argmax(axis=1).reshape(-1, 1)

def benchmark_test(models, x_test, y_test):

	range = np.arange(0.0, 1.2, step=0.2)
	f1_scores = []
	for lambda_, model in zip(range, models):
		y_hat = score_models(model, x_test)
		f1 = f1_score_(y_test, y_hat)
		accuracy = accuracy_score_(y_test, y_hat)
		print(f'Correctly predicted {Color.GREEN}{accuracy * 100:.1f}%{Color.END}, lambda_={Color.WARNING}{round(lambda_, 1)}{Color.END} f1_score = {Color.WARNING}{f1:.1f}{Color.END}')
		f1_scores.append(f1)

	return f1_scores

if __name__ == '__main__':

    data_x = pd.read_csv('./solar_system_census.csv')
    data_y = pd.read_csv('./solar_system_census_planets.csv')
    x = np.array(data_x[['weight', 'height', 'bone_density']])
    y = np.array(data_y[['Origin']])
    x = MyLogReg.add_polynomial_features(x, 3)
    mean, std = x.mean(), x.std()
    x = (x - mean) / std
    _, x_test, _, y_test = MyLogReg.data_spliter(x, y, proportion=0.8)

    with open('models.pickle', 'rb') as handle:
        models = pickle.load(handle)

    f1scores = benchmark_test(models, x_test, y_test)

    plot_f1_scores(f1scores)