import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
	plt.grid(linestyle='-', linewidth=0.2)
	plt.show()

def plot_features(x_true, y_true):
	dim_axs: np.ndarray[plt.Axes]
	_, dim_axs = plt.subplots(ncols=len(FEATURES))
	axes = dim_axs.flatten()
	for idx, feature in enumerate(FEATURES):
		axes[idx].scatter(x_true[:, idx], y_true, label='True target prices')
		axes[idx].set_xlabel(feature)
		axes[idx].set_ylabel('Target price')
	plt.legend()
    
	return axes

def benchmark_test(x, y):
    with open('models.pickle', 'rb') as handle:
        models = pickle.load(handle)

    plot_evaluation_curve(models)

    best_model = min(models, key=lambda x: x.loss)
    print(f'Le modèle  {Color.GREEN}polynomial {best_model.polynomial}{Color.END} avec le {Color.GREEN}lambda_ {best_model.lambda_}{Color.END} est le meilleur{Color.GREEN} ✔ {Color.END}')
    _, x_test, _, y_test = MyRidge.data_spliter(x, y, 0.8)
    x_test = MyRidge.add_polynomial_features(x_test, best_model.polynomial)
    mean, std = x.mean(), x.std()
    x_test = (x_test - mean) / std
    y_hat = best_model.predict_(x_test)    
    loss_ = best_model.loss_(y_test, y_hat)
    best_model.set_params(loss=loss_)

    _, x_test, _, y_test = MyRidge.data_spliter(x, y, 0.8)
    best_models = [model for model in models if model.polynomial == best_model.polynomial]
    axes = plot_features(x_test, y_test)
    for idx, model in enumerate(best_models):
        x_ = MyRidge.add_polynomial_features(x_test, model.polynomial)
        x_norm = (x_ - x_.mean(axis=0)) / x_.std(axis=0)
        y_hat = model.predict_(x_norm)

        for index, feature in enumerate(FEATURES):
            axes[index].scatter(x_test[:, index], y_hat, alpha=0.3, label=f'Pol{model.polynomial}-λ{round(model.lambda_, 1)}')
        plt.legend()
    plt.show()

    axes = plot_features(x_test, y_test)
    submodels = [model for model in models if np.isclose(model.lambda_, 0.0)]
    for idx, model in enumerate(submodels):
        x_ = MyRidge.add_polynomial_features(x_test, model.polynomial)
        x_norm = (x_ - x_.mean(axis=0)) / x_.std(axis=0)
        y_hat = model.predict_(x_norm)

        for index, feature in enumerate(FEATURES):
            axes[index].scatter(x_test[:, index], y_hat, alpha=0.3, label=f'Pol{model.polynomial}-λ{model.lambda_}')
        plt.legend()
    plt.show()

if __name__ == '__main__':
    
    FEATURES = ['weight', 'prod_distance', 'time_delivery']
    data = pd.read_csv('./space_avocado.csv')
    x = np.array(data[['weight', 'prod_distance', 'time_delivery']])
    y = np.array(data[['target']])
    benchmark_test(x, y)

