import numpy as np

class Color:
    BLUE = '\x1b[96m'
    WARNING = '\x1b[93m'
    GREEN = '\x1b[92m'
    END = '\x1b[0m'

class MyLogisticRegression:
	"""
	Description:
		My personnal logistic regression to classify things.
	"""

	supported_penalities = ['l2']

	def __init__(self, thetas, alpha = 0.001, max_iter = 1000, penalty = 'l2', lambda_ = 1.0):
		
		self.thetas = thetas
		self.alpha = alpha
		self.max_iter = max_iter
		self.penalty = penalty
		self.lambda_ = lambda_ if penalty in self.supported_penalities else 0.0
		
	def DataChecker(func):

		def wrapper(self, *args, **kwargs):
			for item in args:
				if not isinstance(item, np.ndarray)\
					or not np.issubdtype(item.dtype, np.number):
					raise('{item} is not a array or not a number')		
			res = func(self, *args, **kwargs)
			return res
		return wrapper

	def get_params(self):

		return vars(self)

	def set_params(self, **params):

		for key, value in params.items():
			if key in vars(self).keys():
				setattr(self, key, value)
		return self
	
	@DataChecker
	def sigmoid_(self, x):

		return 1 / (1 + np.exp(-x))

	@DataChecker
	def gradient_(self, x, y):

		theta_with_0 = self.thetas.copy()
		theta_with_0[0][0] = 0
		ones = np.ones(shape=(x.shape[0], 1))
		x = np.hstack((ones, x))
		y_hat = self.predict__(x)
		return (x.T.dot(y_hat - y) + self.lambda_ * theta_with_0) / y.shape[0]

	@DataChecker
	def gradient__(self, x, y):

		theta_with_0 = self.thetas.copy()
		theta_with_0[0][0] = 0
		y_hat = self.predict__(x)
		return (x.T.dot(y_hat - y) + self.lambda_ * theta_with_0) / y.shape[0]

	@DataChecker
	def gradient_unregularized__(self, x, y):

		y_hat = self.predict__(x)
		return x.T.dot(y_hat - y) / y.shape[0]

	@DataChecker
	def fit_(self, x, y):

		X = np.insert(x, 0, 1, axis=1)
		if self.penalty == 'l2':
			for _ in range(self.max_iter):
				self.thetas = self.thetas - (self.alpha * self.gradient__(X, y))
		else:
			for _ in range(self.max_iter):
				self.thetas -= (self.alpha * self.gradient_unregularized__(X, y))
		return self.thetas

	@DataChecker
	def loss_elem_(self, y, y_hat, eps = 1e-15):

		return y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)

	@DataChecker
	def loss_(self, y, y_hat, eps = 1e-15):
		
		loss_elem = self.loss_elem_(y, y_hat, eps)
		
		return None if loss_elem is None else -np.sum(loss_elem) / y.shape[0]

	@DataChecker
	def predict__(self, x):

		return self.sigmoid_(x @ self.thetas)

	@DataChecker
	def predict_(self, x):
		
		x = x.reshape(x.shape[0], 1) if len(list(x.shape)) < 2 else x
		X = np.insert(x, 0, 1, axis=1)
		
		theta = self.thetas.reshape(-1, 1) if len(self.thetas.shape) < 2 else self.thetas
		
		return None if X.shape[1] != theta.shape[0] else X @ theta

	def data_spliter(x, y, proportion):
		data = np.hstack((x, y))
		p = int(x.shape[0] * proportion)
		np.random.shuffle(data)

		x_train, y_train = data[:p, :-1], data[:p, -1:]
		x_test, y_test = data[p:, :-1], data[p:, -1:]

		return x_train, x_test, y_train, y_test

	def add_polynomial_features(x, power):
		return np.hstack([x ** i for i in range(1, power + 1)])

if __name__ == '__main__':
    
	from my_logistic_regression import MyLogisticRegression as mylogreg

	theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

	# Example 1:
	print('\n********************* Example 1 *********************\n')
	model1 = mylogreg(theta, lambda_=5.0)
	print('penalty ==> ', model1.penalty)
    # Output
    # 'l2'
	print('lambda_ ==> ', model1.lambda_)
    # Output
    # 5.0
    
	# Example 2:
	print('\n********************* Example 2 *********************\n')
	model2 = mylogreg(theta, penalty=None)
	print('penalty ==> ', model2.penalty)
    # Output
    # None
	print('lambda_ ==> ', model2.lambda_)
    # Output
    # 0.0
    
	# Example 3:
	print('\n********************* Example 3 *********************\n')
	model3 = mylogreg(theta, penalty=None, lambda_=2.0)
	print('penalty ==> ', model3.penalty)
    # Output
    # None
	print('lambda_ ==> ', model3.lambda_)
    # Output
    # 0.0