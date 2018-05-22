from sklearn.model_selection import StratifiedKFold

class mySKF:
	def __init__(self, X, y, splits):
		"""用于交叉验证"""
		self.X = X
		self.y = y
		self.skf = StratifiedKFold(n_splits=splits)
		self.skf = self.skf.split(X, y)

	def __next__(self):
		train_index, test_index = self.skf.__next__()
		X_train, X_test = self.X[train_index], self.X[test_index]
		y_train, y_test = self.y[train_index], self.y[test_index]
		return X_train, y_train, X_test, y_test

	def __iter__(self):
		return self