import numpy as np
import pickle
import os
from utils import softmax, cross_entropy_loss, one_hot

class LogisticregressionClassifier:

	def __init__(self):
		self.weights = None

	def fit(self, X, y, epochs=50, alpha=0.001):
		np.random.seed(4)
		self.weights = np.random.randn(4, X.shape[1])

		for epoch in range(epochs):
			loss = 0
			for x, label in zip(X, y):
				y_hat = self.predict(x.transpose().reshape(-1, 1))
				loss += cross_entropy_loss(y_hat, label[0])
				derivative_cross_entropy_softmax = y_hat-one_hot(label[0], 4)
				gradients = np.dot(derivative_cross_entropy_softmax, x.reshape(1, -1))
				self.weights = self.weights - alpha * gradients
			print(f"loss in epoch {epoch}: {loss/X.shape[0]}")


	def predict(self, x):
		return softmax(np.dot(self.weights, x.transpose().reshape(-1, 1)))

	def save_model(self, file_name: str) -> None:
		with open(file_name, 'wb') as f:
			pickle.dump(self, f)

	@staticmethod
	def load_model(file_name: str) -> 'Model':
		if not os.path.exists(file_name):
			raise ValueError()
		with open(file_name, 'rb') as f:
			return pickle.load(f)