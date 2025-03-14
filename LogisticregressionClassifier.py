import numpy as np
import pickle
import os
from math_utils import softmax, cross_entropy_loss, one_hot

class LogisticregressionClassifier:

	def __init__(self):
		self.weights = None

	def evaluate(self, X_val, y_val):
		correct_count = 0
		wrong_count = 0
		for x, y in zip(X_val, y_val):
			y_hat = self.predict(x)
			prediction = np.argmax(y_hat)
			if prediction == y:
				correct_count += 1
			else:
				wrong_count += 1
		return correct_count, wrong_count

	def fit(self, X_train, X_val, y_train, y_val, epochs=50, alpha=0.001):

		print(f'X_train shape: {X_train.shape}')
		print(f'X_val shape: {X_val.shape}')
		print(f'y_train shape: {y_train.shape}')
		print(f'y_val shape: {y_val.shape}')

		np.random.seed(41)
		self.weights = np.random.randn(4, X_train.shape[1])

		for epoch in range(epochs):
			loss = 0
			for x, label in zip(X_train, y_train):
				y_hat = self.predict(x.transpose().reshape(-1, 1))
				loss += cross_entropy_loss(y_hat, label[0])
				derivative_cross_entropy_softmax = y_hat-one_hot(label[0], 4)
				gradients = np.dot(derivative_cross_entropy_softmax, x.reshape(1, -1))
				self.weights = self.weights - alpha * gradients
			correct, wrong = self.evaluate(X_val, y_val)
			print(f"epoch {epoch}: Loss: {loss/X_train.shape[0]:.10f}, accuracy: {correct/(correct + wrong):.2f}")
			
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