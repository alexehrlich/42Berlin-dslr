import numpy as np

def softmax(z):
	"""
		Subtract the max val for numerical stabilty. Does not 
		change the result of the softmax, since it is invariant.
	"""
	temp = np.exp(z - np.max(z))
	return temp / np.sum(temp)

def one_hot(label, nodes):
	"""
	Encode the target value to make it comparable to the network's output vector.
	"""
	one_hot_vector = np.zeros((nodes, 1))
	one_hot_vector[label] = 1
	return one_hot_vector

def cross_entropy_loss(y_hat, y):
	"""
	Sum up the log losses of every output class.
	The epsilon is added for numerical stability.
	"""
	one_hot_target = one_hot(y, len(y_hat))
	epsilon = 1e-15
	y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
	log_predictions = np.log(y_hat)
	temp = np.multiply(log_predictions, one_hot_target)
	loss = -np.sum(temp)
	return loss
