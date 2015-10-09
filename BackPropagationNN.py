import numpy as np
import time

class NeuralNetwork(object):

	def __init__(self, inputs, hidden, outputs, activation='tanh', output_act='softmax'):
		
		# Hidden layer activation function
		if activation == 'sigmoid':
			self.activation = sigmoid
			self.activation_prime = sigmoid_prime
		elif activation == 'tanh':
			self.activation = tanh
			self.activation_prime = tanh_prime
		elif activation == 'linear':
			self.activation = linear
			self.activation_prime = linear_prime

		# Output layer activation function
		if output_act == 'sigmoid':
			self.output_act = sigmoid
			self.output_act_prime = sigmoid_prime
		elif output_act == 'tanh':
			self.output_act = tanh
			self.output_act_prime = tanh_prime
		elif output_act == 'linear':
			self.output_act = linear
			self.output_act_prime = linear_prime
		elif output_act == 'softmax':
			self.output_act = softmax
			self.output_act_prime = linear_prime

		# Weights initializarion
		self.wi = np.random.randn(inputs, hidden)/np.sqrt(inputs)
		self.wo = np.random.randn(hidden + 1, outputs)/np.sqrt(hidden)

		# Weights updates initialization
		self.updatei = 0
		self.updateo = 0


	def feedforward(self, X):

		# Hidden layer activation
		ah = self.activation(np.dot(X, self.wi))
			
		# Adding bias to the hidden layer results
		ah = np.concatenate((np.ones(1).T, np.array(ah)))

		# Outputs
		y = self.output_act(np.dot(ah, self.wo))

		# Return the results
		return y


	def fit(self, X, y, epochs=10, learning_rate=0.2, learning_rate_decay = 0 , momentum = 0, verbose = 0):
		
		# Timer start
		startTime = time.time()

		# Epochs loop
		for k in range(epochs):
	
			# Dataset loop
			for i in range(X.shape[0]):

				# Hidden layer activation
				ah = self.activation(np.dot(X[i], self.wi))
			
				# Adding bias to the hidden layer
				ah = np.concatenate((np.ones(1).T, np.array(ah))) 

				# Output activation
				ao = self.output_act(np.dot(ah, self.wo))

				# Deltas	
				deltao = np.multiply(self.output_act_prime(ao),y[i] - ao)
				deltai = np.multiply(self.activation_prime(ah),np.dot(self.wo, deltao))

				# Weights update with momentum
				self.updateo = momentum*self.updateo + np.multiply(learning_rate, np.outer(ah,deltao))
				self.updatei = momentum*self.updatei + np.multiply(learning_rate, np.outer(X[i],deltai[1:]))

				# Weights update
				self.wo += self.updateo
				self.wi += self.updatei

			# Print training status
			if verbose == 1:
				print 'EPOCH: {0:4d}/{1:4d}\t\tLearning rate: {2:4f}\t\tElapse time [seconds]: {3:5f}'.format(k,epochs,learning_rate, time.time() - startTime)
				
			# Learning rate update
			learning_rate = learning_rate * (1 - learning_rate_decay)

	def predict(self, X): 

		# Allocate memory for the outputs
		y = np.zeros([X.shape[0],self.wo.shape[1]])

		# Loop the inputs
		for i in range(0,X.shape[0]):

			y[i] = self.feedforward(X[i])

		# Return the results
		return y


# Activation functions
def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
	return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
	return np.tanh(x)

def tanh_prime(x):
	return 1.0 - x**2

def softmax(x):
    return (np.exp(np.array(x)) / np.sum(np.exp(np.array(x))))

def softmax_prime(x):
    return softmax(x)*(1.0-softmax(x))

def linear(x):
	return x

def linear_prime(x):
	return 1
