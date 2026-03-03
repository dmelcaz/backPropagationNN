import time
import numpy as np


class NeuralNetwork(object):

    def __init__(self, inputs, hidden, outputs, activation='tanh',
                 output_act='softmax'):

        # Hidden layer activation function
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime
        elif activation == 'relu':
            self.activation = relu
            self.activation_prime = relu_prime
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
        elif output_act == 'relu':
            self.output_act = relu
            self.output_act_prime = relu_prime
        elif output_act == 'linear':
            self.output_act = linear
            self.output_act_prime = linear_prime
        elif output_act == 'softmax':
            self.output_act = softmax
            self.output_act_prime = softmax_prime

        # Weights initialization
        self.wi = np.random.randn(inputs, hidden) / np.sqrt(inputs)
        self.wo = np.random.randn(hidden + 1, outputs) / np.sqrt(hidden)

        # Weights updates initialization
        self.updatei = 0
        self.updateo = 0

    def feed_forward(self, x):

        # Hidden layer activation
        ah = self.activation(np.dot(x, self.wi))

        # Adding bias to the hidden layer results
        ah = np.concatenate((np.ones(1).T, np.array(ah)))

        # Outputs
        y = self.output_act(np.dot(ah, self.wo))

        # Return the results
        return y

    # Back-compat alias for previous public method name.
    def feedforward(self, x):
        return self.feed_forward(x)

    def fit(self, x, y, epochs=10, learning_rate=0.2,
            learning_rate_decay=0, momentum=0, verbose=0):

        # Timer start
        start_time = time.time()

        # Epochs loop
        for k in range(epochs):

            # Dataset loop
            for i in range(x.shape[0]):

                # Hidden layer activation
                ah = self.activation(np.dot(x[i], self.wi))

                # Adding bias to the hidden layer
                ah_bias = np.concatenate((np.ones(1).T, np.array(ah)))

                # Output activation
                ao = self.output_act(np.dot(ah_bias, self.wo))

                # Deltas
                if self.output_act == softmax:
                    deltao = np.dot(softmax_jacobian(ao), (y[i] - ao))
                else:
                    deltao = np.multiply(self.output_act_prime(ao), y[i] - ao)
                deltai = np.multiply(
                    self.activation_prime(ah), np.dot(self.wo[1:], deltao)
                )

                # Weights update with momentum
                self.updateo = momentum * self.updateo + np.multiply(
                    learning_rate, np.outer(ah_bias, deltao)
                )
                self.updatei = momentum * self.updatei + np.multiply(
                    learning_rate, np.outer(x[i], deltai)
                )

                # Weights update
                self.wo += self.updateo
                self.wi += self.updatei

            # Print training status
            if verbose == 1:
                print(
                    'EPOCH: {0:4d}/{1:4d}\t\tLearning rate: {2:4f}\t\t'
                    'Elapse time [seconds]: {3:5f}'.format(
                        k, epochs, learning_rate, time.time() - start_time
                    )
                )

            # Learning rate update
            learning_rate = learning_rate * (1 - learning_rate_decay)

    def predict(self, x):

        # Allocate memory for the outputs
        y = np.zeros([x.shape[0], self.wo.shape[1]])

        # Loop the inputs
        for i in range(0, x.shape[0]):
            y[i] = self.feed_forward(x[i])

        # Return the results
        return y


# Activation functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return x * (1.0 - x)


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1.0 - x**2


def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0).astype(x.dtype)

def softmax(x):
    return np.exp(np.array(x)) / np.sum(np.exp(np.array(x)))


def softmax_jacobian(s):
    s = np.array(s)
    return np.diag(s) - np.outer(s, s)


def softmax_prime(x):
    return softmax_jacobian(softmax(x))


def linear(x):
    return x


def linear_prime(x):
    return 1
