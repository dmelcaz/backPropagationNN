import numpy as np

from BackPropagationNN import NeuralNetwork

from sklearn import datasets
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import metrics

def targetToVector(x):
	# Vector
	a = np.zeros([len(x),10])
	for i in range(0,len(x)):
		a[i,x[i]] = 1
	return a

if __name__ == '__main__':

	# Digits dataset loading
	digits = datasets.load_digits()
	X = preprocessing.scale(digits.data.astype(float))
	y = targetToVector(digits.target)

	# Cross valitation
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
	
	# Neural Network initialization
	NN = NeuralNetwork(64,60,10, output_act = 'softmax')
	NN.fit(X_train,y_train, epochs = 50, learning_rate = .1, learning_rate_decay = .01, verbose = 1)

	# NN predictions
	y_predicted = NN.predict(X_test)

	# Metrics
	y_predicted = np.argmax(y_predicted, axis=1).astype(int)
	y_test = np.argmax(y_test, axis=1).astype(int)

	print("\nClassification report for classifier:\n\n%s\n"
	  % (metrics.classification_report(y_test, y_predicted)))
	print("Confusion matrix:\n\n%s" % metrics.confusion_matrix(y_test, y_predicted))
		