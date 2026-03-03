import argparse
import numpy as np

from sklearn import datasets
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

from BackPropagationNN import NeuralNetwork


def target_to_vector(x):
    # Vector
    a = np.zeros([len(x), 10])
    for i in range(0, len(x)):
        a[i, x[i]] = 1
    return a


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BackPropagationNN demo')
    parser.add_argument(
        '--activation', default='tanh',
        choices=['sigmoid', 'tanh', 'relu', 'linear'],
        help='Hidden layer activation function'
    )
    args = parser.parse_args()

    # Digits dataset loading
    digits = datasets.load_digits()
    x_data = preprocessing.scale(digits.data.astype(float))
    y_data = target_to_vector(digits.target)

    # Cross validation
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x_data, y_data, test_size=0.2, random_state=0
    )

    # Neural Network initialization
    nn = NeuralNetwork(64, 60, 10, activation=args.activation, output_act='softmax')
    nn.fit(
        x_train, y_train, epochs=50, learning_rate=0.1,
        learning_rate_decay=0.01, verbose=1
    )

    # NN predictions
    y_predicted = nn.predict(x_test)

    # Metrics
    y_predicted = np.argmax(y_predicted, axis=1).astype(int)
    y_test = np.argmax(y_test, axis=1).astype(int)

    print(
        "\nClassification report for classifier:\n\n%s\n"
        % (metrics.classification_report(y_test, y_predicted))
    )
    print("Confusion matrix:\n\n%s" % metrics.confusion_matrix(y_test, y_predicted))
