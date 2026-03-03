# simple-backprop

simple-backprop is a tiny, from-scratch backpropagation neural network with one hidden layer, implemented in pure NumPy.

## Features
- One hidden layer MLP with momentum and learning-rate decay
- Hidden activations: `sigmoid`, `tanh`, `relu`, `linear`
- Output activations: `sigmoid`, `tanh`, `relu`, `linear`, `softmax`
- Minimal sklearn demo on the digits dataset (~97% accuracy with default settings)

## Requirements
- Python 3
- NumPy
- scikit-learn (for the demo)

## Quick start
Install dependencies:
```bash
pip3 install numpy scikit-learn
```

Run the demo:
```bash
python3 demo.py
```

Choose the hidden-layer activation:
```bash
python3 demo.py --activation relu
```

## Usage
```python
from BackPropagationNN import NeuralNetwork

nn = NeuralNetwork(
    inputs=64,
    hidden=60,
    outputs=10,
    activation='tanh',
    output_act='softmax',
)

nn.fit(X_train, y_train, epochs=50, learning_rate=0.1, learning_rate_decay=0.01)
preds = nn.predict(X_test)
```

## Notes
- The demo uses a one-hot target vector; see `demo.py` for a reference implementation.
- This is a learning-oriented implementation, not a production library.
