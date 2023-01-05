import numpy as np


class Layer:
    def __init__(self, inputs, neurons):
        self.weights = 0.1 * np.random.randn(inputs, neurons)
        # self.biases = 0.1 * np.random.randn(1, neurons)
        self.biases = np.zeros((1, neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def relu(self, inputs):
        self.output = np.maximum(0, inputs)

    def softmax(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Net:
    def __init__(self, layers) -> None:
        self.layers = [Layer(layer[0], layer[1]) for layer in layers]

    def train(self, inputs, targets, learning_rate):
        """
        training using backpropagation
        """
        # Get the number of samples in the batch
        num_samples = np.array(inputs).shape[0]

        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.forward(inputs)
                layer.relu(layer.output)

            elif i == len(self.layers) - 1:
                layer.forward(self.layers[i-1].output)
                layer.softmax(layer.output)

            else:
                layer.forward(self.layers[i-1].output)
                layer.relu(layer.output)

        # Calculate the error
        output_layer = self.layers[-1]
        error = -np.sum(targets * np.log(output_layer.output)) / \
            np.array(targets).shape[0]

        # Backpropagate the error through the layers
        for i, layer in reversed(list(enumerate(self.layers))):
            if i == len(self.layers) - 1:
                # Input layer
                layer.error = error
            else:
                # Hidden layer
                next_layer = self.layers[i + 1]
                layer.error = np.dot(next_layer.delta, next_layer.weights.T)
            layer.delta = layer.error * self.derivative(layer.output)

        # Update the weights and biases
        for i, layer in enumerate(self.layers):
            if i == 0:
                continue
            layer.weights += learning_rate * \
                np.dot(self.layers[i-1].output.T, layer.delta) / num_samples
            layer.biases += learning_rate * \
                np.sum(layer.delta, axis=0, keepdims=True) / num_samples

    def predict(self, inputs):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.forward(inputs)
                layer.relu(layer.output)

            elif i == len(self.layers) - 1:
                layer.forward(self.layers[i-1].output)
                layer.softmax(layer.output)
                return layer.output

            else:
                layer.forward(self.layers[i-1].output)
                layer.relu(layer.output)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)


batch_size = 5
output_size = 1
net = Net([[batch_size, 8], [8, 6], [6, output_size]])

for i in range(10000):
    training_data = []

    data = [i + j for j in range(batch_size)]
    targets = i + batch_size
    net.train(data, [targets], 0.0001)
    print(net.predict(data))
