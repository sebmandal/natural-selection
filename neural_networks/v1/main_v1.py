import random
import json
import numpy as np
from scipy.special import expit


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initializes the neural network with the given sizes for the input, hidden, and output layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize the weights and biases for the input-to-hidden and hidden-to-output connections
        self.weights_input_to_hidden = [
            [random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
        self.weights_hidden_to_output = [
            [random.uniform(-0.5, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]
        self.biases_hidden = [
            random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        self.biases_output = [
            random.uniform(-0.5, 0.5) for _ in range(output_size)]

    def train(self, input_values, target_values, learning_rate):
        # Feeds the input values through the network to get the predicted output
        hidden_outputs = [self._sigmoid(sum(input_values[i] * self.weights_input_to_hidden[i][j] + self.biases_hidden[j]
                                            for i in range(self.input_size)))
                          for j in range(self.hidden_size)]
        output = [self._sigmoid(sum(hidden_outputs[h] * self.weights_hidden_to_output[h][o] + self.biases_output[o]
                                    for h in range(self.hidden_size)))
                  for o in range(self.output_size)]

        # Calculates the error in the output
        error = [target_values[o] - output[o] for o in range(self.output_size)]

        # Calculates the errors in the hidden layer
        hidden_errors = [sum(error[o] * self.weights_hidden_to_output[h][o] for o in range(self.output_size))
                         for h in range(self.hidden_size)]

        # Updates the weights and biases using backpropagation
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                self.weights_hidden_to_output[h][o] += learning_rate * \
                    error[o] * hidden_outputs[h]
        self.biases_output[o] += learning_rate * error[o]

        for i in range(self.input_size):
            for h in range(self.hidden_size):
                self.weights_input_to_hidden[i][h] += learning_rate * \
                    hidden_errors[h] * input_values[i]
        self.biases_hidden[h] += learning_rate * hidden_errors[h]

    def predict(self, input_values):
        # Feeds the input values through the network to get the predicted output
        hidden_inputs = [sum(input_values[i] * self.weights_input_to_hidden[i][j] + self.biases_hidden[j]
                             for i in range(self.input_size))
                         for j in range(self.hidden_size)]
        hidden_outputs = [self._sigmoid(x) for x in hidden_inputs]
        output_inputs = [sum(hidden_outputs[h] * self.weights_hidden_to_output[h][o] + self.biases_output[o]
                             for h in range(self.hidden_size))
                         for o in range(self.output_size)]
        output = [self._sigmoid(x) for x in output_inputs]
        return output

    def load(self, file_path):
        with open(file_path) as file:
            data = json.load(file)
            self.weights_input_to_hidden = data["weights_input_to_hidden"]
            print("changed", data["weights_input_to_hidden"])
            self.weights_hidden_to_output = data["weights_hidden_to_output"]
            self.biases_hidden = data["biases_hidden"]
            self.biases_output = data["biases_output"]

    def _sigmoid(self, x):
        return expit(x)


"""
Training the NN
ADDITION
ITERATIONS           - 1 -> 100000
INCREASE PER ITER    - 1 -> 10
LEARNING RATE        - 0.01
"""


def addition_train():
    training_amount = 1001
    inc = 11
    learning_rate = 0.1
    for x in np.arange(1, training_amount):
        for i in range(1, inc):
            input_values = [(x + 0 * inc) / (x * 4 * inc),
                            (x + 1 * inc) / (x * 4 * inc),
                            (x + 2 * inc) / (x * 4 * inc)]
            target_values = [(x + 3 * inc) / (x * 4 * inc)]
            nn.train(input_values, target_values, learning_rate)


"""
Training the NN
EXPONENTIAL
ITERATIONS           - 1   -> 1000
LEARNING RATE        - 0.01 -> 100
EXPONENT STEP        - 0.01
"""


def exponential_train():
    training_amount = 1001
    for x in np.arange(1, training_amount, 0.001):
        input_values = [(x ** 1) / (x ** 5),
                        (x ** 2) / (x ** 5),
                        (x ** 3) / (x ** 5)]
        target_values = [(x ** 4) / (x ** 5)]

        learning_rate = min(100, x / 100)
        nn.train(input_values, target_values, learning_rate)


"""
Saving the NN weights and biases
"""


def write_to_file(filepath):
    # relative to terminal curdir
    with open(filepath, "w") as f:
        to_write = {
            "input_size": nn.input_size,
            "hidden_size": nn.hidden_size,
            "output_size": nn.output_size,
            "weights_input_to_hidden": nn.weights_input_to_hidden,
            "weights_hidden_to_output": nn.weights_hidden_to_output,
            "biases_hidden": nn.biases_hidden,
            "biases_output": nn.biases_output,
        }
        f.write(json.dumps(to_write))


"""
generate input arrays
"""


def transform_func_addition(x, inc, length):
    return [(x + i * inc) / (x * (length + 1)) for i in range(length)], (x + length) / (x * (length + 1))
    # input 4, 3
    # hidden [4, 5, 6]
    # output [0.25, 0.3125, 0.375]


def transform_func_exponent(x, length):
    # returns (input_values, target)
    return [(x ** i) / (x ** (length + 1)) for i in range(length)], (x ** length) / (x ** (length + 1))


"""
playground
"""

nn = NeuralNetwork(3, 12, 1)
values, target = transform_func_addition(4, 1, 3)
addition_train()
print("target", target)
print("actual", nn.predict(values)[0])
