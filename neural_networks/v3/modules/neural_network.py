import numpy as np
from scipy.special import expit, logit


def activate(x):
    """
    Activation function for the neural network.
    """
    return np.nan_to_num(expit(x))


def activate_derivative(x):
    """
    Derivative of the activation function for the neural network.
    """
    return np.nan_to_num(logit(x))


class NeuralNetwork:
    def __init__(self, layer_nodes: list):
        """
        Initializes the neural network with the specified number of nodes for each layer.

        Parameters
        ----------
        layer_nodes : list
            A list containing the number of nodes for each layer, including the input and output layers.
            The first element should be the number of input nodes, and the last element should be the number of output nodes.
            The elements in between represent the number of nodes in each hidden layer, in order.
        """
        input_nodes = layer_nodes.pop(0)
        output_nodes = layer_nodes.pop(-1)

        # Initialize the input layer
        self.layers = [
            {
                "nodes": input_nodes,
            }
        ]

        # Initialize the hidden layers
        for i in range(len(layer_nodes)):
            last_layer_nodes = layer_nodes[i-1]

            # Initialize the weights and biases for the connections
            self.layers.append({
                "nodes": layer_nodes[i],
                "weights": np.random.uniform(-0.5, 0.5, size=(last_layer_nodes, layer_nodes[i])),
                "biases": np.random.uniform(-0.5, 0.5, size=(layer_nodes[i])),
            })

        # Initialize the output layer
        self.layers.append({
            "nodes": output_nodes,
            "weights": np.random.uniform(-0.5, 0.5, size=(layer_nodes[-1], output_nodes)),
            "biases": np.random.uniform(-0.5, 0.5, size=(output_nodes)),
        })

    def train(self, input_values, target_values, learning_rate):
        """
        Trains the neural network using the given input and target values, with the specified learning rate.

        Parameters
        ----------
        input_values : list
            A list containing the input values for the training example.
        target_values : list
            A list containing the target values for the training example.
        learning_rate : float
            The learning rate to use for training.
        """

        # Set input values
        self.layers[0]["output"] = input_values

        # Use activate function to generate the outputs
        for i, layer in enumerate(self.layers):
            if i == 0:
                continue

            layer["output"] = []

            for weights, bias in zip(layer["weights"], layer["biases"]):
                # Initialize variable for output of current neuron
                neuron_output = 0

                # Otherwise, use the output of the previous layer as input
                for x, weight in zip(self.layers[i-1]["output"], weights):
                    neuron_output += x * weight

                neuron_output += bias
                layer["output"].append(neuron_output)

            print(layer)

    def predict(self, input_values):
        # Use activate function to generate the outputs
        for i, layer in enumerate(self.layers):
            if i == 0:
                continue

            layer["output"] = []

            for weights, bias in zip(layer["weights"], layer["biases"]):
                # Initialize variable for output of current neuron
                neuron_output = 0

                if i == 1:
                    # If current layer is the first hidden layer, use the input values as input
                    for x, weight in zip(input_values, weights):
                        neuron_output += x * weight
                else:
                    # Otherwise, use the output of the previous layer as input
                    for x, weight in zip(self.layers[i-1]["output"], weights):
                        neuron_output += x * weight

                neuron_output += bias
                layer["output"].append(neuron_output)

        return self.layers[-1]["output"]


nn = NeuralNetwork([3, 8, 8, 8, 1])
for x in range(1, 10000000):
    learning_rate = 0.1

    input_values = [x, x * 2]
    target_values = [x * 3]

    nn.train(input_values, target_values, learning_rate)
