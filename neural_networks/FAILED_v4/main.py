import numpy as np
np.random.seed(0)


def spiral_data(points, classes):
    # https://cs231n.github.io/neural-networks-case-study/
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4,
                        points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_Categorical_Cross_Entropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            # for scalar values
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            # one-hot encoded values
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# X, y data is (x, y), so only a 2 wide
# 100 samples, 3 classes
X, y = spiral_data(100, 3)

# 2 because spiral_data, 3 because 3 neurons
dense_1 = Layer_Dense(2, 3)
activ_1 = Activation_ReLU()

# 3 because 3 neurons in dense_1, we will also have 3 neurons here
dense_2 = Layer_Dense(3, 3)
activ_2 = Activation_Softmax()

# actually processing the data
dense_1.forward(X)
activ_1.forward(dense_1.output)
dense_2.forward(activ_1.output)
activ_2.forward(dense_2.output)


loss_function = Loss_Categorical_Cross_Entropy()
loss = loss_function.calculate(activ_2.output, y)

print(loss)
