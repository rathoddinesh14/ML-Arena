import numpy as np
import nnfs

from nnfs.datasets import spiral_data

nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # weights are initialized with random values from normal distribution
        # with mean 0(mu) and standard deviation 0.1(sigma)
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.layer_shape = (n_inputs, n_neurons)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
    

class Activation_Softmax:
    def forward(self, inputs):
        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        # calculate sample losses
        sample_losses = self.forward(output, y)
        # calculate mean loss
        data_loss = np.mean(sample_losses)
        # return loss
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # number of samples in a batch
        samples = len(y_pred)
        # clip data to prevent division by 0
        # clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            # (y_pred * y_true) performs element-wise multiplication
            # np.sum(, axis=1) sums the values of each row
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(3, 3)

# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()

lowest_loss = 9999999  # some initial value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()


plt_acc = []
plt_loss = []
plt_best_loss = []


def tweak_params(layer):
    layer.weights += 0.05 * np.random.randn(layer.layer_shape[0], layer.layer_shape[1])
    layer.biases  += 0.05 * np.random.randn(1, layer.layer_shape[1])

for iteration in range(10001):

    # new weights for each iteration
    tweak_params(dense1)
    tweak_params(dense2)

    # Make a forward pass of our training data through this layer
    dense1.forward(X)

    # Make a forward pass through activation function
    # it takes the output of first dense layer here
    activation1.forward(dense1.output)

    # Make a forward pass through second Dense layer
    # it takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)

    # Make a forward pass through activation function
    # it takes the output of second dense layer here
    activation2.forward(dense2.output)

    # # Let's see output of the first few samples:
    # print(activation2.output[:5])

    # Calculate loss from output of activation2 and targets
    loss = loss_function.calculate(activation2.output, y)

    # Print loss value
    # print('loss:', loss)

    # calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(activation2.output, axis=1)
    # print("predictions:", predictions)

    # print("y:", y, len(y.shape))
    if len(y.shape) == 2:
        # check if targets are one-hot encoded
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)

    # print("acc:", accuracy)

    plt_acc.append(accuracy)
    plt_loss.append(loss)

    if loss < lowest_loss:
        lowest_loss = loss
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()

        print(f'iteration: {iteration}, '
            f'acc: {accuracy:.3f}, '
            f'loss: {loss:.3f}, '
            f'best_loss: {lowest_loss:.3f}')

    else:
        # revert weights and biases
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
    
    plt_best_loss.append(lowest_loss)

import matplotlib.pyplot as plt

plt.plot(plt_acc, label='acc')
plt.plot(plt_loss, label='loss')
plt.plot(plt_best_loss, label='best_loss')
plt.legend()
# plt.show()
plt.savefig('acc_loss.png')