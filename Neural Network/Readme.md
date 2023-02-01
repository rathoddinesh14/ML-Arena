# Neural Network Forward Propagation Implementation in Python using Numpy
The code above is an implementation of forward propagation in a neural network using Numpy. It demonstrates the creation of a dense layer and the forward pass of training data through it. The implementation is based on the Layer_Dense class, which takes two arguments as inputs: n_inputs and n_neurons. n_inputs represents the number of input features in the data, while n_neurons is the number of neurons in the layer.

## Initializing Weights and Biases
The weights and biases of the layer are initialized in the class constructor using Numpy's random.randn() function. The weights are generated with random values from a normal distribution with mean 0 (mu) and standard deviation 0.1 (sigma). The biases are initialized as an array of zeros with a shape of (1, n_neurons).

## Forward Propagation
The forward propagation process is implemented in the forward method, which takes the inputs as an argument. The dot product of the inputs and weights is calculated using Numpy's dot function, and the biases are added to it. The result is stored in the output attribute.