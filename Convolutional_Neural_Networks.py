import numpy as np
#from layer import Layer
from scipy import signal
from tensorflow.keras.datasets import mnist

# class ConvolutionalNeuralNetwork(Layer):
class ConvolutionalNeuralNetwork():
    def __init__(self, input_shape, kernel_size, depth):    # depthn = num of kernels
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth

        # Size of O/P = Size of I/P - Size of Kernel + 1
        # Thus shape of O/P = (num of o/p, o/p colm, o/p row)
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)

        # this is 4D bcz we need to know the num of kernel, depth of each, their lenght and width
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)

        self.kernels = np.random.rand(*self.kernels_shape)
        self.biases = np.random.rand(*self.output_shape)    # shape of biases = shape of output


    def forward(self, input):
        self.input = input

        # Initailly, output is bias bcz then we just add X ★ K to update
        self.output = np.copy(self.biases)

        # Y[i] = B[i] + Σ(j = 1 to n) X[j] ★ K[i][j]..... i = 1,..., depth
        # Remember B[i] is already accounted for in self.output = np.copy(self.biases)
        # ★ is performed by correlated2d that HAS TO BE (X, K, "valid" or "full")
        # mention "valid"/"full" for no padding/padding
        # alt + 228(numeric keyboard) = Σ
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.convolve2d(self.input[j], self.kernels[i, j], "valid")

        return self.output


    def backwards(self, output_gradient, learning_rate):
        # Intial gradients are 0, we will update them as we calculate the gradients for each kernel and input
        # bias_gradient = output_gradient so we only update it at the end
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        # Gradient of kernel[i][j] = X[j] ★ output_gradient[i]
        # Gradient of bias[i] = output_gradient[i]................... we will update it at the end
        # Gradient of input[j] = output_gradient[i] ★ kernel[i][j]
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.convolve2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient

        return input_gradient


class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * (self.x > 0)


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(1, -1)

    def backward(self, grad):
        return grad.reshape(self.shape)


class Dense:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros((1, out_features))

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad, lr):

        dW = np.dot(self.x.T, grad)
        db = np.sum(grad, axis=0, keepdims=True)
        dx = np.dot(grad, self.W.T)

        self.W -= lr * dW
        self.b -= lr * db

        return dx


class SimpleCNN:
    def __init__(self):

        self.conv = ConvolutionalNeuralNetwork(
            input_shape=(1, 28, 28),
            kernel_size=3,
            depth=8
        )

        self.relu = ReLU()
        self.flatten = Flatten()

        self.fc = Dense(8 * 26 * 26, 10)


    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


    def lossCategoricalCrossEntropy(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        loss = -np.sum(y_true * np.log(y_pred), axis=1)
        return np.mean(loss)


    def forward(self, x):
        z = self.conv.forward(x)
        z = self.relu.forward(z)
        z = self.flatten.forward(z)
        z = self.fc.forward(z)

        self.out = self.softmax(z)

        return self.out


    def backward(self, y_true, lr):

        dZ = self.out - y_true

        d = self.fc.backward(dZ, lr)
        d = self.flatten.backward(d)
        d = self.relu.backward(d)
        self.conv.backwards(d, lr)
