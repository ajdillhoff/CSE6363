
This assignment covers neural networks, backpropagation, and cross-validation techniques.

# Neural Network Library

In the first part of this assignment, you will create a neural network library. The library will be made up of documented classes and functions that allow users to easily construct a neural network with an arbitrary number of layers and nodes. Through implementing this library, you will understand more clearly the atomic components that make up a basic neural network.

## The `Layer` Class

For the layers you will create in this assignment, it is worth it to create a parent class named `Layer` which defined the forward and backward functions that are used by all layers. In this way, we can take advantage of polymorphism to easily compute the forward and backward passes of the entire network.

## `Linear` Layer

Create a class that implements a linear layer. The class should inherit the `Layer` class and implement both a `forward` and `backward` function. For a given input, the forward pass is computed as

$$
f(\mathbf{x}; \mathbf{w}) = \mathbf{x} \mathbf{w}^T + \mathbf{b}.
$$

Here, $\mathbf{x} \in \mathbb{R}^{n \times d}$, $\mathbf{w} \in \mathbb{R}^{h \times d}$, and $\mathbf{b} \in \mathbb{R}^h$, where $n$ is the number of samples, $d$ is the number of input features, and $h$
is the number of output features.

The backward pass should compute the gradient with respect to the weights and bias:

$$
\frac{d}{d\mathbf{w}} f(\mathbf{x}; \mathbf{w}) = \mathbf{x}\\
\frac{d}{d\mathbf{w}} f(\mathbf{x}; \mathbf{w}) = \mathbf{1}
$$

This is then multiplied with the gradients computed by the layer ahead of this one.

Since there may be multiple layers, it should additionally compute $\frac{df}{d\mathbf{x}}$ to complete a chain of backward passes.

## `Sigmoid` Function

Create a class that implements the logistic sigmoid function. The class should inherit the `Layer` class and implement both `forward` and `backward` functions.

It is useful to store the output of forward pass of this layer as a class member so that it may be reused when calling `backward`.

The forward pass is defined as

$$
f(x) = \frac{1}{1 + e^{-x}}.
$$

The backward pass is defined as

$$
\frac{df}{dx} = f(x) (1 - f(x)).
$$

## Hyperbolic Tangent Function

Create a class that implements the hyperbolic tangent function. The class should inherit the `Layer` class and implement both `forward` and `backward` functions.

The forward pass is defined as

$$
f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}.
$$

The backward pass is defined as

$$
\frac{df}{dx} = 1 - f(x)^2.
$$

## The `Softmax` Function

Create a class that implements the `softmax` function. The class should inherit the `Layer` class and implement
the `forward` and `backward` functions.

Since we are only using this with cross-entropy loss, you can simplify the `backward` pass so that it only passes the gradient input.

The softmax function is defined as

$$
p_k = \frac{\exp{f_k}}{\sum_j \exp{f_j}}.
$$

## Cross-Entropy Loss

Create a class that implements cross-entropy loss. The class should inherit the `Layer` class and implement both `forward` and `backward` functions.

Feel free to use the code defined in the class examples when integrating this with your library.

The backward pass, when combined with softmax, is

$$
\frac{df}{dx} = p - y.
$$

## The `Sequential` Class

In order to create a clean interface that includes multiple layers, you will need to create a class that contains a list of layers which make up the network. The `Sequential` class will contain a list of layers. New layers can be added to it by appending them to the current list. This class will also inherit from the `Layer` class so that it can call forward and backward as required.

## Saving and Loading

Implement a weight saving and loading feature for a constructed network such that all model weights can be saved to and loaded form a file. This will enable trained models to be stored and shared.

# Testing your library

Construct a neural network with 1 hidden layer of 2 nodes in order to solve the XOR problem. Construct the input using `numpy`. You can reference the code we used for multi-layer perceptrons in class to help. Train and verify that your model can solve the XOR problem.

This may take many attempts to converge to a solution depending on your architecture, choice of activation function, learning rate, and other factors. Attempt to solve this problem with the architecture described above using sigmoid activations and then again using hyperbolic tangent activations. This code should be in a file named `fit_XOR.py` and should use the neural network functions you implemented previously. These should be imported as if you were using a library. In the same file, leave a comment describing the hyperparameters you used to solve the problem.

Save the weights as `XOR_solved.w`.

## Handwritten Digit Recognition

In the second part of the assignment, you will use your neural network library to construct several networks for handwritten digit recognition. There are 60000 training images and 10000 testing images. You should randomly select 10% of the training images to use as a validation set.

In this part, you can experiment with the number of layers and nodes per layer as you wish. Use the loss of the validation set to guide your selection of hyperparameters. Experiment with at least 3 configurations of hyperparameters, plotting the training and validation loss as you train each configuration. Stop training when the loss does not improve after 5 steps (**early stopping**). Your code should produce the training/validation plots with each choice of hyperparameters.

Once you have trained at least 3 different models, evaluate each one on the test set. Include the test accuracy with your output. The training and testing can be completed in a single file named `fit_MNIST.py`. Before training each model, print a summary of the network and hyperparameters used.

**Example Training Output**
    
```
Model 1
784 -> 100 -> 10
Activation Function: Sigmoid
Learning rate: 0.01
Batch size: 128
Max Epochs: 10
Early stopping: 5
```

**Example Testing Output**

```
Model 1
Test accuracy: 0.81
```

Save the weights of each model as `MNIST_model1.w`, `MNIST_model2.w`, etc.

# Submission

Create a zip file that includes all relevant code. The TA should be able to easily run the code to reproduce all plots and results. Include any additional instructions, if necessary.
