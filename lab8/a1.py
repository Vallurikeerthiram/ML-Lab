# A1: Perceptron Components - Summation, Activation Units, and Error Calculation
import numpy as np
import matplotlib.pyplot as plt

def summation_unit(inputs, weights, bias):
    """
    Calculates weighted sum of inputs plus bias
    inputs: list of input values
    weights: list of weight values  
    bias: bias value
    """
    return np.dot(inputs, weights) + bias

def step_activation(x):
    """Step activation function"""
    return 1 if x >= 0 else 0

def bipolar_step_activation(x):
    """Bipolar step activation function"""
    return 1 if x >= 0 else -1

def sigmoid_activation(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def tanh_activation(x):
    """Hyperbolic tangent activation function"""
    return np.tanh(x)

def relu_activation(x):
    """ReLU activation function"""
    return max(0, x)

def leaky_relu_activation(x, alpha=0.01):
    """Leaky ReLU activation function"""
    return x if x > 0 else alpha * x

def comparator_unit(predicted, actual):
    """
    Error calculation unit
    """
    error = actual - predicted
    return error, error**2

# Test the functions
if __name__ == "__main__":
    # Test data
    inputs = [0, 1]
    weights = [0.2, -0.75]
    bias = 10

    # Test summation unit
    net_input = summation_unit(inputs, weights, bias)
    print(f"Net input: {net_input}")

    # Test all activation functions
    print(f"Step: {step_activation(net_input)}")
    print(f"Bipolar Step: {bipolar_step_activation(net_input)}")
    print(f"Sigmoid: {sigmoid_activation(net_input)}")
    print(f"Tanh: {tanh_activation(net_input)}")
    print(f"ReLU: {relu_activation(net_input)}")
    print(f"Leaky ReLU: {leaky_relu_activation(net_input)}")

    # Test error calculation
    predicted = step_activation(net_input)
    actual = 0  # Expected output for AND gate [0,1] -> 0
    error, sq_error = comparator_unit(predicted, actual)
    print(f"Error: {error}, Squared Error: {sq_error}")
