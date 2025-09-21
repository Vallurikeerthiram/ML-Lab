# A2: Perceptron for AND Gate with Step Activation
import numpy as np
import matplotlib.pyplot as plt

def summation_unit(inputs, weights, bias):
    """Calculates weighted sum of inputs plus bias"""
    return np.dot(inputs, weights) + bias

def step_activation(x):
    """Step activation function"""
    return 1 if x >= 0 else 0

def comparator_unit(predicted, actual):
    """Error calculation unit"""
    error = actual - predicted
    return error, error**2

class Perceptron:
    def __init__(self, weights, bias, learning_rate=0.05):
        self.weights = np.array(weights)
        self.bias = bias
        self.learning_rate = learning_rate
        self.errors = []
        self.epoch_count = 0

    def predict(self, inputs):
        net_input = summation_unit(inputs, self.weights, self.bias)
        return step_activation(net_input)

    def train(self, X, y, max_epochs=1000, convergence_error=0.002):
        """Train perceptron using given training data"""
        for epoch in range(max_epochs):
            epoch_error = 0
            for i in range(len(X)):
                # Forward pass
                predicted = self.predict(X[i])

                # Calculate error
                error, sq_error = comparator_unit(predicted, y[i])
                epoch_error += sq_error

                # Update weights and bias if error exists
                if error != 0:
                    self.weights += self.learning_rate * error * np.array(X[i])
                    self.bias += self.learning_rate * error

            # Calculate sum squared error for epoch
            sse = epoch_error / len(X)
            self.errors.append(sse)
            self.epoch_count = epoch + 1

            # Check convergence
            if sse <= convergence_error:
                print(f"Converged at epoch {epoch + 1} with error {sse}")
                break

        if sse > convergence_error:
            print(f"Did not converge after {max_epochs} epochs. Final error: {sse}")

    def plot_error(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.errors) + 1), self.errors, 'b-', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Sum Squared Error')
        plt.title('Error vs Epochs - AND Gate Learning with Step Activation')
        plt.grid(True)
        plt.savefig('A2_and_gate_error_plot.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # AND gate training data
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 0, 0, 1]

    # Initialize perceptron with given parameters
    # W0 = 10, W1 = 0.2, w2 = -0.75, learning rate (α) = 0.05
    perceptron = Perceptron(weights=[0.2, -0.75], bias=10, learning_rate=0.05)

    print("Initial weights:", perceptron.weights)
    print("Initial bias:", perceptron.bias)
    print("Learning rate:", perceptron.learning_rate)

    # Train the perceptron
    print("\nTraining perceptron on AND gate...")
    perceptron.train(X, y)

    print(f"\nFinal weights: {perceptron.weights}")
    print(f"Final bias: {perceptron.bias}")
    print(f"Total epochs: {perceptron.epoch_count}")

    # Test the trained perceptron
    print("\nTesting trained perceptron:")
    for i, inputs in enumerate(X):
        prediction = perceptron.predict(inputs)
        print(f"Input: {inputs}, Expected: {y[i]}, Predicted: {prediction}")

    # Plot error vs epochs
    perceptron.plot_error()

if __name__ == "__main__":
    main()
