# A3: Compare Different Activation Functions for AND Gate
import numpy as np
import matplotlib.pyplot as plt

def summation_unit(inputs, weights, bias):
    """Calculates weighted sum of inputs plus bias"""
    return np.dot(inputs, weights) + bias

def step_activation(x):
    """Step activation function"""
    return 1 if x >= 0 else 0

def bipolar_step_activation(x):
    """Bipolar step activation function"""
    return 1 if x >= 0 else -1

def sigmoid_activation(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

def relu_activation(x):
    """ReLU activation function"""
    return max(0, x)

def comparator_unit(predicted, actual):
    """Error calculation unit"""
    error = actual - predicted
    return error, error**2

class PerceptronMultiActivation:
    def __init__(self, weights, bias, learning_rate=0.05, activation_func='step'):
        self.weights = np.array(weights, dtype=float)
        self.bias = float(bias)
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.errors = []
        self.epoch_count = 0

    def activation(self, x):
        if self.activation_func == 'step':
            return step_activation(x)
        elif self.activation_func == 'bipolar_step':
            return bipolar_step_activation(x)
        elif self.activation_func == 'sigmoid':
            return sigmoid_activation(x)
        elif self.activation_func == 'relu':
            return relu_activation(x)

    def predict(self, inputs):
        net_input = summation_unit(inputs, self.weights, self.bias)
        return self.activation(net_input)

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
                print(f"{self.activation_func.upper()}: Converged at epoch {epoch + 1} with error {sse}")
                break

        if sse > convergence_error:
            print(f"{self.activation_func.upper()}: Did not converge after {max_epochs} epochs. Final error: {sse}")

        return self.epoch_count

def main():
    # AND gate training data
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 0, 0, 1]

    # For bipolar step, we need different target values
    y_bipolar = [-1, -1, -1, 1]

    activation_functions = ['step', 'bipolar_step', 'sigmoid', 'relu']
    results = {}

    print("Training perceptrons with different activation functions:")
    print("-" * 60)

    for activation in activation_functions:
        # Use bipolar targets for bipolar step function
        targets = y_bipolar if activation == 'bipolar_step' else y

        # Initialize perceptron
        perceptron = PerceptronMultiActivation(
            weights=[0.2, -0.75], 
            bias=10, 
            learning_rate=0.05,
            activation_func=activation
        )

        # Train
        epochs = perceptron.train(X, targets)
        results[activation] = {
            'epochs': epochs,
            'final_weights': perceptron.weights.copy(),
            'final_bias': perceptron.bias,
            'errors': perceptron.errors.copy()
        }

        # Test the trained perceptron
        print(f"\nTesting {activation.upper()}:")
        for i, inputs in enumerate(X):
            prediction = perceptron.predict(inputs)
            expected = targets[i]
            print(f"Input: {inputs}, Expected: {expected}, Predicted: {prediction:.4f}")

    # Plot comparison
    plt.figure(figsize=(15, 10))

    # Plot 1: Error curves
    plt.subplot(2, 2, 1)
    for activation in activation_functions:
        errors = results[activation]['errors']
        plt.plot(range(1, len(errors) + 1), errors, label=activation.replace('_', ' ').title(), linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Sum Squared Error')
    plt.title('Error vs Epochs - Different Activation Functions')
    plt.legend()
    plt.grid(True)

    # Plot 2: Convergence comparison
    plt.subplot(2, 2, 2)
    activations = [act.replace('_', ' ').title() for act in activation_functions]
    epochs_to_converge = [results[act]['epochs'] for act in activation_functions]
    plt.bar(activations, epochs_to_converge, color=['blue', 'orange', 'green', 'red'])
    plt.xlabel('Activation Function')
    plt.ylabel('Epochs to Converge')
    plt.title('Convergence Comparison')
    plt.xticks(rotation=45)

    # Summary table
    print("\n" + "=" * 80)
    print("CONVERGENCE COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Activation Function':<20} {'Epochs':<10} {'Final Weights':<25} {'Final Bias':<15}")
    print("-" * 80)
    for activation in activation_functions:
        result = results[activation]
        weights_str = f"[{result['final_weights'][0]:.3f}, {result['final_weights'][1]:.3f}]"
        print(f"{activation.replace('_', ' ').title():<20} {result['epochs']:<10} {weights_str:<25} {result['final_bias']:.3f}")

    plt.tight_layout()
    plt.savefig('A3_activation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
