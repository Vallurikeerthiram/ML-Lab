# A5: XOR Gate Implementation (A1-A3 for XOR)
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
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu_activation(x):
    """ReLU activation function"""
    return max(0, x)

def comparator_unit(predicted, actual):
    """Error calculation unit"""
    error = actual - predicted
    return error, error**2

class PerceptronXOR:
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
    print("=" * 80)
    print("XOR GATE PERCEPTRON ANALYSIS")
    print("=" * 80)
    print("\nNote: XOR is not linearly separable, so single perceptron cannot learn it perfectly.")
    print("This demonstrates the limitations of single-layer perceptrons.")

    # XOR gate training data
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]  # XOR truth table

    # For bipolar step, we need different target values
    y_bipolar = [-1, 1, 1, -1]

    activation_functions = ['step', 'bipolar_step', 'sigmoid', 'relu']
    results = {}

    print("\nTraining perceptrons with different activation functions on XOR:")
    print("-" * 70)

    for activation in activation_functions:
        print(f"\nTesting {activation.upper()}...")

        # Use bipolar targets for bipolar step function
        targets = y_bipolar if activation == 'bipolar_step' else y

        # Initialize perceptron
        perceptron = PerceptronXOR(
            weights=[0.2, -0.75], 
            bias=10, 
            learning_rate=0.05,
            activation_func=activation
        )

        # Train
        epochs = perceptron.train(X, targets, max_epochs=1000)
        results[activation] = {
            'epochs': epochs,
            'converged': epochs < 1000,
            'final_weights': perceptron.weights.copy(),
            'final_bias': perceptron.bias,
            'errors': perceptron.errors.copy()
        }

        # Test the trained perceptron
        print(f"Testing {activation.upper()} results:")
        correct_predictions = 0
        for i, inputs in enumerate(X):
            prediction = perceptron.predict(inputs)
            expected = targets[i]
            is_correct = abs(prediction - expected) < 0.5  # Allow some tolerance for continuous outputs
            if is_correct:
                correct_predictions += 1
            status = "✓" if is_correct else "✗"
            print(f"  Input: {inputs}, Expected: {expected:>4}, Predicted: {prediction:>8.4f} {status}")

        accuracy = (correct_predictions / len(X)) * 100
        print(f"  Accuracy: {accuracy:.1f}%")

    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Error curves comparison
    ax1 = plt.subplot(2, 3, 1)
    colors = ['blue', 'orange', 'green', 'red']
    for i, activation in enumerate(activation_functions):
        errors = results[activation]['errors'][:100]  # Show first 100 epochs
        plt.plot(range(1, len(errors) + 1), errors, 
                color=colors[i], label=activation.replace('_', ' ').title(), linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Sum Squared Error')
    plt.title('XOR Gate: Error vs Epochs')
    plt.legend()
    plt.grid(True)

    # Plot 2: Convergence comparison
    ax2 = plt.subplot(2, 3, 2)
    activations = [act.replace('_', ' ').title() for act in activation_functions]
    epochs_to_converge = [results[act]['epochs'] for act in activation_functions]
    converged_status = [results[act]['converged'] for act in activation_functions]
    colors_bar = ['green' if conv else 'red' for conv in converged_status]

    bars = plt.bar(activations, epochs_to_converge, color=colors_bar, alpha=0.7)
    plt.xlabel('Activation Function')
    plt.ylabel('Epochs (Max = 1000)')
    plt.title('XOR Gate: Convergence Analysis')
    plt.xticks(rotation=45)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='Converged'),
                      Patch(facecolor='red', alpha=0.7, label='Did not converge')]
    plt.legend(handles=legend_elements)

    # Plot 3: Decision boundary visualization for step function
    ax3 = plt.subplot(2, 3, 3)
    # Create mesh for decision boundary
    h = 0.1
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Get the step function perceptron
    step_perceptron = PerceptronXOR(
        weights=results['step']['final_weights'], 
        bias=results['step']['final_bias'],
        activation_func='step'
    )

    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = []
    for point in mesh_points:
        Z.append(step_perceptron.predict(point))
    Z = np.array(Z).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)

    # Plot XOR data points
    colors_points = ['red', 'blue', 'blue', 'red']  # XOR pattern
    for i, (inputs, target) in enumerate(zip(X, y)):
        plt.scatter(inputs[0], inputs[1], c=colors_points[i], s=200, 
                   marker='o', edgecolors='black', linewidth=2)
        plt.annotate(f'({inputs[0]},{inputs[1]})→{target}', 
                    (inputs[0], inputs[1]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10)

    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('XOR Gate: Decision Boundary (Step)')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Plot 4-6: Individual error curves
    for i, activation in enumerate(['step', 'sigmoid', 'relu']):
        ax = plt.subplot(2, 3, 4 + i)
        errors = results[activation]['errors']
        plt.plot(range(1, len(errors) + 1), errors, 
                color=colors[activation_functions.index(activation)], linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Sum Squared Error')
        plt.title(f'{activation.replace("_", " ").title()} Activation')
        plt.grid(True)

        # Add text box with final accuracy
        final_error = errors[-1] if errors else 0
        plt.text(0.7, 0.9, f'Final Error: {final_error:.4f}', 
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('A5_xor_gate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Analysis summary
    print("\n" + "=" * 80)
    print("XOR GATE ANALYSIS SUMMARY")
    print("=" * 80)
    print("\nWhy XOR cannot be learned by single perceptron:")
    print("- XOR is not linearly separable")
    print("- No single line can separate the two classes")
    print("- Single perceptron can only learn linearly separable functions")
    print("\nExpected behavior:")
    print("- All activation functions should fail to converge to perfect solution")
    print("- Error should plateau at a non-zero value")
    print("- This demonstrates the need for multi-layer networks")

    print(f"\n{'Activation':<15} {'Epochs':<8} {'Converged':<10} {'Final Error':<12}")
    print("-" * 50)
    for activation in activation_functions:
        result = results[activation]
        converged = "Yes" if result['converged'] else "No"
        final_error = result['errors'][-1] if result['errors'] else 0
        print(f"{activation.replace('_', ' ').title():<15} {result['epochs']:<8} "
              f"{converged:<10} {final_error:<12.6f}")

if __name__ == "__main__":
    main()
