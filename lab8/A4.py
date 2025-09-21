# A4: Varying Learning Rate Analysis
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

class PerceptronLearningRate:
    def __init__(self, weights, bias, learning_rate=0.05):
        self.initial_weights = np.array(weights, dtype=float)
        self.initial_bias = float(bias)
        self.learning_rate = learning_rate
        self.reset()

    def reset(self):
        self.weights = self.initial_weights.copy()
        self.bias = self.initial_bias
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
                break

        return self.epoch_count

def main():
    # AND gate training data
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 0, 0, 1]

    # Learning rates to test
    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = {}

    print("Testing different learning rates:")
    print("-" * 50)

    for lr in learning_rates:
        # Initialize perceptron with current learning rate
        perceptron = PerceptronLearningRate(
            weights=[0.2, -0.75], 
            bias=10, 
            learning_rate=lr
        )

        # Train
        epochs = perceptron.train(X, y)
        results[lr] = {
            'epochs': epochs,
            'converged': epochs < 1000,
            'final_weights': perceptron.weights.copy(),
            'final_bias': perceptron.bias,
            'errors': perceptron.errors.copy()
        }

        status = "Converged" if results[lr]['converged'] else "Did not converge"
        print(f"Learning Rate {lr:.1f}: {epochs} epochs - {status}")

    # Create comprehensive plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Learning rate vs epochs to convergence
    converged_lrs = [lr for lr in learning_rates if results[lr]['converged']]
    converged_epochs = [results[lr]['epochs'] for lr in converged_lrs]

    ax1.plot(converged_lrs, converged_epochs, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Epochs to Convergence')
    ax1.set_title('Learning Rate vs Convergence Speed')
    ax1.grid(True)
    ax1.set_xticks(learning_rates)

    # Plot 2: Error curves for selected learning rates
    selected_lrs = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for i, lr in enumerate(selected_lrs):
        if lr in results:
            errors = results[lr]['errors'][:50]  # Show first 50 epochs for clarity
            ax2.plot(range(1, len(errors) + 1), errors, 
                    color=colors[i], label=f'LR = {lr}', linewidth=2)

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Sum Squared Error')
    ax2.set_title('Error Curves for Different Learning Rates')
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Bar chart of convergence
    all_epochs = [results[lr]['epochs'] if results[lr]['converged'] else 1000 for lr in learning_rates]
    colors_bar = ['green' if results[lr]['converged'] else 'red' for lr in learning_rates]

    ax3.bar([str(lr) for lr in learning_rates], all_epochs, color=colors_bar, alpha=0.7)
    ax3.set_xlabel('Learning Rate')
    ax3.set_ylabel('Epochs (Max = 1000)')
    ax3.set_title('Convergence Status by Learning Rate')
    ax3.set_ylim(0, 1100)

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='Converged'),
                      Patch(facecolor='red', alpha=0.7, label='Did not converge')]
    ax3.legend(handles=legend_elements)

    # Plot 4: Final weights analysis
    final_w1 = [results[lr]['final_weights'][0] for lr in learning_rates]
    final_w2 = [results[lr]['final_weights'][1] for lr in learning_rates]

    ax4.plot(learning_rates, final_w1, 'bo-', label='Weight 1', linewidth=2, markersize=6)
    ax4.plot(learning_rates, final_w2, 'ro-', label='Weight 2', linewidth=2, markersize=6)
    ax4.set_xlabel('Learning Rate')
    ax4.set_ylabel('Final Weight Values')
    ax4.set_title('Final Weights vs Learning Rate')
    ax4.legend()
    ax4.grid(True)
    ax4.set_xticks(learning_rates)

    plt.tight_layout()
    plt.savefig('A4_learning_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Summary table
    print("\n" + "=" * 90)
    print("LEARNING RATE ANALYSIS SUMMARY")
    print("=" * 90)
    print(f"{'LR':<5} {'Epochs':<8} {'Converged':<10} {'Final W1':<12} {'Final W2':<12} {'Final Bias':<12}")
    print("-" * 90)
    for lr in learning_rates:
        result = results[lr]
        converged = "Yes" if result['converged'] else "No"
        print(f"{lr:<5} {result['epochs']:<8} {converged:<10} "
              f"{result['final_weights'][0]:<12.3f} {result['final_weights'][1]:<12.3f} "
              f"{result['final_bias']:<12.3f}")

    # Find optimal learning rate
    optimal_lr = min(converged_lrs, key=lambda x: results[x]['epochs'])
    print(f"\nOptimal Learning Rate: {optimal_lr} (converged in {results[optimal_lr]['epochs']} epochs)")

if __name__ == "__main__":
    main()
