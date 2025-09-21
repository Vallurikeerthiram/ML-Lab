# A9: Backpropagation for XOR Gate
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    s = sigmoid(x)
    return s * (1 - s)

class BackpropagationXOR:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.05):
        """Initialize neural network with backpropagation for XOR"""
        # Initialize weights with Xavier initialization
        self.W1 = np.random.normal(0, np.sqrt(2.0/input_size), (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.normal(0, np.sqrt(2.0/hidden_size), (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

        self.learning_rate = learning_rate
        self.errors = []
        self.epoch_count = 0

    def forward_pass(self, X):
        """Forward propagation"""
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)

        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    def backward_pass(self, X, y, output):
        """Backward propagation"""
        m = X.shape[0]

        # Calculate output layer gradients
        dZ2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        # Calculate hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs=1000, convergence_error=0.002):
        """Train the neural network"""
        for epoch in range(epochs):
            # Forward pass
            output = self.forward_pass(X)

            # Calculate error (MSE)
            error = np.mean((output - y)**2)
            self.errors.append(error)

            # Backward pass
            self.backward_pass(X, y, output)

            self.epoch_count = epoch + 1

            # Check convergence
            if error <= convergence_error:
                print(f"Converged at epoch {epoch + 1} with error {error:.6f}")
                break

        if error > convergence_error:
            print(f"Did not converge after {epochs} epochs. Final error: {error:.6f}")

    def predict(self, X):
        """Make predictions"""
        return self.forward_pass(X)

def main():
    print("=" * 80)
    print("A9: BACKPROPAGATION FOR XOR GATE")
    print("=" * 80)
    print("\nNote: XOR requires hidden layers as it's not linearly separable.")
    print("Single perceptron cannot learn XOR, but multi-layer network can.")

    # XOR gate training data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)  # XOR truth table

    print("\nTraining Data (XOR Gate):")
    print("-" * 30)
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i][0]}")

    # Test different network architectures and learning rates
    experiments = [
        {'hidden_size': 2, 'lr': 0.05, 'name': 'Small Net, LR=0.05'},
        {'hidden_size': 4, 'lr': 0.05, 'name': 'Medium Net, LR=0.05'},
        {'hidden_size': 8, 'lr': 0.05, 'name': 'Large Net, LR=0.05'},
        {'hidden_size': 4, 'lr': 0.1, 'name': 'Medium Net, LR=0.1'},
        {'hidden_size': 4, 'lr': 0.5, 'name': 'Medium Net, LR=0.5'},
    ]

    results = {}

    print(f"\nTesting different configurations for XOR:")
    print("-" * 60)

    for exp in experiments:
        print(f"\nTraining {exp['name']}...")

        # Initialize network
        nn = BackpropagationXOR(
            input_size=2, 
            hidden_size=exp['hidden_size'], 
            output_size=1, 
            learning_rate=exp['lr']
        )

        print(f"Architecture: 2 → {exp['hidden_size']} → 1")
        print(f"Learning rate: {exp['lr']}")

        # Train
        nn.train(X, y, epochs=5000, convergence_error=0.01)  # More epochs for XOR

        # Test
        predictions = nn.predict(X)

        print(f"\nTest Results for {exp['name']}:")
        print("-" * 50)
        correct = 0
        for i in range(len(X)):
            pred_binary = 1 if predictions[i][0] >= 0.5 else 0
            actual = int(y[i][0])
            is_correct = pred_binary == actual
            if is_correct:
                correct += 1
            status = "✓" if is_correct else "✗"
            print(f"Input: {X[i]}, Target: {actual}, Predicted: {predictions[i][0]:.4f} ({pred_binary}) {status}")

        # Calculate accuracy
        accuracy = (correct / len(X)) * 100
        print(f"Accuracy: {accuracy:.1f}%")

        results[exp['name']] = {
            'network': nn,
            'accuracy': accuracy,
            'epochs': nn.epoch_count,
            'final_error': nn.errors[-1] if nn.errors else 1.0,
            'predictions': predictions,
            'hidden_size': exp['hidden_size'],
            'learning_rate': exp['lr'],
            'converged': nn.epoch_count < 5000
        }

    # Final analysis
    print("\n" + "=" * 80)
    print("XOR BACKPROPAGATION ANALYSIS SUMMARY")
    print("=" * 80)

    print("\nPerformance Results:")
    print("-" * 60)
    print(f"{'Configuration':<25} {'Hidden':<8} {'LR':<6} {'Epochs':<8} {'Accuracy':<10} {'Converged'}")
    print("-" * 70)

    for name, result in results.items():
        converged_str = "Yes" if result['converged'] else "No"
        config_name = name.split(',')[0]
        print(f"{config_name:<25} {result['hidden_size']:<8} {result['learning_rate']:<6} "
              f"{result['epochs']:<8} {result['accuracy']:<10.1f} {converged_str}")

    # Find successful configurations
    successful = [name for name, result in results.items() if result['accuracy'] == 100]

    if successful:
        print(f"\nSuccessful Configurations (100% accuracy):")
        print("-" * 50)
        for name in successful:
            result = results[name]
            print(f"  • {name}")
            print(f"    Hidden neurons: {result['hidden_size']}, LR: {result['learning_rate']}")
            print(f"    Converged in {result['epochs']} epochs")
    else:
        print("\nNo configuration achieved 100% accuracy.")
        best = max(results.keys(), key=lambda k: results[k]['accuracy'])
        print(f"Best result: {best} with {results[best]['accuracy']:.1f}% accuracy")

if __name__ == "__main__":
    main()
