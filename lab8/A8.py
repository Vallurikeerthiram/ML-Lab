# A8: Neural Network with Backpropagation for AND Gate
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    s = sigmoid(x)
    return s * (1 - s)

class BackpropagationNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.05):
        """Initialize neural network with backpropagation"""
        # Initialize weights randomly
        self.W1 = np.random.normal(0, 0.5, (input_size, hidden_size))
        self.b1 = np.random.normal(0, 0.5, (1, hidden_size))
        self.W2 = np.random.normal(0, 0.5, (hidden_size, output_size))
        self.b2 = np.random.normal(0, 0.5, (1, output_size))

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

        # Calculate output layer error
        dZ2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        # Calculate hidden layer error
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # Update weights
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
    print("A8: NEURAL NETWORK WITH BACKPROPAGATION FOR AND GATE")
    print("=" * 80)

    # AND gate training data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [0], [0], [1]], dtype=float)

    print("Training Data (AND Gate):")
    print("-" * 30)
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i][0]}")

    # Test different network architectures
    architectures = [
        {'hidden_size': 2, 'name': 'Minimal (2 hidden)'},
        {'hidden_size': 4, 'name': 'Small (4 hidden)'},
        {'hidden_size': 8, 'name': 'Medium (8 hidden)'}
    ]

    results = {}

    print(f"\nTesting different network architectures:")
    print("-" * 50)

    for arch in architectures:
        print(f"\nTraining {arch['name']} network...")

        # Initialize network
        nn = BackpropagationNN(
            input_size=2, 
            hidden_size=arch['hidden_size'], 
            output_size=1, 
            learning_rate=0.05
        )

        print(f"Initial weights W1:\n{nn.W1}")
        print(f"Initial weights W2:\n{nn.W2}")
        print(f"Initial bias b1: {nn.b1}")
        print(f"Initial bias b2: {nn.b2}")

        # Train
        nn.train(X, y, epochs=1000, convergence_error=0.002)

        # Test
        predictions = nn.predict(X)

        print(f"\nFinal weights W1:\n{nn.W1}")
        print(f"Final weights W2:\n{nn.W2}")
        print(f"Final bias b1: {nn.b1}")
        print(f"Final bias b2: {nn.b2}")

        print(f"\nTest Results for {arch['name']}:")
        print("-" * 40)
        for i in range(len(X)):
            pred_binary = 1 if predictions[i][0] >= 0.5 else 0
            print(f"Input: {X[i]}, Target: {y[i][0]}, Predicted: {predictions[i][0]:.4f} ({pred_binary})")

        # Calculate accuracy
        pred_binary = (predictions >= 0.5).astype(int)
        accuracy = np.mean(pred_binary == y) * 100
        print(f"Accuracy: {accuracy:.1f}%")

        results[arch['name']] = {
            'network': nn,
            'accuracy': accuracy,
            'epochs': nn.epoch_count,
            'final_error': nn.errors[-1],
            'predictions': predictions
        }

    # Final analysis
    print("\n" + "=" * 80)
    print("BACKPROPAGATION ANALYSIS SUMMARY")
    print("=" * 80)

    print("\nPerformance Comparison:")
    print("-" * 40)
    print(f"{'Architecture':<20} {'Accuracy':<10} {'Epochs':<8} {'Final Error':<12}")
    print("-" * 50)
    for name, result in results.items():
        arch_name = name.split('(')[0].strip()
        print(f"{arch_name:<20} {result['accuracy']:<10.1f} {result['epochs']:<8} {result['final_error']:<12.6f}")

    best_network_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    print(f"\nBest performing network: {best_network_name}")
    print(f"  Accuracy: {results[best_network_name]['accuracy']:.1f}%")
    print(f"  Convergence: {results[best_network_name]['epochs']} epochs")

if __name__ == "__main__":
    main()
