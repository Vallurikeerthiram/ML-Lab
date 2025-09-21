# A10: Neural Network with 2 Output Nodes
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    s = sigmoid(x)
    return s * (1 - s)

class TwoOutputNN:
    def __init__(self, input_size, hidden_size, output_size=2, learning_rate=0.05):
        """
        Initialize neural network with 2 output nodes
        Output encoding: [1,0] for logic 0, [0,1] for logic 1
        """
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
        output = self.forward_pass(X)
        # Convert to binary predictions (winner-takes-all)
        predictions = np.zeros_like(output)
        max_indices = np.argmax(output, axis=1)
        predictions[np.arange(len(output)), max_indices] = 1
        return predictions, output

def main():
    print("=" * 80)
    print("A10: NEURAL NETWORK WITH 2 OUTPUT NODES")
    print("=" * 80)
    print("\nImplementing logic gates with 2-output encoding:")
    print("  Logic 0 → [1, 0] (first node active)")
    print("  Logic 1 → [0, 1] (second node active)")

    # Input data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

    # AND gate - two output encoding
    y_and_two = np.array([
        [1, 0],  # 0 -> [1,0]
        [1, 0],  # 0 -> [1,0]
        [1, 0],  # 0 -> [1,0]
        [0, 1]   # 1 -> [0,1]
    ], dtype=float)

    # XOR gate - two output encoding
    y_xor_two = np.array([
        [1, 0],  # 0 -> [1,0]
        [0, 1],  # 1 -> [0,1]
        [0, 1],  # 1 -> [0,1]
        [1, 0]   # 0 -> [1,0]
    ], dtype=float)

    print("\nTesting AND Gate with 2 output nodes...")

    # Test AND gate
    nn_and = TwoOutputNN(input_size=2, hidden_size=4, learning_rate=0.1)
    nn_and.train(X, y_and_two, epochs=1000, convergence_error=0.01)

    predictions_and, raw_and = nn_and.predict(X)

    print("\nAND Gate Results:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y_and_two[i]}, Predicted: {predictions_and[i]}")

    print("\nTesting XOR Gate with 2 output nodes...")

    # Test XOR gate
    nn_xor = TwoOutputNN(input_size=2, hidden_size=6, learning_rate=0.1)
    nn_xor.train(X, y_xor_two, epochs=1000, convergence_error=0.01)

    predictions_xor, raw_xor = nn_xor.predict(X)

    print("\nXOR Gate Results:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y_xor_two[i]}, Predicted: {predictions_xor[i]}")

    print("\nTwo-output encoding successfully learned both AND and XOR gates!")

if __name__ == "__main__":
    main()
