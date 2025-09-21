# A6: Customer Transaction Classification
import numpy as np
import matplotlib.pyplot as plt

def sigmoid_activation(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    s = sigmoid_activation(x)
    return s * (1 - s)

def summation_unit(inputs, weights, bias):
    """Calculates weighted sum of inputs plus bias"""
    return np.dot(inputs, weights) + bias

class CustomerPerceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # Initialize weights randomly
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.learning_rate = learning_rate
        self.errors = []
        self.epoch_count = 0

    def predict(self, inputs):
        net_input = summation_unit(inputs, self.weights, self.bias)
        return sigmoid_activation(net_input)

    def train(self, X, y, max_epochs=1000, convergence_error=0.002):
        """Train perceptron using gradient descent"""
        X = np.array(X)
        y = np.array(y)

        for epoch in range(max_epochs):
            epoch_error = 0

            for i in range(len(X)):
                # Forward pass
                net_input = summation_unit(X[i], self.weights, self.bias)
                predicted = sigmoid_activation(net_input)

                # Calculate error
                error = y[i] - predicted
                epoch_error += error**2

                # Backpropagation (gradient descent)
                gradient = error * sigmoid_derivative(net_input)

                # Update weights and bias
                self.weights += self.learning_rate * gradient * X[i]
                self.bias += self.learning_rate * gradient

            # Calculate mean squared error for epoch
            mse = epoch_error / len(X)
            self.errors.append(mse)
            self.epoch_count = epoch + 1

            # Check convergence
            if mse <= convergence_error:
                print(f"Converged at epoch {epoch + 1} with MSE {mse:.6f}")
                break

        if mse > convergence_error:
            print(f"Did not converge after {max_epochs} epochs. Final MSE: {mse:.6f}")

def main():
    print("=" * 80)
    print("CUSTOMER TRANSACTION CLASSIFICATION")
    print("=" * 80)

    # Customer data from the lab
    customers = [
        [20, 6, 2, 386],   # C_1 - Yes
        [16, 3, 6, 289],   # C_2 - Yes  
        [27, 6, 2, 393],   # C_3 - Yes
        [19, 1, 2, 110],   # C_4 - No
        [24, 4, 2, 280],   # C_5 - Yes
        [22, 1, 5, 167],   # C_6 - No
        [15, 4, 2, 271],   # C_7 - Yes
        [18, 4, 2, 274],   # C_8 - Yes
        [21, 1, 4, 148],   # C_9 - No
        [16, 2, 4, 198]    # C_10 - No
    ]

    # Labels: 1 for High Value, 0 for Low Value
    labels = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0]

    # Feature names
    feature_names = ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']

    print("\nCustomer Transaction Data:")
    print("-" * 60)
    print(f"{'Customer':<10} {'Candies':<8} {'Mangoes':<8} {'Milk':<8} {'Payment':<10} {'High Value':<10}")
    print("-" * 60)
    for i, (customer, label) in enumerate(zip(customers, labels)):
        high_value = "Yes" if label else "No"
        print(f"C_{i+1:<9} {customer[0]:<8} {customer[1]:<8} {customer[2]:<8} {customer[3]:<10} {high_value}")

    # Normalize features for better training
    X = np.array(customers, dtype=float)

    # Feature scaling (standardization)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normalized = (X - X_mean) / X_std

    print("\nFeature Statistics (Original Data):")
    print("-" * 50)
    for i, feature in enumerate(feature_names):
        print(f"{feature:<20}: Mean={X_mean[i]:.2f}, Std={X_std[i]:.2f}")

    # Initialize and train perceptron
    perceptron = CustomerPerceptron(input_size=4, learning_rate=0.1)

    print(f"\nInitial weights: {perceptron.weights}")
    print(f"Initial bias: {perceptron.bias:.4f}")
    print(f"Learning rate: {perceptron.learning_rate}")

    print("\nTraining perceptron...")
    perceptron.train(X_normalized, labels, max_epochs=1000, convergence_error=0.01)

    print(f"\nFinal weights: {perceptron.weights}")
    print(f"Final bias: {perceptron.bias:.4f}")
    print(f"Total epochs: {perceptron.epoch_count}")

    # Test the trained perceptron
    print("\nTesting trained perceptron:")
    print("-" * 70)
    print(f"{'Customer':<10} {'Features':<30} {'Expected':<10} {'Predicted':<12} {'Probability':<12} {'Correct'}")
    print("-" * 70)

    correct_predictions = 0
    for i in range(len(customers)):
        prediction_prob = perceptron.predict(X_normalized[i])
        prediction = 1 if prediction_prob >= 0.5 else 0
        is_correct = prediction == labels[i]
        if is_correct:
            correct_predictions += 1

        status = "✓" if is_correct else "✗"
        feature_str = f"{customers[i]}"
        print(f"C_{i+1:<9} {feature_str:<30} {labels[i]:<10} {prediction:<12} {prediction_prob:<12.4f} {status}")

    accuracy = (correct_predictions / len(customers)) * 100
    print(f"\nAccuracy: {accuracy:.1f}%")

    # Visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Training curve
    ax1.plot(range(1, len(perceptron.errors) + 1), perceptron.errors, 'b-', linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('Training Curve - Customer Classification')
    ax1.grid(True)

    # Plot 2: Feature importance (absolute weights)
    feature_importance = np.abs(perceptron.weights)
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    bars = ax2.bar(feature_names, feature_importance, color=colors)
    ax2.set_xlabel('Features')
    ax2.set_ylabel('|Weight| (Importance)')
    ax2.set_title('Feature Importance')
    ax2.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, importance in zip(bars, feature_importance):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{importance:.3f}', ha='center', va='bottom')

    # Plot 3: Prediction vs Actual
    predictions = [perceptron.predict(X_normalized[i]) for i in range(len(customers))]

    # Scatter plot
    high_value_idx = [i for i, label in enumerate(labels) if label == 1]
    low_value_idx = [i for i, label in enumerate(labels) if label == 0]

    ax3.scatter([predictions[i] for i in high_value_idx], [1]*len(high_value_idx), 
               c='green', label='High Value (Actual)', s=100, alpha=0.7)
    ax3.scatter([predictions[i] for i in low_value_idx], [0]*len(low_value_idx), 
               c='red', label='Low Value (Actual)', s=100, alpha=0.7)

    # Add threshold line
    ax3.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Decision Threshold')

    ax3.set_xlabel('Predicted Probability')
    ax3.set_ylabel('Actual Class')
    ax3.set_title('Predicted vs Actual Classifications')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Low Value', 'High Value'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Data distribution by features
    # Show relationship between Payment and High Value transactions
    payments = [customer[3] for customer in customers]
    high_payments = [customers[i][3] for i in high_value_idx]
    low_payments = [customers[i][3] for i in low_value_idx]

    ax4.hist(low_payments, bins=5, alpha=0.7, label='Low Value', color='red', density=True)
    ax4.hist(high_payments, bins=5, alpha=0.7, label='High Value', color='green', density=True)
    ax4.set_xlabel('Payment Amount (Rs)')
    ax4.set_ylabel('Density')
    ax4.set_title('Payment Distribution by Transaction Type')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('A6_customer_classification.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Analysis of results
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)

    print("\nFeature Analysis:")
    print("-" * 30)
    feature_weights = list(zip(feature_names, perceptron.weights))
    feature_weights.sort(key=lambda x: abs(x[1]), reverse=True)

    for i, (feature, weight) in enumerate(feature_weights):
        direction = "positively" if weight > 0 else "negatively"
        print(f"{i+1}. {feature}: {weight:.4f} (contributes {direction})")

    print("\nDecision Logic:")
    print("-" * 20)
    print("The perceptron learned to classify transactions as high-value based on:")
    for feature, weight in feature_weights:
        if abs(weight) > 0.1:  # Only show significant features
            if weight > 0:
                print(f"  • Higher {feature} → Higher probability of high-value transaction")
            else:
                print(f"  • Higher {feature} → Lower probability of high-value transaction")

    # Show some example predictions
    print("\nExample Predictions:")
    print("-" * 25)
    for i in [0, 3, 6, 9]:  # Show a few examples
        prob = perceptron.predict(X_normalized[i])
        pred_class = "High Value" if prob >= 0.5 else "Low Value"
        actual_class = "High Value" if labels[i] == 1 else "Low Value"
        print(f"Customer C_{i+1}: {customers[i]} → {pred_class} ({prob:.3f}) [Actual: {actual_class}]")

if __name__ == "__main__":
    main()
