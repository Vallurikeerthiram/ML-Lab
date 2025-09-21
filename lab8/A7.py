# A7: Pseudo-inverse Matrix Solution Comparison
import numpy as np
import matplotlib.pyplot as plt

def sigmoid_activation(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class PseudoInverseClassifier:
    def __init__(self):
        self.weights = None
        self.X_pinv = None

    def fit(self, X, y):
        """Fit using pseudo-inverse (Moore-Penrose inverse)"""
        # Add bias column to X
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])

        # Calculate pseudo-inverse
        self.X_pinv = np.linalg.pinv(X_with_bias)

        # Calculate weights (including bias)
        weights_with_bias = np.dot(self.X_pinv, y)

        # Separate bias and weights
        self.bias = weights_with_bias[0]
        self.weights = weights_with_bias[1:]

    def predict(self, X):
        """Predict using linear combination (without activation)"""
        return np.dot(X, self.weights) + self.bias

    def predict_sigmoid(self, X):
        """Predict using sigmoid activation"""
        linear_output = self.predict(X)
        return sigmoid_activation(linear_output)

# Customer Perceptron class
class CustomerPerceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.learning_rate = learning_rate
        self.errors = []
        self.epoch_count = 0

    def predict(self, inputs):
        net_input = np.dot(inputs, self.weights) + self.bias
        return sigmoid_activation(net_input)

    def train(self, X, y, max_epochs=1000, convergence_error=0.01):
        X = np.array(X)
        y = np.array(y)

        for epoch in range(max_epochs):
            epoch_error = 0

            for i in range(len(X)):
                net_input = np.dot(X[i], self.weights) + self.bias
                predicted = sigmoid_activation(net_input)

                error = y[i] - predicted
                epoch_error += error**2

                # Gradient calculation
                sigmoid_output = sigmoid_activation(net_input)
                gradient = error * sigmoid_output * (1 - sigmoid_output)

                self.weights += self.learning_rate * gradient * X[i]
                self.bias += self.learning_rate * gradient

            mse = epoch_error / len(X)
            self.errors.append(mse)
            self.epoch_count = epoch + 1

            if mse <= convergence_error:
                break

def calculate_metrics(y_true, y_pred):
    """Calculate accuracy, precision, recall, F1-score"""
    y_pred_binary = (y_pred >= 0.5).astype(int)

    accuracy = np.mean(y_true == y_pred_binary) * 100

    # For binary classification
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    tn = np.sum((y_true == 0) & (y_pred_binary == 0))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score

def main():
    print("=" * 90)
    print("PERCEPTRON vs PSEUDO-INVERSE MATRIX COMPARISON")
    print("=" * 90)

    # Customer data
    customers = np.array([
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
    ], dtype=float)

    labels = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])

    feature_names = ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']

    # Normalize features
    X_mean = np.mean(customers, axis=0)
    X_std = np.std(customers, axis=0)
    X_normalized = (customers - X_mean) / X_std

    print("\nTraining both models...")
    print("-" * 50)

    # 1. Train Perceptron
    print("1. Training Perceptron with Gradient Descent...")
    perceptron = CustomerPerceptron(input_size=4, learning_rate=0.1)
    perceptron.train(X_normalized, labels, max_epochs=1000, convergence_error=0.01)

    perceptron_predictions = np.array([perceptron.predict(x) for x in X_normalized])
    perceptron_metrics = calculate_metrics(labels, perceptron_predictions)

    print(f"   Epochs to converge: {perceptron.epoch_count}")
    print(f"   Final weights: {perceptron.weights}")
    print(f"   Final bias: {perceptron.bias:.4f}")

    # 2. Train Pseudo-inverse
    print("\n2. Training Pseudo-inverse Matrix Solution...")
    pseudo_inv = PseudoInverseClassifier()
    pseudo_inv.fit(X_normalized, labels)

    pseudo_predictions_linear = pseudo_inv.predict(X_normalized)
    pseudo_predictions_sigmoid = pseudo_inv.predict_sigmoid(X_normalized)
    pseudo_metrics_linear = calculate_metrics(labels, pseudo_predictions_linear)
    pseudo_metrics_sigmoid = calculate_metrics(labels, pseudo_predictions_sigmoid)

    print(f"   Weights: {pseudo_inv.weights}")
    print(f"   Bias: {pseudo_inv.bias:.4f}")
    print(f"   Solution obtained instantly (no iterations needed)")

    # Detailed comparison
    print("\n" + "=" * 90)
    print("DETAILED COMPARISON")
    print("=" * 90)

    print("\nPrediction Results:")
    print("-" * 80)
    headers = ["Customer", "Actual", "Perceptron", "Pseudo-Inv Linear", "Pseudo-Inv Sigmoid"]
    print(f"{headers[0]:<12}{headers[1]:<8}{headers[2]:<12}{headers[3]:<18}{headers[4]:<18}")
    print("-" * 80)

    for i in range(len(customers)):
        print(f"C_{i+1:<11}{labels[i]:<8}{perceptron_predictions[i]:<12.4f}"
              f"{pseudo_predictions_linear[i]:<18.4f}{pseudo_predictions_sigmoid[i]:<18.4f}")

    # Metrics comparison
    print("\nPerformance Metrics:")
    print("-" * 60)
    metrics_names = ["Accuracy (%)", "Precision", "Recall", "F1-Score"]

    print(f"{'Metric':<15}{'Perceptron':<12}{'Pseudo-Inv (Linear)':<20}{'Pseudo-Inv (Sigmoid)':<20}")
    print("-" * 70)

    for i, metric in enumerate(metrics_names):
        print(f"{metric:<15}{perceptron_metrics[i]:<12.3f}"
              f"{pseudo_metrics_linear[i]:<20.3f}{pseudo_metrics_sigmoid[i]:<20.3f}")

    # Visualizations
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Training convergence (only for perceptron)
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(range(1, len(perceptron.errors) + 1), perceptron.errors, 'b-', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Perceptron Training Convergence')
    plt.grid(True)

    # Plot 2: Weights comparison
    ax2 = plt.subplot(2, 3, 2)
    x = np.arange(len(feature_names))
    width = 0.35

    plt.bar(x - width/2, perceptron.weights, width, label='Perceptron', alpha=0.8)
    plt.bar(x + width/2, pseudo_inv.weights, width, label='Pseudo-inverse', alpha=0.8)

    plt.xlabel('Features')
    plt.ylabel('Weight Values')
    plt.title('Weight Comparison')
    plt.xticks(x, feature_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Predictions comparison
    ax3 = plt.subplot(2, 3, 3)
    plt.scatter(perceptron_predictions, pseudo_predictions_sigmoid, 
               c=labels, cmap='RdBu', s=100, alpha=0.7, edgecolors='black')
    plt.xlabel('Perceptron Predictions')
    plt.ylabel('Pseudo-inverse Predictions (Sigmoid)')
    plt.title('Predictions Correlation')

    # Add diagonal line for perfect correlation
    min_val = min(min(perceptron_predictions), min(pseudo_predictions_sigmoid))
    max_val = max(max(perceptron_predictions), max(pseudo_predictions_sigmoid))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    plt.colorbar(label='Actual Class')
    plt.grid(True, alpha=0.3)

    # Plot 4: Decision boundaries comparison (simplified 2D projection)
    ax4 = plt.subplot(2, 3, 4)

    # Use first two normalized features for visualization
    X_2d = X_normalized[:, :2]

    # Create mesh
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # For visualization, create simplified 2D classifiers
    mesh_points = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.ravel().shape), np.zeros(xx.ravel().shape)]

    Z_perceptron = np.array([perceptron.predict(point) for point in mesh_points])
    Z_perceptron = Z_perceptron.reshape(xx.shape)

    plt.contourf(xx, yy, Z_perceptron, levels=50, alpha=0.3, cmap='RdBu')

    # Plot data points
    colors = ['red' if label == 0 else 'blue' for label in labels]
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=100, edgecolors='black')

    plt.xlabel('Feature 1 (Normalized)')
    plt.ylabel('Feature 2 (Normalized)')
    plt.title('Perceptron Decision Boundary (2D Projection)')

    # Plot 5: Error analysis
    ax5 = plt.subplot(2, 3, 5)

    methods = ['Perceptron', 'Pseudo-Inv\n(Linear)', 'Pseudo-Inv\n(Sigmoid)']
    accuracies = [perceptron_metrics[0], pseudo_metrics_linear[0], pseudo_metrics_sigmoid[0]]
    colors = ['skyblue', 'lightcoral', 'lightgreen']

    bars = plt.bar(methods, accuracies, color=colors, alpha=0.8)
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison')
    plt.ylim(0, 100)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom')

    # Plot 6: Computational complexity
    ax6 = plt.subplot(2, 3, 6)

    methods = ['Perceptron', 'Pseudo-Inverse']
    times = [perceptron.epoch_count, 1]  # Pseudo-inverse is one-shot
    colors = ['orange', 'green']

    bars = plt.bar(methods, times, color=colors, alpha=0.8)
    plt.ylabel('Computational Units\n(Epochs vs Single Calculation)')
    plt.title('Computational Efficiency')
    plt.yscale('log')

    for bar, time in zip(bars, times):
        height = bar.get_height()
        label = f'{time} epochs' if time > 1 else 'Single calculation'
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                label, ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plt.savefig('A7_perceptron_vs_pseudoinverse.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Final analysis
    print("\n" + "=" * 90)
    print("ANALYSIS AND CONCLUSIONS")
    print("=" * 90)

    print("\nKey Differences:")
    print("-" * 20)
    print("1. COMPUTATIONAL APPROACH:")
    print("   • Perceptron: Iterative gradient descent")
    print("   • Pseudo-inverse: Direct matrix solution")

    print("\n2. CONVERGENCE:")
    print(f"   • Perceptron: Required {perceptron.epoch_count} epochs")
    print("   • Pseudo-inverse: Instant solution")

    print("\n3. ACTIVATION FUNCTION:")
    print("   • Perceptron: Built-in sigmoid activation")
    print("   • Pseudo-inverse: Linear (can add sigmoid post-processing)")

    print("\n4. PERFORMANCE:")
    perceptron_acc = perceptron_metrics[0]
    pseudo_sigmoid_acc = pseudo_metrics_sigmoid[0]

    if abs(perceptron_acc - pseudo_sigmoid_acc) < 5:
        print("   • Similar accuracy for both methods")
    elif perceptron_acc > pseudo_sigmoid_acc:
        print("   • Perceptron achieved slightly better accuracy")
    else:
        print("   • Pseudo-inverse achieved slightly better accuracy")

    print("\n5. ADVANTAGES:")
    print("   Perceptron:")
    print("   • Naturally handles non-linear activation")
    print("   • More biologically inspired")
    print("   • Can handle streaming data")
    print("\n   Pseudo-inverse:")
    print("   • Guaranteed global minimum for linear case")
    print("   • No hyperparameter tuning needed")
    print("   • Computationally efficient for small datasets")

    print("\n6. WHEN TO USE:")
    print("   • Perceptron: When you need non-linear activation and can afford iterations")
    print("   • Pseudo-inverse: When you need quick, simple linear solution")

if __name__ == "__main__":
    main()
