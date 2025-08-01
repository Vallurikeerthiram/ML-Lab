import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Function to generate training data with 2 classes (20 points total)
def generate_training_data(seed=42):
    np.random.seed(seed)
    feature_x = np.random.uniform(1, 10, 20)
    feature_y = np.random.uniform(1, 10, 20)
    features = np.column_stack((feature_x, feature_y))

    # First 10 are class 0, next 10 are class 1
    labels = np.array([0]*10 + [1]*10)
    return features, labels

# Function to generate dense grid of test points in feature space
def generate_test_data(x_min=0, x_max=10, y_min=0, y_max=10, step=0.1):
    x_range = np.arange(x_min, x_max, step)
    y_range = np.arange(y_min, y_max, step)
    xx, yy = np.meshgrid(x_range, y_range)
    test_points = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, test_points

# Function to train kNN model and predict test point classes
def classify_test_data(train_features, train_labels, test_points, k):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(train_features, train_labels)
    predicted_classes = knn_model.predict(test_points)
    return predicted_classes

# Function to plot decision boundary
def plot_classification_result(xx, yy, predicted_classes, train_features, train_labels, k):
    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, predicted_classes.reshape(xx.shape), alpha=0.3, cmap=plt.cm.bwr)
    plt.scatter(train_features[:, 0], train_features[:, 1], c=train_labels, cmap=plt.cm.bwr, edgecolor='black', s=80)
    plt.title(f"Decision Boundary (k = {k})")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.grid(True)
    plt.show()

# Main program
if __name__ == "__main__":
    train_X, train_y = generate_training_data()
    xx, yy, test_points = generate_test_data()

    # Try for multiple k values
    k_values = [1, 3, 5, 7, 11]
    for k in k_values:
        test_preds = classify_test_data(train_X, train_y, test_points, k)
        plot_classification_result(xx, yy, test_preds, train_X, train_y, k)
