import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Function to generate random training data
def generate_random_classified_data(num_points=20, low=1, high=10):
    x_values = np.random.uniform(low, high, num_points)
    y_values = np.random.uniform(low, high, num_points)
    X_features = np.column_stack((x_values, y_values))
    threshold = (low + high)
    class_labels = np.where(x_values + y_values > threshold, 1, 0)
    return X_features, class_labels

# Function to plot training data
def plot_training_data(X_features, class_labels):
    for i in range(len(X_features)):
        if class_labels[i] == 0:
            plt.scatter(X_features[i][0], X_features[i][1], color='blue', label='Class 0' if i == 0 else "")
        else:
            plt.scatter(X_features[i][0], X_features[i][1], color='red', label='Class 1' if i == 0 else "")
    plt.xlabel("X Feature")
    plt.ylabel("Y Feature")
    plt.title("Training Data")
    plt.legend()
    plt.grid(True)

# Generate training data
np.random.seed(42)
X_train, y_train = generate_random_classified_data()

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Generate test data (grid of 0 to 10 with step 0.1)
x_test_vals = np.arange(0, 10.1, 0.1)
y_test_vals = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(x_test_vals, y_test_vals)
test_points = np.c_[xx.ravel(), yy.ravel()]  # Shape: (10000, 2)

# Predict class labels for test data
test_predictions = knn.predict(test_points)

# Plot predicted test points
plt.figure(figsize=(8, 6))
plt.scatter(test_points[test_predictions == 0][:, 0],
            test_points[test_predictions == 0][:, 1],
            color='blue', alpha=0.2, label='Class 0 (Blue)')

plt.scatter(test_points[test_predictions == 1][:, 0],
            test_points[test_predictions == 1][:, 1],
            color='red', alpha=0.2, label='Class 1 (Red)')

# Optionally, overlay training data on top
plot_training_data(X_train, y_train)

plt.title("kNN Classification of Test Data (k=3)")
plt.legend()
plt.tight_layout()
plt.show()

#test plot shows how the model divides the 2D space into regions belonging to Class 0 (blue) and Class 1 (red).
#The boundary between blue and red regions represents the decision surface where the classifier is unsure (i.e., it could be either class depending on the neighbors).

#Areas close to blue training points are mostly predicted as blue.
#Areas close to red training points are predicted as red.
#if training points are clustered tightly, those regions will have strong class dominance.
#If training points are scattered, the boundary might look noisy or irregular.

#If the class regions look too wiggly or chaotic, it may be a sign of overfitting, meaning the model is too sensitive to the training data.
#If the boundaries are smooth and intuitive, the model is likely generalizing well.

#If you see clearly separated zones, it means the training classes are well-separated in the feature space.
#If there’s blending or overlap, your data may not be easily separable with simple models like kNN.
