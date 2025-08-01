import numpy as np
import matplotlib.pyplot as plt

# Generate 20 random points with 2 features (X and Y) between 1 and 10
def generate_data(num_points=20, low=1, high=10):
    x = np.random.uniform(low, high, num_points)
    y = np.random.uniform(low, high, num_points)
    data = np.column_stack((x, y))
    
    # Assign class based on a simple rule: if x + y > threshold, it's class 1 (Red), else class 0 (Blue)
    threshold = (low + high)
    labels = np.where(x + y > threshold, 1, 0)
    
    return data, labels

# Plotting function
def plot_data(data, labels):
    for i in range(len(data)):
        if labels[i] == 0:
            plt.scatter(data[i][0], data[i][1], color='blue', label='Class 0' if i == 0 else "")
        else:
            plt.scatter(data[i][0], data[i][1], color='red', label='Class 1' if i == 0 else "")
    
    plt.xlabel("X Feature")
    plt.ylabel("Y Feature")
    plt.title("Scatter Plot of Training Data")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run the code
if __name__ == "__main__":
    np.random.seed(42)  # Optional, for consistent results
    points, classes = generate_data()
    plot_data(points, classes)
