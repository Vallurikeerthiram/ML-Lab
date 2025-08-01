import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Load and clean data
def load_groundwater_data(csv_path, feature1, feature2):
    df = pd.read_csv(csv_path, encoding='latin1')

    # List of invalid entries to remove
    invalid_values = ['Dry', 'Filled up', 'Not Measured', '-', '', ' ']

    # Filter out rows with invalid values in either feature column
    df = df[~df[feature1].isin(invalid_values)]
    df = df[~df[feature2].isin(invalid_values)]

    # Drop missing (NaN) values in those columns
    df = df.dropna(subset=[feature1, feature2])

    # Try converting to float safely
    df[feature1] = pd.to_numeric(df[feature1], errors='coerce')
    df[feature2] = pd.to_numeric(df[feature2], errors='coerce')

    # Drop any rows that couldn't be converted
    df = df.dropna(subset=[feature1, feature2])

    return df[[feature1, feature2]]

# Step 2: Assign binary class labels based on average threshold
def assign_classes(df):
    threshold = df.mean().mean()  # Global average of both features
    y = np.where(df.mean(axis=1) > threshold, 1, 0)
    return df.values, y

# Step 3: Train k-NN model
def train_knn(X, y, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)
    return model

# Step 4: Predict over grid to visualize decision boundaries
def predict_on_grid(model, X, step=0.1):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))

    test_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(test_points).reshape(xx.shape)
    return xx, yy, Z

# Step 5: Plotting decision boundaries
def plot_decision_boundary(X, y, xx, yy, Z, k, feature1, feature2):
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='k')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f"k-NN Decision Boundary (k={k})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------- MAIN DRIVER ----------------------
if __name__ == "__main__":
    # File path and selected features
    csv_path = r"C:\Users\keert\OneDrive - Amrita vishwa vidyapeetham\Amrita\Sem5\ML\Lab4\Atal_Jal_Disclosed_Ground_Water_Level-2015-2022.csv"
    feature1 = "Pre-monsoon_2015 (meters below ground level)"
    feature2 = "Post-monsoon_2022 (meters below ground level)"

    # Load and prepare data
    df = load_groundwater_data(csv_path, feature1, feature2)
    X, y = assign_classes(df)

    # Try different k values to observe boundary change
    for k in [1, 3, 5, 9]:
        model = train_knn(X, y, k)
        xx, yy, Z = predict_on_grid(model, X)
        plot_decision_boundary(X, y, xx, yy, Z, k, feature1, feature2)
