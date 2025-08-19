# q7.py - Decision Tree Classification & Decision Boundary (A7)
"""
A7. Train a decision tree classifier using 2 features from your dataset
and visualize both the decision tree and its decision boundary.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

# ---------------------- FUNCTIONS ---------------------- #

def discretize_target(y, n_bins=4, strategy="uniform"):
    """Convert continuous target into discrete bins for classification"""
    y = y.values.reshape(-1, 1)
    kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    y_binned = kb.fit_transform(y).astype(int).ravel()
    return y_binned

def train_decision_tree(X, y, max_depth=3):
    """Train a DecisionTreeClassifier"""
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X, y)
    return clf

def plot_decision_boundary(X, y, clf, feature_names):
    """Plot the decision boundary for a classifier in 2D"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFD700'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#B8860B'])

    plt.figure(figsize=(10,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=50)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Decision Tree Decision Boundary")
    plt.show()

def visualize_tree(clf, feature_names):
    """Plot the trained decision tree"""
    plt.figure(figsize=(12,6))
    plot_tree(clf, feature_names=feature_names, filled=True, rounded=True)
    plt.show()

# ---------------------- MAIN ---------------------- #
if __name__ == "__main__":
    # Load your dataset
    df = pd.read_excel("rajasthan.xlsx")

    # Select 2 features for classification
    feature_cols = ["MAY_R/F_2018", "JAN_%DEP_2018"]
    target_col = "JAN_R/F_2018"

    # Drop rows with NaNs in selected columns
    df = df.dropna(subset=feature_cols + [target_col])

    X = df[feature_cols].values
    y = discretize_target(df[target_col], n_bins=4, strategy="uniform")

    print("Sample features (first 5 rows):\n", X[:5])
    print("Binned target labels (first 5 rows):\n", y[:5])

    # Train Decision Tree
    clf = train_decision_tree(X, y, max_depth=3)
    print("\nDecision Tree trained with max_depth=3")

    # Visualize tree
    visualize_tree(clf, feature_cols)

    # Plot decision boundary
    plot_decision_boundary(X, y, clf, feature_cols)
