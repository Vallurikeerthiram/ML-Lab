# q6.py - Decision Tree Visualization with sklearn (A6)
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# ---------------------- FUNCTIONS ---------------------- #
def load_data(file_path, target_col, exclude_cols):
    """Load dataset and separate features and target"""
    df = pd.read_excel(file_path)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)
    return X, y, feature_cols

def train_decision_tree(X, y, max_depth=3):
    """Train a decision tree regressor"""
    tree = DecisionTreeRegressor(max_depth=max_depth)
    tree.fit(X, y)
    return tree

def visualize_tree(tree, feature_cols):
    """Visualize the trained decision tree"""
    plt.figure(figsize=(20, 10))
    plot_tree(
        tree,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.show()

def predict_sample(tree, sample):
    """Predict for a single sample (dict or pd.Series)"""
    if isinstance(sample, dict):
        sample = pd.DataFrame([sample])
    return tree.predict(sample)[0]

# ---------------------- MAIN ---------------------- #
if __name__ == "__main__":
    file_path = "rajasthan.xlsx"
    target_col = "JAN_R/F_2018"
    exclude_cols = ["State", "District", target_col]

    # Load data
    X, y, feature_cols = load_data(file_path, target_col, exclude_cols)

    # Train tree
    tree = train_decision_tree(X, y, max_depth=3)

    # Visualize tree
    print("\n=== DECISION TREE VISUALIZATION ===")
    visualize_tree(tree, feature_cols)

    # Example prediction: first row
    sample = X.iloc[0].to_dict()
    prediction = predict_sample(tree, sample)
    print("\nPrediction for first row:", prediction)
