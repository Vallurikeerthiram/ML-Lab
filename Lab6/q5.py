# q5.py - Decision Tree Module (A5)
"""
A5. Decision Tree Construction Module
This module expands A4 by building a full decision tree
using Information Gain and binning options.
"""

import pandas as pd
from q4 import calculate_information_gain, find_best_root_feature

# ---------------------- TREE NODE ---------------------- #
class DecisionNode:
    def __init__(self, feature=None, children=None, label=None):
        """
        Decision Tree Node
        feature : str -> feature used for split
        children : dict -> {feature_value: child_node}
        label : str -> leaf label if pure
        """
        self.feature = feature
        self.children = children if children else {}
        self.label = label

    def is_leaf(self):
        return self.label is not None


# ---------------------- DECISION TREE ---------------------- #
class DecisionTree:
    def __init__(self, target_col, num_bins=4, method="equal_width", max_depth=None):
        self.target_col = target_col
        self.num_bins = num_bins
        self.method = method
        self.max_depth = max_depth
        self.root = None

    def fit(self, data, feature_cols, depth=0):
        """
        Build decision tree recursively
        """
        labels = data[self.target_col].tolist()

        # Case 1: All labels same -> leaf
        if len(set(labels)) == 1:
            return DecisionNode(label=labels[0])

        # Case 2: No features left or reached max depth -> majority vote
        if not feature_cols or (self.max_depth and depth >= self.max_depth):
            majority_label = data[self.target_col].mode()[0]
            return DecisionNode(label=majority_label)

        # Case 3: Choose best feature
        best_feature, best_gain, _ = find_best_root_feature(
            data, self.target_col, feature_cols, self.num_bins, self.method
        )

        if best_gain <= 0:  # No useful split
            majority_label = data[self.target_col].mode()[0]
            return DecisionNode(label=majority_label)

        node = DecisionNode(feature=best_feature)

        # Split dataset on feature
        for value, subset in data.groupby(best_feature):
            child = self.fit(
                subset, [f for f in feature_cols if f != best_feature], depth + 1
            )
            node.children[value] = child

        return node

    def train(self, data, feature_cols):
        """Entry point to train"""
        self.root = self.fit(data, feature_cols)

    def predict_one(self, sample, node=None):
        """Predict label for one sample"""
        if node is None:
            node = self.root

        if node.is_leaf():
            return node.label

        feature_val = sample.get(node.feature)
        child = node.children.get(feature_val)

        if child is None:
            # If unseen value, return majority class
            return node.label if node.is_leaf() else None

        return self.predict_one(sample, child)

    def predict(self, df):
        """Predict for a dataframe"""
        return [self.predict_one(row) for _, row in df.iterrows()]

    def print_tree(self, node=None, indent=""):
        """Pretty print tree"""
        if node is None:
            node = self.root

        if node.is_leaf():
            print(indent + "Leaf:", node.label)
            return

        print(indent + f"[Feature: {node.feature}]")
        for value, child in node.children.items():
            print(indent + f" -> {value}:")
            self.print_tree(child, indent + "   ")


# ---------------------- MAIN ---------------------- #
if __name__ == "__main__":
    df = pd.read_excel("rajasthan.xlsx")

    target_col = "JAN_R/F_2018"
    exclude_cols = ["State", "District", target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    tree = DecisionTree(target_col, num_bins=4, method="equal_width", max_depth=3)
    tree.train(df, feature_cols)

    print("\n=== DECISION TREE ===")
    tree.print_tree()

    # Example prediction
    sample = df.iloc[0].to_dict()
    print("\nPrediction for first row:", tree.predict_one(sample))
