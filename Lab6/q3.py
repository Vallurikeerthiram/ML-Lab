# q3.py - Information Gain (A3)
"""
A3. Information Gain and Root Node Detection Module
This module provides functions to calculate information gain and identify the best
feature for splitting in decision tree construction.
"""

import pandas as pd
import numpy as np
from collections import Counter
from q1 import calculate_entropy, equal_width_binning


def calculate_information_gain(data, feature_col, target_col, num_bins=4):
    """
    Calculate information gain for a feature
    """
    # Get target values
    target_values = data[target_col].tolist()
    feature_values = data[feature_col].tolist()
    
    # Check if feature needs binning (continuous with many unique values)
    if data[feature_col].dtype in ['float64','int64']:
        feature_values= equal_width_binning(feature_values,num_bins)
    
    parent_entropy = calculate_entropy(target_values)

    # Group by feature
    feature_groups = {}
    for i, fval in enumerate(feature_values):
        feature_groups.setdefault(fval, []).append(target_values[i])

    # Weighted entropy of splits
    total_samples = len(target_values)
    weighted_entropy = 0
    for subset in feature_groups.values():
        weight = len(subset) / total_samples
        weighted_entropy += weight * calculate_entropy(subset)

    # Info gain
    return parent_entropy - weighted_entropy

def find_best_root_feature(data, target_col, feature_cols, num_bins=4):
    """
    Find the feature with max info gain
    """
    best_feature = None
    best_gain = -1
    gains = {}

    for feature in feature_cols:
        gain = calculate_information_gain(data, feature, target_col, num_bins)
        gains[feature] = gain
        if gain > best_gain:
            best_gain = gain
            best_feature = feature

    return best_feature, best_gain, gains

if __name__ == "__main__":
    df = pd.read_excel("rajasthan.xlsx")

    target_col = "JAN_R/F_2018"   # 🔹 choose one target column
    exclude_cols = ["State", "District", target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]


    best_feature, best_gain, all_gains = find_best_root_feature(df, target_col, feature_cols)

    print("Information Gain values for all features:")
    for f, g in all_gains.items():
        print(f"{f:15} : {g:.4f}")

    print(f"\nBest root feature: {best_feature} (Gain = {best_gain:.4f})")
