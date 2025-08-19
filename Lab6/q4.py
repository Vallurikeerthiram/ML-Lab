# q4.py - Information Gain with Binning Options (A4)
"""
A4. Information Gain and Root Node Detection Module with Binning
This module extends A3 by adding:
- Equal-width or equal-frequency binning
- Method and bin count as parameters
- Function overloading with default parameters
"""

import pandas as pd
import numpy as np
from collections import Counter
from q1 import calculate_entropy


# ---------------------- BINNING FUNCTIONS ---------------------- #

def equal_width_binning(values, num_bins=4):
    """
    Perform equal-width binning on numeric values
    """
    values = pd.Series(values)
    
    # If all values are same, just return one bin label
    if values.nunique() == 1:
        return ["bin_0"] * len(values)
    
    bins = num_bins
    labels = [f"bin_{i}" for i in range(bins)]
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True, duplicates="drop").astype(str).tolist()



def equal_frequency_binning(values, num_bins=4):
    """
    Perform equal-frequency (quantile) binning on numeric values
    """
    values = pd.Series(values)
    
    # If all values are same, just return one bin label
    if values.nunique() == 1:
        return ["bin_0"] * len(values)
    
    try:
        labels = [f"bin_{i}" for i in range(num_bins)]
        return pd.qcut(values, q=num_bins, labels=labels, duplicates="drop").astype(str).tolist()
    except ValueError:
        # Fallback if qcut fails due to not enough unique values
        return ["bin_0"] * len(values)


# ---------------------- INFO GAIN FUNCTION ---------------------- #

def calculate_information_gain(data, feature_col, target_col, num_bins=4, method="equal_width"):
    """
    Calculate information gain for a feature with binning.
    method: "equal_width" or "equal_frequency"
    """
    target_values = data[target_col].tolist()
    feature_values = data[feature_col].tolist()

    # Apply binning only if feature is numeric
    if data[feature_col].dtype in ['float64', 'int64']:
        if method == "equal_width":
            feature_values = equal_width_binning(feature_values, num_bins)
        elif method == "equal_frequency":
            feature_values = equal_frequency_binning(feature_values, num_bins)
        else:
            raise ValueError("Invalid binning method! Use 'equal_width' or 'equal_frequency'.")

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

    return parent_entropy - weighted_entropy


# ---------------------- OVERLOADED FUNCTION ---------------------- #

def calculate_information_gain_overloaded(data, feature_col, target_col, num_bins=None, method=None):
    """Overloaded version with default params if not given"""
    if num_bins is None:
        num_bins = 4
    if method is None:
        method = "equal_width"
    return calculate_information_gain(data, feature_col, target_col, num_bins, method)


# ---------------------- ROOT FEATURE ---------------------- #

def find_best_root_feature(data, target_col, feature_cols, num_bins=4, method="equal_width"):
    best_feature = None
    best_gain = -1
    gains = {}

    for feature in feature_cols:
        gain = calculate_information_gain(data, feature, target_col, num_bins, method)
        gains[feature] = gain
        if gain > best_gain:
            best_gain = gain
            best_feature = feature

    return best_feature, best_gain, gains


# ---------------------- MAIN ---------------------- #

if __name__ == "__main__":
    df = pd.read_excel("rajasthan.xlsx")

    target_col = "JAN_R/F_2018"   # 🔹 choose one target column
    exclude_cols = ["State", "District", target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Example 1: Default (equal-width, 4 bins)
    print("\n=== Equal-Width Binning (default) ===")
    best_feature, best_gain, all_gains = find_best_root_feature(df, target_col, feature_cols)
    for f, g in all_gains.items():
        print(f"{f:15} : {g:.4f}")
    print(f"\nBest root feature: {best_feature} (Gain = {best_gain:.4f})")

    # Example 2: Equal-Frequency Binning (6 bins)
    print("\n=== Equal-Frequency Binning (6 bins) ===")
    best_feature, best_gain, all_gains = find_best_root_feature(df, target_col, feature_cols, num_bins=6, method="equal_frequency")
    for f, g in all_gains.items():
        print(f"{f:15} : {g:.4f}")
    print(f"\nBest root feature: {best_feature} (Gain = {best_gain:.4f})")
