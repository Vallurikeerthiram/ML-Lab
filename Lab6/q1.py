# q1.py - Entropy Calculator (A1)
"""
A1. Entropy Calculation Module
This module provides functions to calculate entropy for decision tree construction.
"""

import numpy as np
from collections import Counter
from math import log2
import pandas as pd

def calculate_entropy(labels):
    """
    Calculate entropy for a list of labels.
    
    Entropy = -Σ(pi * log2(pi)) where pi is the probability of each class
    
    Parameters:
    -----------
    labels : list or array-like
        List of class labels
    
    Returns:
    --------
    float
        Entropy value (0 means pure, higher values mean more impure)
    """
    if len(labels) == 0:
        return 0
    
    # Count occurrences of each label
    label_counts = Counter(labels)
    total_count = len(labels)
    
    # Calculate entropy
    entropy = 0
    for count in label_counts.values():
        probability = count / total_count
        entropy -= probability * log2(probability)
    
    return entropy


def equal_width_binning(values, num_bins=4):
    """
    Convert continuous values to categorical using equal width binning.
    
    Parameters:
    -----------
    values : list or array-like
        Continuous numerical values to bin
    num_bins : int, default=4
        Number of bins to create
    
    Returns:
    --------
    list
        List of bin labels for each value
    """
    values = np.array(values)
    if len(values) == 0:
        return []
    
    min_val = np.min(values)
    max_val = np.max(values)
    
    if min_val == max_val:
        return [f"Bin_0"] * len(values)
    
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    bin_indices = np.digitize(values, bin_edges, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    return [f"Bin_{i}" for i in bin_indices]


def entropy_for_continuous_target(target_values, num_bins=4):
    """
    Calculate entropy for continuous target variable using equal width binning.
    
    Parameters:
    -----------
    target_values : list or array-like
        Continuous target values
    num_bins : int, default=4
        Number of bins to create for discretization
    
    Returns:
    --------
    tuple
        (entropy_value, binned_labels)
    """
    binned_labels = equal_width_binning(target_values, num_bins)
    entropy = calculate_entropy(binned_labels)
    return entropy, binned_labels


if __name__ == "__main__":
    # Load dataset in main (not inside functions)
    file_path = r"C:\Users\keert\OneDrive - Amrita vishwa vidyapeetham\Amrita\Sem5\ML\Lab6\rajasthan.xlsx"
    df = pd.read_excel(file_path)

    # Choose a rainfall column
    target_column = "JAN_R/F_2018"
    target_values = df[target_column].dropna().tolist()

    # Call function
    entropy_val, binned_labels = entropy_for_continuous_target(target_values, num_bins=4)

    # Print results (only in main)
    print(f"Entropy for {target_column}: {entropy_val:.4f}")
    print(f"Unique bins distribution: {dict(pd.Series(binned_labels).value_counts())}")
    
