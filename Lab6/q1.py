# q1.py - Entropy Calculator (A1)
"""
A1. Entropy Calculation Module
This module provides functions to calculate entropy for decision tree construction.
"""

import numpy as np
from collections import Counter
from math import log2


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
        if count > 0:
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
    if len(values) == 0:
        return []
    
    min_val = min(values)
    max_val = max(values)
    
    # Handle case where all values are the same
    if min_val == max_val:
        return [f"Bin_0"] * len(values)
    
    bin_width = (max_val - min_val) / num_bins
    
    binned_values = []
    for value in values:
        if value == max_val:
            # Handle edge case where value equals maximum
            bin_number = num_bins - 1
        else:
            bin_number = int((value - min_val) / bin_width)
        binned_values.append(f"Bin_{bin_number}")
    
    return binned_values


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


def validate_entropy_calculation():
    """Validation function to test entropy calculations with known examples."""
    print("Validating Entropy Calculations:")
    print("-" * 40)
    
    # Test 1: Pure dataset (entropy should be 0)
    pure_labels = ['A'] * 10
    pure_entropy = calculate_entropy(pure_labels)
    print(f"Test 1 - Pure dataset: {pure_entropy:.4f} (Expected: 0.0000)")
    
    # Test 2: Balanced binary dataset (entropy should be 1.0)
    balanced_binary = ['A'] * 5 + ['B'] * 5
    balanced_entropy = calculate_entropy(balanced_binary)
    print(f"Test 2 - Balanced binary: {balanced_entropy:.4f} (Expected: 1.0000)")
    
    # Test 3: Imbalanced dataset
    imbalanced = ['A'] * 8 + ['B'] * 2
    imbalanced_entropy = calculate_entropy(imbalanced)
    print(f"Test 3 - Imbalanced (80-20): {imbalanced_entropy:.4f}")
    
    # Test 4: Four-class balanced
    four_class = ['A'] * 5 + ['B'] * 5 + ['C'] * 5 + ['D'] * 5
    four_entropy = calculate_entropy(four_class)
    print(f"Test 4 - Four-class balanced: {four_entropy:.4f} (Expected: 2.0000)")
    
    # Test 5: Empty dataset
    empty_entropy = calculate_entropy([])
    print(f"Test 5 - Empty dataset: {empty_entropy:.4f} (Expected: 0.0000)")
    
    print("\nBinning Tests:")
    print("-" * 40)
    
    # Test binning
    continuous_values = [1.0, 2.5, 4.8, 7.2, 9.9, 12.1, 15.3, 18.7, 21.0, 25.0]
    binned = equal_width_binning(continuous_values, 4)
    bin_counts = Counter(binned)
    
    print(f"Original values: {continuous_values}")
    print(f"Binned labels: {binned}")
    print(f"Bin distribution: {dict(bin_counts)}")
    
    # Calculate entropy of binned data
    bin_entropy = calculate_entropy(binned)
    print(f"Entropy of binned data: {bin_entropy:.4f}")


if __name__ == "__main__":
    validate_entropy_calculation()
