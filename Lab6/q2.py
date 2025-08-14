# q2.py - Gini Calculator (A2)
"""
A2. Gini Index Calculation Module
This module provides functions to calculate Gini impurity for decision tree construction.
"""

import numpy as np
from collections import Counter


def calculate_gini_index(labels):
    """
    Calculate Gini index (Gini impurity) for a list of labels.
    
    Gini = 1 - Σ(pi^2) where pi is the probability of each class
    
    Parameters:
    -----------
    labels : list or array-like
        List of class labels
    
    Returns:
    --------
    float
        Gini index value (0 means pure, higher values mean more impure)
    """
    if len(labels) == 0:
        return 0
    
    # Count occurrences of each label
    label_counts = Counter(labels)
    total_count = len(labels)
    
    # Calculate Gini index
    gini = 1
    for count in label_counts.values():
        probability = count / total_count
        gini -= probability ** 2
    
    return gini


def calculate_weighted_gini(groups):
    """
    Calculate weighted Gini index for multiple groups (used in splitting).
    
    Parameters:
    -----------
    groups : dict
        Dictionary where keys are group names and values are lists of labels
    
    Returns:
    --------
    float
        Weighted Gini index for all groups combined
    """
    total_samples = sum(len(group) for group in groups.values())
    
    if total_samples == 0:
        return 0
    
    weighted_gini = 0
    for group_name, group_labels in groups.items():
        if len(group_labels) > 0:
            group_gini = calculate_gini_index(group_labels)
            weight = len(group_labels) / total_samples
            weighted_gini += weight * group_gini
    
    return weighted_gini


def gini_gain(parent_labels, left_labels, right_labels):
    """
    Calculate the Gini gain from a binary split.
    
    Parameters:
    -----------
    parent_labels : list
        Labels before the split
    left_labels : list
        Labels in the left child after split
    right_labels : list
        Labels in the right child after split
    
    Returns:
    --------
    float
        Gini gain (higher values indicate better splits)
    """
    parent_gini = calculate_gini_index(parent_labels)
    
    total_samples = len(parent_labels)
    if total_samples == 0:
        return 0
    
    # Calculate weighted Gini of children
    left_weight = len(left_labels) / total_samples
    right_weight = len(right_labels) / total_samples
    
    left_gini = calculate_gini_index(left_labels)
    right_gini = calculate_gini_index(right_labels)
    
    weighted_child_gini = left_weight * left_gini + right_weight * right_gini
    
    return parent_gini - weighted_child_gini


def validate_gini_calculation():
    """
    Validation function to test Gini index calculations with known examples.
    """
    print("Validating Gini Index Calculations:")
    print("-" * 45)
    
    # Test 1: Pure dataset (Gini should be 0)
    pure_labels = ['A'] * 10
    pure_gini = calculate_gini_index(pure_labels)
    print(f"Test 1 - Pure dataset: {pure_gini:.4f} (Expected: 0.0000)")
    
    # Test 2: Balanced binary dataset (Gini should be 0.5)
    balanced_binary = ['A'] * 5 + ['B'] * 5
    balanced_gini = calculate_gini_index(balanced_binary)
    print(f"Test 2 - Balanced binary: {balanced_gini:.4f} (Expected: 0.5000)")
    
    # Test 3: Imbalanced dataset
    imbalanced = ['A'] * 8 + ['B'] * 2
    imbalanced_gini = calculate_gini_index(imbalanced)
    print(f"Test 3 - Imbalanced (80-20): {imbalanced_gini:.4f}")
    
    # Test 4: Three-class balanced (Gini should be 2/3 ≈ 0.6667)
    three_class = ['A'] * 5 + ['B'] * 5 + ['C'] * 5
    three_gini = calculate_gini_index(three_class)
    print(f"Test 4 - Three-class balanced: {three_gini:.4f} (Expected: 0.6667)")
    
    # Test 5: Four-class balanced (Gini should be 3/4 = 0.75)
    four_class = ['A'] * 5 + ['B'] * 5 + ['C'] * 5 + ['D'] * 5
    four_gini = calculate_gini_index(four_class)
    print(f"Test 5 - Four-class balanced: {four_gini:.4f} (Expected: 0.7500)")
    
    # Test 6: Empty dataset
    empty_gini = calculate_gini_index([])
    print(f"Test 6 - Empty dataset: {empty_gini:.4f} (Expected: 0.0000)")
    
    print("\nGini Gain Tests:")
    print("-" * 45)
    
    # Test Gini gain calculation
    parent = ['A'] * 4 + ['B'] * 6
    left_child = ['A'] * 3 + ['B'] * 1
    right_child = ['A'] * 1 + ['B'] * 5
    
    gain = gini_gain(parent, left_child, right_child)
    print(f"Parent Gini: {calculate_gini_index(parent):.4f}")
    print(f"Left child Gini: {calculate_gini_index(left_child):.4f}")
    print(f"Right child Gini: {calculate_gini_index(right_child):.4f}")
    print(f"Gini Gain: {gain:.4f}")
    
    print("\nWeighted Gini Test:")
    print("-" * 45)
    
    groups = {
        'group1': ['A', 'A', 'B'],
        'group2': ['B', 'C', 'C', 'C'],
        'group3': ['A']
    }
    
    weighted_gini = calculate_weighted_gini(groups)
    print(f"Groups: {groups}")
    print(f"Weighted Gini: {weighted_gini:.4f}")


def gini_properties_analysis():
    """
    Analyze properties of Gini index with different distributions.
    """
    print("\nGini Index Properties Analysis:")
    print("=" * 50)
    
    # Test different binary class distributions
    print("\nBinary Classification (A vs B):")
    print("-" * 30)
    
    binary_distributions = [
        (10, 0),   # 100% - 0%
        (9, 1),    # 90% - 10%
        (8, 2),    # 80% - 20%
        (7, 3),    # 70% - 30%
        (6, 4),    # 60% - 40%
        (5, 5),    # 50% - 50%
    ]
    
    for a_count, b_count in binary_distributions:
        labels = ['A'] * a_count + ['B'] * b_count
        gini = calculate_gini_index(labels)
        total = a_count + b_count
        if total > 0:
            a_pct = (a_count / total) * 100
            b_pct = (b_count / total) * 100
            print(f"{a_pct:3.0f}% - {b_pct:3.0f}%: Gini = {gini:.4f}")
    
    # Maximum Gini for different numbers of classes
    print(f"\nMaximum Gini Index for Balanced Classes:")
    print("-" * 40)
    
    for n_classes in range(2, 11):
        # Create perfectly balanced dataset
        labels = []
        samples_per_class = 10
        for i in range(n_classes):
            labels.extend([f'Class_{i}'] * samples_per_class)
        
        gini = calculate_gini_index(labels)
        theoretical_max = 1 - (1/n_classes)
        print(f"{n_classes} classes: Gini = {gini:.4f}, Theoretical Max = {theoretical_max:.4f}")


if __name__ == "__main__":
    validate_gini_calculation()
    gini_properties_analysis()