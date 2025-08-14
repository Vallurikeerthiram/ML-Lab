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
from q2 import calculate_gini_index, calculate_weighted_gini


def calculate_information_gain(data, feature_col, target_col, criterion='entropy', 
                             binning_method='equal_width', num_bins=4):
    """
    Calculate information gain for a feature.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset containing features and target
    feature_col : str
        Name of the feature column to evaluate
    target_col : str
        Name of the target column
    criterion : str, default='entropy'
        Impurity measure ('entropy' or 'gini')
    binning_method : str, default='equal_width'
        Method for binning continuous features
    num_bins : int, default=4
        Number of bins for continuous features
    
    Returns:
    --------
    tuple
        (information_gain, feature_groups, binned_feature_values)
    """
    # Get target values
    target_values = data[target_col].tolist()
    feature_values = data[feature_col].tolist()
    
    # Check if feature needs binning (continuous with many unique values)
    unique_values = len(set(feature_values))
    is_continuous = (data[feature_col].dtype in ['float64', 'int64'] and unique_values > 10)
    
    binned_feature_values = feature_values.copy()
    
    # Apply binning if necessary
    if is_continuous:
        if binning_method == 'equal_width':
            binned_feature_values = equal_width_binning(feature_values, num_bins)
    
    # Calculate parent impurity
    if criterion == 'entropy':
        parent_impurity = calculate_entropy(target_values)
    elif criterion == 'gini':
        parent_impurity = calculate_gini_index(target_values)
    else:
        raise ValueError("Criterion must be 'entropy' or 'gini'")
    
    # Group data by feature values
    feature_groups = {}
    for i, feature_val in enumerate(binned_feature_values):
        if feature_val not in feature_groups:
            feature_groups[feature_val] = []
        feature_groups[feature_val].append(target_values[i])
    
    # Calculate weighted impurity of subsets
    total_samples = len(target_values)
    weighted_impurity = 0
    
    for feature_val, subset_targets in feature_groups.items():
        if len(subset_targets) > 0:
            subset_size = len(subset_targets)
            weight = subset_size / total_samples
            
            if criterion == 'entropy':
                subset_impurity = calculate_entropy(subset_targets)
            else:  # gini
                subset_impurity = calculate_gini_index(subset_targets)
            
            weighted_impurity += weight * subset_impurity
    
    # Calculate information gain
    information_gain = parent_impurity - weighted_impurity
    
    return information_gain, feature_groups, binned_feature_values


def find_best_root_feature(data, target_col, feature_cols=None, criterion='entropy',
                          binning_method='equal_width', num_bins=4):
    """
    Find the best feature for the root node of a decision tree.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset containing features and target
    target_col : str
        Name of the target column
    feature_cols : list, optional
        List of feature columns to consider
    criterion : str, default='entropy'
        Impurity measure ('entropy' or 'gini')
    
    Returns:
    --------
    dict
        Dictionary containing best feature information
    """
    if feature_cols is None:
        feature_cols = [col for col in data.columns if col != target_col]
    
    best_feature = None
    best_gain = -1
    best_groups = None
    best_binned_values = None
    all_gains = {}
    
    print(f"Evaluating features for root node (criterion: {criterion}):")
    print("-" * 60)
    
    for feature in feature_cols:
        try:
            gain, groups, binned_vals = calculate_information_gain(
                data, feature, target_col, criterion, binning_method, num_bins
            )
            
            all_gains[feature] = gain
            
            # Track feature statistics
            unique_original = len(set(data[feature]))
            unique_after_binning = len(set(binned_vals)) if binned_vals != data[feature].tolist() else unique_original
            
            print(f"Feature: {feature:15} | Gain: {gain:.6f} | "
                  f"Unique values: {unique_original:3d} → {unique_after_binning:3d}")
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_groups = groups
                best_binned_values = binned_vals
                
        except Exception as e:
            print(f"Error processing feature {feature}: {str(e)}")
            all_gains[feature] = 0
    
    print("-" * 60)
    print(f"Best feature selected: {best_feature} with gain = {best_gain:.6f}")
    
    return {
        'best_feature': best_feature,
        'best_gain': best_gain,
        'best_groups': best_groups,
        'all_gains': all_gains,
        'binned_values': best_binned_values,
        'criterion_used': criterion
    }


def validate_information_gain():
    """
    Validate information gain calculations with known examples.
    """
    print("Validating Information Gain Calculations:")
    print("=" * 50)
    
    # Create test dataset
    test_data = pd.DataFrame({
        'perfect_split': ['low', 'low', 'high', 'high', 'high', 'high'],
        'no_split': ['same', 'same', 'same', 'same', 'same', 'same'],
        'partial_split': ['A', 'A', 'B', 'B', 'C', 'A'],
        'target': ['yes', 'yes', 'no', 'no', 'no', 'no']
    })
    
    print(f"Test dataset:")
    print(test_data)
    print()
    
    # Test each feature
    features = ['perfect_split', 'no_split', 'partial_split']
    
    for feature in features:
        gain, groups, _ = calculate_information_gain(test_data, feature, 'target')
        print(f"Feature '{feature}':")
        print(f"  Information Gain: {gain:.4f}")
        print(f"  Groups: {len(groups)}")
        for group_name, group_targets in groups.items():
            print(f"    {group_name}: {Counter(group_targets)}")
        print()
    
    # Find best root feature
    result = find_best_root_feature(test_data, 'target')
    print(f"Best root feature: {result['best_feature']}")
    print(f"Best gain: {result['best_gain']:.4f}")


if __name__ == "__main__":
    validate_information_gain()