# q4.py - Advanced Binning (A4)
"""
A4. Advanced Binning Utilities Module with Function Overloading
This module provides comprehensive binning functions with overloading capabilities
for converting continuous features to categorical values.
"""

import numpy as np
import pandas as pd
from collections import Counter


def equal_width_binning(values, num_bins=4, custom_range=None):
    """
    Convert continuous values to categorical using equal width binning.
    
    Parameters:
    -----------
    values : list or array-like
        Continuous numerical values to bin
    num_bins : int, default=4
        Number of bins to create
    custom_range : tuple, optional
        Custom (min, max) range for binning
    
    Returns:
    --------
    tuple
        (binned_labels, bin_edges, bin_info)
    """
    if len(values) == 0:
        return [], np.array([]), {'method': 'equal_width', 'num_bins': num_bins}
    
    values = np.array(values)
    
    # Determine range
    if custom_range is not None:
        min_val, max_val = custom_range
    else:
        min_val, max_val = values.min(), values.max()
    
    # Handle case where all values are the same
    if min_val == max_val:
        return ['EW_Bin_0'] * len(values), np.array([min_val, max_val]), {
            'method': 'equal_width', 
            'num_bins': 1, 
            'range': (min_val, max_val),
            'bin_width': 0
        }
    
    # Create bin edges
    bin_width = (max_val - min_val) / num_bins
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    
    # Assign values to bins
    binned_values = []
    for value in values:
        if value == max_val:
            # Handle edge case where value equals maximum
            bin_number = num_bins - 1
        else:
            bin_number = int((value - min_val) / bin_width)
            bin_number = max(0, min(bin_number, num_bins - 1))  # Ensure within bounds
        
        binned_values.append(f"EW_Bin_{bin_number}")
    
    # Create bin info
    bin_info = {
        'method': 'equal_width',
        'num_bins': num_bins,
        'range': (min_val, max_val),
        'bin_width': bin_width,
        'bin_edges': bin_edges.tolist()
    }
    
    return binned_values, bin_edges, bin_info


def equal_frequency_binning(values, num_bins=4, handle_ties='first'):
    """
    Convert continuous values to categorical using equal frequency binning.
    
    Parameters:
    -----------
    values : list or array-like
        Continuous numerical values to bin
    num_bins : int, default=4
        Number of bins to create
    handle_ties : str, default='first'
        How to handle tied values
    
    Returns:
    --------
    tuple
        (binned_labels, quantiles, bin_info)
    """
    if len(values) == 0:
        return [], np.array([]), {'method': 'equal_frequency', 'num_bins': num_bins}
    
    values = np.array(values)
    
    # Handle case where all values are the same
    if len(np.unique(values)) == 1:
        return ['EF_Bin_0'] * len(values), np.array([values[0]]), {
            'method': 'equal_frequency',
            'num_bins': 1,
            'unique_values': 1
        }
    
    # Calculate quantiles
    quantile_positions = np.linspace(0, 100, num_bins + 1)
    quantiles = np.percentile(values, quantile_positions)
    
    # Handle duplicate quantiles
    unique_quantiles = []
    for i, q in enumerate(quantiles):
        if i == 0 or q > unique_quantiles[-1]:
            unique_quantiles.append(q)
    
    if len(unique_quantiles) < num_bins + 1:
        actual_num_bins = len(unique_quantiles) - 1
        quantiles = np.array(unique_quantiles)
    else:
        actual_num_bins = num_bins
    
    # Assign values to bins
    binned_values = []
    for value in values:
        bin_number = 0
        for i in range(1, len(quantiles)):
            if value <= quantiles[i]:
                bin_number = i - 1
                break
        
        if value == quantiles[-1]:
            bin_number = len(quantiles) - 2
        
        binned_values.append(f"EF_Bin_{bin_number}")
    
    bin_info = {
        'method': 'equal_frequency',
        'num_bins': actual_num_bins,
        'requested_bins': num_bins,
        'quantiles': quantiles.tolist(),
        'handle_ties': handle_ties
    }
    
    return binned_values, quantiles, bin_info


def custom_binning(values, bin_edges, labels=None):
    """
    Apply custom binning with user-defined bin edges.
    
    Parameters:
    -----------
    values : list or array-like
        Values to bin
    bin_edges : list
        List of bin edge values (must be sorted)
    labels : list, optional
        Custom labels for bins
    
    Returns:
    --------
    tuple
        (binned_labels, bin_edges_array, bin_info)
    """
    if len(values) == 0:
        return [], np.array(bin_edges), {'method': 'custom', 'num_bins': len(bin_edges) - 1}
    
    values = np.array(values)
    bin_edges = np.array(bin_edges)
    num_bins = len(bin_edges) - 1
    
    # Generate default labels if not provided
    if labels is None:
        labels = [f"Custom_Bin_{i}" for i in range(num_bins)]
    elif len(labels) != num_bins:
        raise ValueError(f"Number of labels ({len(labels)}) must match number of bins ({num_bins})")
    
    # Assign values to bins
    binned_values = []
    for value in values:
        bin_number = 0
        for i in range(1, len(bin_edges)):
            if value <= bin_edges[i]:
                bin_number = i - 1
                break
        
        # Handle values outside the range
        if value < bin_edges[0]:
            bin_number = 0
        elif value > bin_edges[-1]:
            bin_number = num_bins - 1
        
        binned_values.append(labels[bin_number])
    
    bin_info = {
        'method': 'custom',
        'num_bins': num_bins,
        'bin_edges': bin_edges.tolist(),
        'labels': labels
    }
    
    return binned_values, bin_edges, bin_info


class AdvancedBinning:
    """
    Advanced binning class with function overloading capabilities.
    
    This class implements function overloading through method variants.
    """
    
    def __init__(self, method='equal_width', num_bins=4, **kwargs):
        """Initialize binning configuration."""
        self.default_method = method
        self.default_num_bins = num_bins
        self.default_kwargs = kwargs
        self.binning_history = []
    
    def bin_data(self, values, method=None, num_bins=None, **kwargs):
        """
        Main binning function with overloading support.
        
        Parameters:
        -----------
        values : list or array-like
            Values to bin
        method : str, optional
            Binning method
        num_bins : int, optional
            Number of bins
        
        Returns:
        --------
        dict
            Complete binning results with metadata
        """
        # Use defaults if parameters not provided
        method = method or self.default_method
        num_bins = num_bins or self.default_num_bins
        
        # Merge kwargs with defaults
        combined_kwargs = {**self.default_kwargs, **kwargs}
        
        # Route to appropriate binning method
        if method == 'equal_width':
            labels, edges, info = equal_width_binning(
                values, num_bins, combined_kwargs.get('custom_range')
            )
        elif method == 'equal_frequency':
            labels, edges, info = equal_frequency_binning(
                values, num_bins, combined_kwargs.get('handle_ties', 'first')
            )
        elif method == 'custom':
            if 'bin_edges' not in combined_kwargs:
                raise ValueError("Custom binning requires 'bin_edges' parameter")
            labels, edges, info = custom_binning(
                values, combined_kwargs['bin_edges'], combined_kwargs.get('labels')
            )
        else:
            raise ValueError(f"Unknown binning method: {method}")
        
        # Create complete result
        result = {
            'binned_labels': labels,
            'bin_edges': edges,
            'bin_info': info,
            'original_values': values,
            'distribution': Counter(labels)
        }
        
        # Store in history
        self.binning_history.append({
            'method': method,
            'num_bins': num_bins,
            'result_summary': {
                'num_values': len(values),
                'num_unique_bins': len(set(labels)),
                'distribution': dict(Counter(labels))
            }
        })
        
        return result
    
    # Function overloading variants
    def bin_with_defaults(self, values):
        """Overloaded method: bin with all default parameters."""
        return self.bin_data(values)
    
    def bin_with_method(self, values, method):
        """Overloaded method: bin with specified method only."""
        return self.bin_data(values, method=method)
    
    def bin_with_bins(self, values, num_bins):
        """Overloaded method: bin with specified number of bins only."""
        return self.bin_data(values, num_bins=num_bins)
    
    def bin_complete(self, values, method, num_bins, **kwargs):
        """Overloaded method: bin with all parameters specified."""
        return self.bin_data(values, method=method, num_bins=num_bins, **kwargs)


def validate_binning_functions():
    """
    Comprehensive validation of all binning functions.
    """
    print("Validating Binning Functions:")
    print("=" * 50)
    
    # Test data
    test_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    # Test equal width binning
    print("\n1. Equal Width Binning Test:")
    ew_labels, ew_edges, ew_info = equal_width_binning(test_values, 3)
    print(f"   Values: {test_values}")
    print(f"   Labels: {ew_labels}")
    print(f"   Edges: {ew_edges}")
    print(f"   Distribution: {Counter(ew_labels)}")
    
    # Test equal frequency binning
    print("\n2. Equal Frequency Binning Test:")
    ef_labels, ef_quantiles, ef_info = equal_frequency_binning(test_values, 3)
    print(f"   Labels: {ef_labels}")
    print(f"   Quantiles: {ef_quantiles}")
    print(f"   Distribution: {Counter(ef_labels)}")
    
    # Test custom binning
    print("\n3. Custom Binning Test:")
    custom_edges = [0, 4, 8, 13]
    custom_labels_list = ['Low', 'Medium', 'High']
    c_labels, c_edges, c_info = custom_binning(test_values, custom_edges, custom_labels_list)
    print(f"   Custom edges: {custom_edges}")
    print(f"   Labels: {c_labels}")
    print(f"   Distribution: {Counter(c_labels)}")
    
    # Test advanced binning class
    print("\n4. Advanced Binning Class Test:")
    binner = AdvancedBinning('equal_width', 4)
    
    # Test method overloading
    result1 = binner.bin_with_defaults(test_values)
    print(f"   Default binning: {Counter(result1['binned_labels'])}")
    
    result2 = binner.bin_with_method(test_values, 'equal_frequency')
    print(f"   Frequency binning: {Counter(result2['binned_labels'])}")
    
    result3 = binner.bin_with_bins(test_values, 2)
    print(f"   2-bin binning: {Counter(result3['binned_labels'])}")


if __name__ == "__main__":
    validate_binning_functions()