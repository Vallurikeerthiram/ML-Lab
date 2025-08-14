# q5.py - Decision Tree (A5) - FIXED VERSION
"""
A5. Complete Decision Tree Module - CORRECTED TO PREVENT OVERFITTING
This module implements a full decision tree classifier with proper stopping criteria.
"""

import numpy as np
import pandas as pd
from collections import Counter
from q1 import calculate_entropy
from q2 import calculate_gini_index
from q3 import calculate_information_gain
from q4 import AdvancedBinning


class DecisionTreeNode:
    """Class representing a single node in the decision tree."""
    
    def __init__(self, depth=0):
        self.feature = None
        self.threshold = None
        self.prediction = None
        self.is_leaf = False
        self.children = {}
        self.samples = 0
        self.impurity = 0.0
        self.class_distribution = {}
        self.depth = depth
        self.information_gain = 0.0
        self.binning_info = None
    
    def add_child(self, condition, child_node):
        """Add a child node with a specific condition."""
        self.children[condition] = child_node
    
    def get_child(self, feature_value):
        """Get child node for a given feature value."""
        return self.children.get(feature_value, None)
    
    def make_leaf(self, prediction, class_distribution):
        """Convert this node to a leaf with given prediction."""
        self.is_leaf = True
        self.prediction = prediction
        self.class_distribution = class_distribution
    
    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(prediction={self.prediction}, samples={self.samples})"
        else:
            return f"Node(feature={self.feature}, samples={self.samples}, children={len(self.children)})"


class CustomDecisionTree:
    """
    Custom Decision Tree Classifier implementation with proper overfitting prevention.
    """
    
    def __init__(self, criterion='entropy', max_depth=5, min_samples_split=5,
                 min_samples_leaf=2, max_features=None, binning_method='equal_width',
                 num_bins=4, random_state=None):
        """
        Initialize the Decision Tree Classifier with PROPER PARAMETERS.
        
        Parameters:
        -----------
        criterion : str, default='entropy'
            Splitting criterion ('entropy' or 'gini')
        max_depth : int, default=5  # CHANGED FROM None TO 5
            Maximum depth of the tree
        min_samples_split : int, default=5  # CHANGED FROM 2 TO 5
            Minimum samples required to split a node
        min_samples_leaf : int, default=2  # CHANGED FROM 1 TO 2
            Minimum samples required at a leaf node
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.binning_method = binning_method
        self.num_bins = num_bins
        self.random_state = random_state
        
        # Initialize after training
        self.root = None
        self.classes_ = None
        self.feature_names_ = None
        self.feature_importances_ = None
        self.tree_depth_ = 0
        self.n_nodes_ = 0
        self.n_leaves_ = 0
        
        # Binning utilities
        self.binner = AdvancedBinning(binning_method, num_bins)
        self.feature_binning_info = {}
        
        # Training history
        self.training_history = {
            'nodes_created': 0,
            'splits_evaluated': 0,
            'pruning_actions': 0
        }
    
    def _calculate_impurity(self, y):
        """Calculate impurity based on the chosen criterion."""
        if len(y) == 0:
            return 0.0
        
        if self.criterion == 'entropy':
            return calculate_entropy(y)
        elif self.criterion == 'gini':
            return calculate_gini_index(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _get_majority_class(self, y):
        """Get the most frequent class in y."""
        if len(y) == 0:
            return None
        return Counter(y).most_common(1)[0][0]
    
    def _should_stop_splitting(self, X, y, depth):
        """Check if we should stop splitting at this node."""
        # Check depth limit
        if self.max_depth is not None and depth >= self.max_depth:
            return True, "Max depth reached"
        
        # Check minimum samples for splitting
        if len(y) < self.min_samples_split:
            return True, f"Too few samples for split ({len(y)} < {self.min_samples_split})"
        
        # Check if all samples have the same class
        if len(set(y)) == 1:
            return True, "Pure node (all samples same class)"
        
        # Check if we have any features left
        if X.shape[1] == 0:
            return True, "No features available"
        
        return False, "Continue splitting"
    
    def _find_best_split(self, X, y, feature_names):
        """Find the best feature and split point."""
        best_feature = None
        best_gain = -1
        best_groups = None
        best_binning_info = None
        
        n_features = X.shape[1]
        
        # Determine which features to consider
        if self.max_features is not None:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            feature_indices = np.random.choice(n_features, 
                                             min(self.max_features, n_features), 
                                             replace=False)
        else:
            feature_indices = range(n_features)
        
        # Evaluate each feature
        for feature_idx in feature_indices:
            self.training_history['splits_evaluated'] += 1
            
            feature_name = feature_names[feature_idx]
            feature_values = X[:, feature_idx]
            
            # Create temporary dataframe for information gain calculation
            temp_df = pd.DataFrame({
                feature_name: feature_values,
                'target': y
            })
            
            try:
                gain, groups, binned_values = calculate_information_gain(
                    temp_df, feature_name, 'target', 
                    self.criterion, self.binning_method, self.num_bins
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_name
                    best_groups = groups
                    
                    # Store binning information if feature was binned
                    if binned_values != feature_values.tolist():
                        binning_result = self.binner.bin_data(
                            feature_values, self.binning_method, self.num_bins
                        )
                        best_binning_info = binning_result['bin_info']
                    else:
                        best_binning_info = None
                        
            except Exception as e:
                print(f"Warning: Error evaluating feature {feature_name}: {e}")
                continue
        
        return best_feature, best_gain, best_groups, best_binning_info
    
    def _split_data(self, X, y, feature_name, feature_groups, binning_info=None):
        """Split the data based on the best feature."""
        feature_idx = list(self.feature_names_).index(feature_name)
        feature_values = X[:, feature_idx]
        
        # Apply binning if necessary
        if binning_info is not None:
            binning_result = self.binner.bin_data(
                feature_values, binning_info['method'], binning_info['num_bins']
            )
            binned_values = binning_result['binned_labels']
        else:
            binned_values = feature_values.tolist()
        
        # Split data into groups
        split_data = {}
        for group_value in feature_groups.keys():
            # Find indices for this group
            indices = [i for i, val in enumerate(binned_values) if val == group_value]
            
            if len(indices) > 0:
                split_data[group_value] = {
                    'X': X[indices],
                    'y': [y[i] for i in indices],
                    'indices': indices
                }
        
        return split_data
    
    def _build_tree(self, X, y, feature_names, depth=0, parent_class=None):
        """Recursively build the decision tree."""
        node = DecisionTreeNode(depth)
        node.samples = len(y)
        node.impurity = self._calculate_impurity(y)
        node.class_distribution = dict(Counter(y))
        
        self.training_history['nodes_created'] += 1
        
        # Check stopping criteria
        should_stop, reason = self._should_stop_splitting(X, y, depth)
        
        if should_stop:
            # Create leaf node
            majority_class = self._get_majority_class(y) or parent_class
            node.make_leaf(majority_class, node.class_distribution)
            self.n_leaves_ += 1
            return node
        
        # Find best split
        best_feature, best_gain, best_groups, best_binning_info = self._find_best_split(
            X, y, feature_names
        )
        
        # ADDITIONAL STOPPING CRITERIA TO PREVENT OVERFITTING
        if best_feature is None or best_gain <= 0.01:  # Minimum gain threshold
            # No good split found, create leaf
            majority_class = self._get_majority_class(y)
            node.make_leaf(majority_class, node.class_distribution)
            self.n_leaves_ += 1
            return node
        
        # Set node properties
        node.feature = best_feature
        node.information_gain = best_gain
        node.binning_info = best_binning_info
        
        # Store feature binning info for prediction
        if best_binning_info is not None:
            self.feature_binning_info[best_feature] = best_binning_info
        
        # Split data and create children
        split_data = self._split_data(X, y, best_feature, best_groups, best_binning_info)
        
        majority_class = self._get_majority_class(y)
        
        for group_value, group_data in split_data.items():
            # Check minimum samples for leaf - PROPER ENFORCEMENT
            if len(group_data['y']) < self.min_samples_leaf:
                # Create leaf with parent's majority class
                leaf = DecisionTreeNode(depth + 1)
                leaf.samples = len(group_data['y'])
                leaf.make_leaf(majority_class, dict(Counter(group_data['y'])))
                node.add_child(group_value, leaf)
                self.n_leaves_ += 1
            else:
                # Recursively build child
                child = self._build_tree(
                    group_data['X'], group_data['y'], feature_names, 
                    depth + 1, majority_class
                )
                node.add_child(group_value, child)
        
        return node
    
    def fit(self, X, y):
        """Train the decision tree on the given data."""
        # Convert to numpy arrays and extract feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        else:
            X = np.array(X)
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        # Store classes
        self.classes_ = sorted(list(set(y)))
        
        # Reset training history
        self.training_history = {
            'nodes_created': 0,
            'splits_evaluated': 0,
            'pruning_actions': 0
        }
        
        # Build the tree
        self.root = self._build_tree(X, y, self.feature_names_)
        
        # Calculate tree statistics
        self.tree_depth_ = self._calculate_tree_depth(self.root)
        self.n_nodes_ = self.training_history['nodes_created']
        
        # Calculate feature importances
        self._calculate_feature_importances()
        
        return self
    
    def _predict_sample(self, x, node):
        """Predict a single sample."""
        if node.is_leaf:
            return node.prediction
        
        feature_value = x[list(self.feature_names_).index(node.feature)]
        
        # Apply binning if necessary
        if node.feature in self.feature_binning_info:
            binning_info = self.feature_binning_info[node.feature]
            binning_result = self.binner.bin_data(
                [feature_value], binning_info['method'], binning_info['num_bins']
            )
            feature_value = binning_result['binned_labels'][0]
        
        # Navigate to child node
        child = node.get_child(feature_value)
        
        if child is not None:
            return self._predict_sample(x, child)
        else:
            # No child for this value, return current node's prediction
            return node.prediction if hasattr(node, 'prediction') else self._get_majority_class(
                list(node.class_distribution.keys())
            )
    
    def predict(self, X):
        """Predict classes for the given samples."""
        if self.root is None:
            raise ValueError("Tree has not been trained yet. Call fit() first.")
        
        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.array(X)
        
        predictions = []
        for sample in X:
            pred = self._predict_sample(sample, self.root)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _calculate_tree_depth(self, node):
        """Calculate the depth of the tree."""
        if node.is_leaf:
            return node.depth
        
        max_child_depth = node.depth
        for child in node.children.values():
            child_depth = self._calculate_tree_depth(child)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _calculate_feature_importances(self):
        """Calculate feature importances based on information gain."""
        importances = {feature: 0.0 for feature in self.feature_names_}
        
        def traverse_node(node):
            if not node.is_leaf and node.feature is not None:
                importances[node.feature] += node.information_gain * node.samples
                for child in node.children.values():
                    traverse_node(child)
        
        if self.root is not None:
            traverse_node(self.root)
            
            # Normalize by total samples
            total_samples = self.root.samples
            for feature in importances:
                importances[feature] /= total_samples
        
        self.feature_importances_ = importances
    
    def print_tree(self, max_depth=None):
        """Print a human-readable representation of the tree."""
        def print_node(node, prefix="", is_last=True, current_depth=0):
            if max_depth is not None and current_depth > max_depth:
                return
            
            # Create the current line
            connector = "└── " if is_last else "├── "
            
            if node.is_leaf:
                print(f"{prefix}{connector}Predict: {node.prediction} "
                      f"(samples: {node.samples}, impurity: {node.impurity:.3f})")
            else:
                print(f"{prefix}{connector}Feature: {node.feature} "
                      f"(samples: {node.samples}, gain: {node.information_gain:.3f})")
                
                # Print children
                children = list(node.children.items())
                for i, (condition, child) in enumerate(children):
                    is_last_child = (i == len(children) - 1)
                    child_prefix = prefix + ("    " if is_last else "│   ")
                    
                    print(f"{child_prefix}{'└── ' if is_last_child else '├── '}{condition}:")
                    print_node(child, child_prefix + ("    " if is_last_child else "│   "), 
                              True, current_depth + 1)
        
        if self.root is None:
            print("Tree has not been trained yet.")
            return
        
        print("Decision Tree Structure:")
        print("=" * 50)
        print_node(self.root)


def validate_decision_tree():
    """Validate the decision tree implementation with test data."""
    print("Validating Custom Decision Tree Implementation:")
    print("=" * 60)
    
    # Create test dataset
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 100
    X = pd.DataFrame({
        'feature1': np.random.normal(5, 2, n_samples),
        'feature2': np.random.uniform(0, 10, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    # Create target based on rules
    y = []
    for _, row in X.iterrows():
        if row['feature1'] > 5 and row['feature2'] > 5:
            y.append('Class1')
        elif row['feature3'] == 'A':
            y.append('Class2')
        else:
            y.append('Class3')
    
    y = np.array(y)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {Counter(y)}")
    
    # Train decision tree with PROPER PARAMETERS
    dt = CustomDecisionTree(
        criterion='entropy', 
        max_depth=5,           # Prevent deep trees
        min_samples_split=5,   # Require minimum samples to split
        min_samples_leaf=2     # Require minimum samples in leaves
    )
    dt.fit(X, y)
    
    # Make predictions
    predictions = dt.predict(X)
    accuracy = np.mean(predictions == y)
    
    print(f"\nTraining Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Tree depth: {dt.tree_depth_}")
    print(f"Number of nodes: {dt.n_nodes_}")
    print(f"Number of leaves: {dt.n_leaves_}")
    
    # Print tree structure
    dt.print_tree(max_depth=3)
    
    # Get training summary
    print(f"\nFeature Importances:")
    for feature, importance in dt.feature_importances_.items():
        print(f"  {feature}: {importance:.4f}")
    
    # Show that overfitting is prevented
    print(f"\nOverfitting Prevention Check:")
    print(f"  Samples per leaf (avg): {X.shape[0] / dt.n_leaves_:.1f}")
    print(f"  Tree complexity (nodes/samples): {dt.n_nodes_ / X.shape[0]:.3f}")
    
    if dt.n_leaves_ < X.shape[0] * 0.5:  # Less than 50% of samples as leaves
        print("  ✓ GOOD: Proper generalization - not overfitting")
    else:
        print("  ⚠ WARNING: Potential overfitting detected")


if __name__ == "__main__":
    validate_decision_tree()