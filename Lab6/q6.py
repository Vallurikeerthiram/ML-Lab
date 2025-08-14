# q6.py - Tree Visualization (A6)
"""
A6. Decision Tree Visualization Module
This module provides comprehensive visualization capabilities for decision trees.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from collections import Counter
import json


class TreeVisualizer:
    """
    Comprehensive tree visualization class with multiple display options.
    """
    
    def __init__(self, tree_model=None):
        """Initialize the visualizer."""
        self.tree_model = tree_model
        self.color_scheme = {
            'node_colors': {
                'internal': '#E8F4FD',
                'leaf_pure': '#D4E6F1',
                'leaf_impure': '#F8D7DA'
            },
            'text_colors': {
                'feature': '#2C3E50',
                'prediction': '#27AE60',
                'statistics': '#8E44AD'
            }
        }
    
    def print_tree_structure(self, max_depth=None, show_statistics=True):
        """
        Print a detailed text representation of the tree structure.
        """
        if self.tree_model is None or self.tree_model.root is None:
            print("No tree model provided or tree not trained.")
            return
        
        def print_node(node, prefix="", is_last=True, current_depth=0):
            if max_depth is not None and current_depth > max_depth:
                print(f"{prefix}{'└── ' if is_last else '├── '}[... max depth reached]")
                return
            
            # Create the current line
            connector = "└── " if is_last else "├── "
            
            if node.is_leaf:
                # Leaf node display
                purity = "PURE" if node.impurity == 0 else f"Impurity: {node.impurity:.3f}"
                
                if show_statistics:
                    class_dist = ", ".join([f"{k}: {v}" for k, v in node.class_distribution.items()])
                    print(f"{prefix}{connector}🍃 PREDICT: {node.prediction}")
                    print(f"{prefix}{'    ' if is_last else '│   '}   📊 {node.samples} samples, {purity}")
                    print(f"{prefix}{'    ' if is_last else '│   '}   📈 Distribution: {class_dist}")
                else:
                    print(f"{prefix}{connector}🍃 {node.prediction} ({node.samples} samples)")
            else:
                # Internal node display
                if show_statistics:
                    print(f"{prefix}{connector}🔍 SPLIT on {node.feature}")
                    print(f"{prefix}{'    ' if is_last else '│   '}   📊 {node.samples} samples, "
                          f"Gain: {node.information_gain:.4f}")
                    print(f"{prefix}{'    ' if is_last else '│   '}   📈 Impurity: {node.impurity:.3f}")
                else:
                    print(f"{prefix}{connector}🔍 {node.feature}")
                
                # Print children
                children = list(node.children.items())
                for i, (condition, child) in enumerate(children):
                    is_last_child = (i == len(children) - 1)
                    child_prefix = prefix + ("    " if is_last else "│   ")
                    
                    condition_display = f"[{condition}]"
                    print(f"{child_prefix}{'└── ' if is_last_child else '├── '}{condition_display}")
                    print_node(child, 
                              child_prefix + ("    " if is_last_child else "│   "), 
                              True, current_depth + 1)
        
        print("Decision Tree Structure")
        print("=" * 60)
        print(f"Criterion: {self.tree_model.criterion}")
        print(f"Total Nodes: {self.tree_model.n_nodes_}")
        print(f"Leaves: {self.tree_model.n_leaves_}")
        print(f"Max Depth: {self.tree_model.tree_depth_}")
        print("=" * 60)
        
        print_node(self.tree_model.root)
    
    def create_feature_importance_plot(self, figsize=(10, 6), save_path=None):
        """Create a feature importance plot."""
        if self.tree_model is None or self.tree_model.feature_importances_ is None:
            print("No feature importances available.")
            return None
        
        # Prepare data
        features = list(self.tree_model.feature_importances_.keys())
        importances = list(self.tree_model.feature_importances_.values())
        
        # Sort by importance
        sorted_indices = np.argsort(importances)[::-1]
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))
        bars = ax.bar(range(len(sorted_features)), sorted_importances, color=colors)
        
        ax.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance in Decision Tree', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(sorted_features)))
        ax.set_xticklabels(sorted_features, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, sorted_importances)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{importance:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        
        plt.show()
        return fig
    
    def generate_tree_report(self):
        """Generate a comprehensive text report about the tree."""
        if self.tree_model is None:
            return "No tree model available."
        
        report = []
        report.append("DECISION TREE ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Basic statistics
        report.append("TREE STATISTICS:")
        report.append(f"  • Total Nodes: {self.tree_model.n_nodes_}")
        report.append(f"  • Leaf Nodes: {self.tree_model.n_leaves_}")
        report.append(f"  • Internal Nodes: {self.tree_model.n_nodes_ - self.tree_model.n_leaves_}")
        report.append(f"  • Maximum Depth: {self.tree_model.tree_depth_}")
        report.append(f"  • Number of Features: {len(self.tree_model.feature_names_)}")
        report.append(f"  • Splitting Criterion: {self.tree_model.criterion}")
        report.append("")
        
        # Feature importance
        if self.tree_model.feature_importances_:
            report.append("FEATURE IMPORTANCE RANKING:")
            sorted_features = sorted(self.tree_model.feature_importances_.items(), 
                                   key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_features, 1):
                report.append(f"  {i:2d}. {feature:15s} : {importance:.6f}")
            report.append("")
        
        # Training history
        if self.tree_model.training_history:
            report.append("TRAINING STATISTICS:")
            history = self.tree_model.training_history
            report.append(f"  • Nodes Created: {history.get('nodes_created', 'N/A')}")
            report.append(f"  • Splits Evaluated: {history.get('splits_evaluated', 'N/A')}")
            report.append("")
        
        # Tree complexity analysis
        report.append("COMPLEXITY ANALYSIS:")
        leaves_ratio = self.tree_model.n_leaves_ / self.tree_model.n_nodes_ if self.tree_model.n_nodes_ > 0 else 0
        
        report.append(f"  • Leaf Ratio: {leaves_ratio:.3f}")
        
        if self.tree_model.tree_depth_ > 10:
            report.append("  ⚠️  Warning: Deep tree detected (may overfit)")
        if leaves_ratio < 0.3:
            report.append("  ⚠️  Warning: Low leaf ratio (tree may be too complex)")
        
        return "\n".join(report)
    
    def plot_tree_matplotlib(self, figsize=(12, 8), save_path=None):
        """Create a matplotlib visualization of the decision tree."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib is required for tree plotting. Please install it.")
            return None
        
        if self.tree_model is None or self.tree_model.root is None:
            print("No tree model available for plotting.")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # For simplicity, create a text-based tree representation
        def draw_simple_tree(node, x, y, width, depth=0):
            if node is None:
                return
            
            # Draw node
            if node.is_leaf:
                rect = plt.Rectangle((x-width/4, y-0.05), width/2, 0.1, 
                                   facecolor='lightgreen', edgecolor='black')
                ax.add_patch(rect)
                ax.text(x, y, f"{node.prediction}\n({node.samples})", 
                       ha='center', va='center', fontsize=8)
            else:
                rect = plt.Rectangle((x-width/4, y-0.05), width/2, 0.1, 
                                   facecolor='lightblue', edgecolor='black')
                ax.add_patch(rect)
                ax.text(x, y, f"{node.feature}\n({node.samples})", 
                       ha='center', va='center', fontsize=8)
                
                # Draw children
                children = list(node.children.items())
                if len(children) > 0:
                    child_width = width / len(children)
                    start_x = x - width/2 + child_width/2
                    
                    for i, (condition, child) in enumerate(children):
                        child_x = start_x + i * child_width
                        child_y = y - 0.3
                        
                        # Draw connection line
                        ax.plot([x, child_x], [y-0.05, child_y+0.05], 'k-', alpha=0.6)
                        
                        # Draw condition label
                        ax.text((x + child_x)/2, (y + child_y)/2, str(condition), 
                               ha='center', va='center', fontsize=6, 
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
                        
                        draw_simple_tree(child, child_x, child_y, child_width, depth+1)
        
        # Start drawing from root
        draw_simple_tree(self.tree_model.root, 0, 0, 2.0)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-2, 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.title(f"Decision Tree Visualization ({self.tree_model.criterion} criterion)", 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='lightblue', label='Internal Node'),
            mpatches.Patch(color='lightgreen', label='Leaf Node')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Tree visualization saved to: {save_path}")
        
        plt.show()
        return fig


def demonstrate_tree_visualization():
    """Demonstrate tree visualization capabilities with sample data."""
    print("Demonstrating Tree Visualization Capabilities:")
    print("=" * 60)
    
    # This would be used with actual tree models
    print("Note: This demonstration requires trained decision tree models.")
    print("Use this module with your CustomDecisionTree instances.")
    print("")
    
    print("Available visualization methods:")
    print("1. print_tree_structure() - Text-based tree display")
    print("2. plot_tree_matplotlib() - Matplotlib tree diagram") 
    print("3. create_feature_importance_plot() - Feature importance plot")
    print("4. generate_tree_report() - Comprehensive text report")


if __name__ == "__main__":
    demonstrate_tree_visualization()