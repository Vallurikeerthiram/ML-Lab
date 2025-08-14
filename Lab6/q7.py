# q7.py - Decision Boundary Visualization (A7)
"""
A7. Decision Boundary Visualization Module
This module provides functions to visualize decision boundaries created by 
decision trees in 2D feature space.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter


class DecisionBoundaryVisualizer:
    """
    Visualize decision boundaries for 2D classification problems.
    """
    
    def __init__(self, tree_model=None):
        """Initialize the decision boundary visualizer."""
        self.tree_model = tree_model
        self.feature_names = None
        self.class_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
            '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD',
            '#A8E6CF', '#FFD93D', '#6C5CE7', '#FD79A8'
        ]
    
    def plot_decision_boundary(self, X, y, feature_names=None, resolution=100, 
                              figsize=(12, 8), save_path=None, show_grid=True,
                              show_training_points=True, alpha=0.8):
        """
        Plot decision boundary for 2D data.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            2D feature data (n_samples x 2)
        y : array-like
            Target labels
        feature_names : list, optional
            Names of the two features
        resolution : int, default=100
            Resolution of the boundary mesh
        figsize : tuple, default=(12, 8)
            Figure size in inches
        save_path : str, optional
            Path to save the plot
        show_training_points : bool, default=True
            Whether to overlay training points
        alpha : float, default=0.8
            Transparency of decision regions
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        if self.tree_model is None or self.tree_model.root is None:
            raise ValueError("No trained tree model available.")
        
        # Convert to numpy arrays and extract feature names
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns[:2].tolist()
            X = X.values
        else:
            X = np.array(X)
            if feature_names is None:
                feature_names = ['Feature 1', 'Feature 2']
        
        if X.shape[1] != 2:
            raise ValueError("Decision boundary visualization requires exactly 2 features.")
        
        y = np.array(y)
        
        # Store feature names for internal use
        self.feature_names = feature_names
        
        # Get data bounds with some padding
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        # Create mesh grid
        h = max((x_max - x_min) / resolution, (y_max - y_min) / resolution)
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Make predictions on mesh grid
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_df = pd.DataFrame(mesh_points, columns=self.feature_names)
        
        try:
            Z = self.tree_model.predict(mesh_df)
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None
        
        # Reshape predictions to mesh grid shape
        Z = np.array(Z).reshape(xx.shape)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create color map for classes
        unique_classes = sorted(list(set(y)))
        n_classes = len(unique_classes)
        colors = self.class_colors[:n_classes]
        cmap = ListedColormap(colors)
        
        # Map class labels to numbers for plotting
        class_to_num = {cls: i for i, cls in enumerate(unique_classes)}
        Z_numeric = np.array([[class_to_num.get(pred, 0) for pred in row] for row in Z])
        
        # Plot decision regions
        ax.contourf(xx, yy, Z_numeric, levels=n_classes, alpha=alpha, cmap=cmap)
        
        # Plot decision boundaries (contour lines)
        ax.contour(xx, yy, Z_numeric, levels=n_classes, colors='black', 
                  linewidths=1.5, alpha=0.6)
        
        # Show grid points if requested
        if show_grid:
            ax.scatter(xx[::5, ::5], yy[::5, ::5], c='gray', s=1, alpha=0.3)
        
        # Plot training points if requested
        if show_training_points:
            for i, cls in enumerate(unique_classes):
                mask = y == cls
                ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=60, 
                          alpha=0.9, edgecolors='black', linewidth=1,
                          label=f'{cls} (n={np.sum(mask)})')
        
        # Customize plot
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(feature_names[0], fontsize=12, fontweight='bold')
        ax.set_ylabel(feature_names[1], fontsize=12, fontweight='bold')
        ax.set_title('Decision Tree - Decision Boundary Visualization', 
                    fontsize=14, fontweight='bold')
        
        if show_training_points:
            ax.legend(title='Classes', loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add tree information as text
        tree_info = f"Depth: {self.tree_model.tree_depth_}, Nodes: {self.tree_model.n_nodes_}, Leaves: {self.tree_model.n_leaves_}"
        ax.text(0.02, 0.98, tree_info, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Decision boundary plot saved to: {save_path}")
        
        plt.show()
        return fig
    
    def analyze_decision_regions(self, X, y, feature_names=None, resolution=100):
        """
        Analyze the characteristics of decision regions created by the tree.
        
        Parameters:
        -----------
        X : array-like
            2D feature data
        y : array-like
            Target labels
        feature_names : list, optional
            Feature names
        resolution : int, default=100
            Analysis resolution
        
        Returns:
        --------
        dict
            Analysis results
        """
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns[:2].tolist()
            X = X.values
        else:
            X = np.array(X)
            if feature_names is None:
                feature_names = ['Feature 1', 'Feature 2']
        
        y = np.array(y)
        
        # Create analysis mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        h = max((x_max - x_min) / resolution, (y_max - y_min) / resolution)
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_df = pd.DataFrame(mesh_points, columns=feature_names)
        predictions = self.tree_model.predict(mesh_df)
        
        # Analyze regions
        unique_classes = sorted(list(set(y)))
        region_analysis = {}
        
        total_area = (x_max - x_min) * (y_max - y_min)
        unit_area = h * h
        
        for cls in unique_classes:
            cls_points = sum(1 for pred in predictions if pred == cls)
            cls_area = cls_points * unit_area
            area_percentage = (cls_area / total_area) * 100
            
            # Find training points in this class
            training_points = sum(1 for label in y if label == cls)
            
            region_analysis[cls] = {
                'predicted_points': cls_points,
                'area_estimate': cls_area,
                'area_percentage': area_percentage,
                'training_points': training_points,
                'area_per_training_point': cls_area / training_points if training_points > 0 else 0
            }
        
        # Overall statistics
        analysis_results = {
            'region_analysis': region_analysis,
            'total_area': total_area,
            'resolution': resolution,
            'mesh_size': len(predictions),
            'feature_ranges': {
                feature_names[0]: (x_min, x_max),
                feature_names[1]: (y_min, y_max)
            }
        }
        
        return analysis_results
    
    def print_region_analysis(self, X, y, feature_names=None):
        """Print a detailed analysis of decision regions."""
        analysis = self.analyze_decision_regions(X, y, feature_names)
        
        print("DECISION REGION ANALYSIS")
        print("=" * 50)
        print(f"Total analyzed area: {analysis['total_area']:.2f} square units")
        print(f"Analysis resolution: {analysis['resolution']} points per dimension")
        print(f"Total mesh points: {analysis['mesh_size']}")
        print()
        
        print("REGION BREAKDOWN:")
        print("-" * 30)
        
        for cls, stats in analysis['region_analysis'].items():
            print(f"\nClass: {cls}")
            print(f"  • Predicted area: {stats['area_estimate']:.2f} sq units ({stats['area_percentage']:.1f}%)")
            print(f"  • Training points: {stats['training_points']}")
            print(f"  • Area per training point: {stats['area_per_training_point']:.2f}")
            print(f"  • Mesh points: {stats['predicted_points']}")
        
        print("\nFEATURE RANGES:")
        print("-" * 20)
        for feature, (min_val, max_val) in analysis['feature_ranges'].items():
            print(f"  {feature}: [{min_val:.2f}, {max_val:.2f}] (range: {max_val - min_val:.2f})")
    
    def plot_decision_regions_detailed(self, X, y, feature_names=None, resolution=50,
                                     figsize=(15, 5), save_path=None):
        """
        Create a detailed visualization with multiple subplots.
        
        Parameters:
        -----------
        X : array-like
            2D feature data
        y : array-like
            Target labels
        feature_names : list, optional
            Feature names
        resolution : int, default=50
            Mesh resolution
        figsize : tuple, default=(15, 5)
            Figure size
        save_path : str, optional
            Save path
        """
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns[:2].tolist()
            X = X.values
        else:
            X = np.array(X)
            if feature_names is None:
                feature_names = ['Feature 1', 'Feature 2']
        
        y = np.array(y)
        unique_classes = sorted(list(set(y)))
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Common mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        h = max((x_max - x_min) / resolution, (y_max - y_min) / resolution)
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_df = pd.DataFrame(mesh_points, columns=feature_names)
        Z = self.tree_model.predict(mesh_df)
        
        # Map predictions to numbers
        class_to_num = {cls: i for i, cls in enumerate(unique_classes)}
        Z_numeric = np.array([class_to_num[pred] for pred in Z]).reshape(xx.shape)
        
        colors = self.class_colors[:len(unique_classes)]
        cmap = ListedColormap(colors)
        
        # Plot 1: Decision regions only
        axes[0].contourf(xx, yy, Z_numeric, alpha=0.8, cmap=cmap)
        axes[0].contour(xx, yy, Z_numeric, colors='black', linewidths=1)
        axes[0].set_title('Decision Regions', fontweight='bold')
        axes[0].set_xlabel(feature_names[0])
        axes[0].set_ylabel(feature_names[1])
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Training points only
        for i, cls in enumerate(unique_classes):
            mask = y == cls
            axes[1].scatter(X[mask, 0], X[mask, 1], c=colors[i], s=60, 
                          alpha=0.8, edgecolors='black', linewidth=1, label=cls)
        axes[1].set_title('Training Data Points', fontweight='bold')
        axes[1].set_xlabel(feature_names[0])
        axes[1].set_ylabel(feature_names[1])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Combined view
        axes[2].contourf(xx, yy, Z_numeric, alpha=0.6, cmap=cmap)
        axes[2].contour(xx, yy, Z_numeric, colors='black', linewidths=1.5, alpha=0.8)
        for i, cls in enumerate(unique_classes):
            mask = y == cls
            axes[2].scatter(X[mask, 0], X[mask, 1], c=colors[i], s=60, 
                          alpha=0.9, edgecolors='black', linewidth=1, label=cls)
        axes[2].set_title('Decision Boundary + Data', fontweight='bold')
        axes[2].set_xlabel(feature_names[0])
        axes[2].set_ylabel(feature_names[1])
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Set consistent limits
        for ax in axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        plt.suptitle('Decision Tree - Comprehensive Boundary Analysis', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Detailed boundary plot saved to: {save_path}")
        
        plt.show()
        return fig


def demonstrate_boundary_visualization():
    """Demonstrate decision boundary visualization with sample data."""
    print("Decision Boundary Visualization Demonstration")
    print("=" * 60)
    
    # Generate sample 2D data
    np.random.seed(42)
    n_samples = 100
    
    # Create synthetic dataset with clear decision boundaries
    X = np.random.rand(n_samples, 2) * 10
    y = []
    
    for point in X:
        if point[0] > 5 and point[1] > 5:
            y.append('Class_A')
        elif point[0] < 3:
            y.append('Class_B')  
        else:
            y.append('Class_C')
    
    y = np.array(y)
    
    print(f"Generated {n_samples} samples with 2 features")
    print(f"Class distribution: {dict(Counter(y))}")
    print("\nThis demonstration requires a trained CustomDecisionTree model.")
    print("Use the DecisionBoundaryVisualizer with your trained models.")
    
    # Show available methods
    print("\nAvailable visualization methods:")
    print("1. plot_decision_boundary() - Basic boundary plot")
    print("2. plot_decision_regions_detailed() - Detailed multi-panel view")
    print("3. analyze_decision_regions() - Region analysis")
    print("4. print_region_analysis() - Text-based analysis")


if __name__ == "__main__":
    demonstrate_boundary_visualization()