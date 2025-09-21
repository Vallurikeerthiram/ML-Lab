# A11: AND and XOR Gates using Scikit-Learn MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, LeaveOneOut
import warnings
warnings.filterwarnings('ignore')

def test_logic_gate_fixed(X, y, gate_name):
    """Test different MLP configurations for a logic gate with proper CV"""

    print(f"\n{'='*60}")
    print(f"TESTING {gate_name} GATE WITH MLPClassifier")
    print(f"{'='*60}")

    print(f"\nTraining data for {gate_name}:")
    for i in range(len(X)):
        print(f"  Input: {X[i]}, Output: {y[i]}")

    # Different MLP configurations to test
    configurations = [
        {
            'name': 'Single Hidden Layer (4 neurons)',
            'params': {
                'hidden_layer_sizes': (4,),
                'activation': 'logistic',
                'learning_rate_init': 0.1,
                'max_iter': 1000,
                'random_state': 42
            }
        },
        {
            'name': 'Two Hidden Layers (8,4)',
            'params': {
                'hidden_layer_sizes': (8, 4),
                'activation': 'logistic',
                'learning_rate_init': 0.1,
                'max_iter': 1000,
                'random_state': 42
            }
        },
        {
            'name': 'ReLU Activation (4 neurons)',
            'params': {
                'hidden_layer_sizes': (4,),
                'activation': 'relu',
                'learning_rate_init': 0.1,
                'max_iter': 1000,
                'random_state': 42
            }
        }
    ]

    results = {}

    for config in configurations:
        print(f"\nTesting {config['name']}...")

        # Create and train MLP
        mlp = MLPClassifier(**config['params'])
        mlp.fit(X, y)

        # Make predictions
        y_pred = mlp.predict(X)

        # Calculate accuracy
        accuracy = accuracy_score(y, y_pred) * 100

        print(f"  Iterations to converge: {mlp.n_iter_}")
        print(f"  Training accuracy: {accuracy:.1f}%")

        # Detailed results
        for i in range(len(X)):
            print(f"    Input: {X[i]}, Target: {y[i]}, Predicted: {y_pred[i]}")

        results[config['name']] = {
            'model': mlp,
            'accuracy': accuracy,
            'iterations': mlp.n_iter_,
            'predictions': y_pred,
            'loss_curve': mlp.loss_curve_
        }

    # Grid search with Leave-One-Out CV for small dataset
    print(f"\nPerforming grid search with Leave-One-Out CV...")

    param_grid = {
        'hidden_layer_sizes': [(4,), (8,), (4, 2)],
        'activation': ['logistic', 'relu'],
        'learning_rate_init': [0.1, 0.5]
    }

    grid_mlp = MLPClassifier(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(grid_mlp, param_grid, cv=LeaveOneOut(), scoring='accuracy')
    grid_search.fit(X, y)

    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV score: {grid_search.best_score_:.3f}")

    results['Grid Search Best'] = {
        'model': grid_search.best_estimator_,
        'accuracy': grid_search.best_estimator_.score(X, y) * 100,
        'iterations': grid_search.best_estimator_.n_iter_,
        'predictions': grid_search.best_estimator_.predict(X),
        'loss_curve': grid_search.best_estimator_.loss_curve_,
        'params': grid_search.best_params_
    }

    return results

def main():
    print("=" * 80)
    print("A11: AND AND XOR GATES USING SCIKIT-LEARN MLPClassifier")
    print("=" * 80)

    # Define logic gates data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # AND gate
    y_and = np.array([0, 0, 0, 1])

    # XOR gate  
    y_xor = np.array([0, 1, 1, 0])

    # Test both gates
    and_results = test_logic_gate_fixed(X, y_and, "AND")
    xor_results = test_logic_gate_fixed(X, y_xor, "XOR")

    # Final comparison
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("=" * 80)

    print("\nAND GATE RESULTS:")
    print("-" * 40)
    print(f"{'Configuration':<25} {'Accuracy':<10} {'Iterations':<12}")
    print("-" * 50)
    for name, result in and_results.items():
        config_name = name.split('(')[0] if '(' in name else name
        print(f"{config_name:<25} {result['accuracy']:<10.1f} {result['iterations']:<12}")

    print("\nXOR GATE RESULTS:")
    print("-" * 40)
    print(f"{'Configuration':<25} {'Accuracy':<10} {'Iterations':<12}")
    print("-" * 50)
    for name, result in xor_results.items():
        config_name = name.split('(')[0] if '(' in name else name  
        print(f"{config_name:<25} {result['accuracy']:<10.1f} {result['iterations']:<12}")

    # Best models
    best_and_name = max(and_results.keys(), key=lambda k: and_results[k]['accuracy'])
    best_xor_name = max(xor_results.keys(), key=lambda k: xor_results[k]['accuracy'])

    print("\nBEST PERFORMING MODELS:")
    print("-" * 30)
    print(f"AND Gate: {best_and_name}")
    print(f"  Accuracy: {and_results[best_and_name]['accuracy']:.1f}%")
    print(f"  Iterations: {and_results[best_and_name]['iterations']}")

    print(f"\nXOR Gate: {best_xor_name}")
    print(f"  Accuracy: {xor_results[best_xor_name]['accuracy']:.1f}%")
    print(f"  Iterations: {xor_results[best_xor_name]['iterations']}")

    print("\nKEY OBSERVATIONS:")
    print("-" * 20)
    print("  • AND Gate: Linearly separable, most configurations achieve 100%")
    print("  • XOR Gate: Non-linearly separable, requires hidden layers") 
    print("  • Sigmoid (logistic) activation works well for both")
    print("  • XOR generally requires more iterations to converge")
    print("  • Grid search with Leave-One-Out CV works for small datasets")
    print("  • MLPClassifier provides easy-to-use neural networks")

if __name__ == "__main__":
    main()
