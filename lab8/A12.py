# A12: Project Dataset Analysis using MLPClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

def load_and_prepare_data():
    """Load and prepare the datasets"""

    # Load Alwar data (continuous monthly data)
    print("Loading Alwar dataset...")
    alwar_data = pd.read_excel('Alwar.xlsx')
    print(f"Alwar data shape: {alwar_data.shape}")
    print(f"Alwar columns: {alwar_data.columns.tolist()}")
    print(f"Alwar head:\n{alwar_data.head()}")

    # Load Rajasthan data (multiple districts, multiple years with monthly breakdown)
    print("\nLoading Rajasthan dataset...")
    rajasthan_data = pd.read_excel('rajasthan.xlsx')
    print(f"Rajasthan data shape: {rajasthan_data.shape}")
    print(f"Rajasthan columns: {rajasthan_data.columns.tolist()}")
    print(f"Rajasthan head:\n{rajasthan_data.head()}")

    return alwar_data, rajasthan_data

def analyze_alwar_data(alwar_data):
    """Analyze Alwar continuous monthly data"""
    print("\n" + "=" * 60)
    print("ALWAR DATA ANALYSIS")
    print("=" * 60)

    # Assume RF is rainfall and DEP is departure from normal
    # Create classification based on departure values

    # Create features and target
    X = alwar_data[['RF']].values  # Rainfall as feature

    # Create target classes based on departure values
    # Positive departure = Above normal (1), Negative = Below normal (0)
    y = (alwar_data['DEP'] > 0).astype(int)

    print(f"Features shape: {X.shape}")
    print(f"Target distribution:")
    print(f"  Below Normal (0): {np.sum(y == 0)} samples")
    print(f"  Above Normal (1): {np.sum(y == 1)} samples")

    # Train MLP Classifier
    print("\nTraining MLP Classifier...")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train multiple MLP configurations
    mlp_configs = {
        'Small': MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42),
        'Medium': MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=1000, random_state=42),
        'Large': MLPClassifier(hidden_layer_sizes=(50, 20, 10), max_iter=1000, random_state=42)
    }

    results = {}

    for name, mlp in mlp_configs.items():
        print(f"\nTesting {name} MLP configuration...")

        # Train
        mlp.fit(X_train, y_train)

        # Predict
        y_pred = mlp.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(mlp, X_scaled, y, cv=5)

        results[name] = {
            'model': mlp,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_test': y_test
        }

        print(f"  Test Accuracy: {accuracy:.3f}")
        print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"  Iterations to converge: {mlp.n_iter_}")

    return results, X_scaled, y, scaler

def analyze_rajasthan_data(rajasthan_data):
    """Analyze Rajasthan multi-district, multi-year data"""
    print("\n" + "=" * 60)
    print("RAJASTHAN DATA ANALYSIS")
    print("=" * 60)

    # Extract features from multiple months and years
    # Select rainfall columns (R/F columns)
    rf_columns = [col for col in rajasthan_data.columns if 'R/F' in col]
    dep_columns = [col for col in rajasthan_data.columns if '%DEP' in col]

    print(f"Found {len(rf_columns)} rainfall columns")
    print(f"Found {len(dep_columns)} departure columns")

    # Prepare features (rainfall data)
    X = rajasthan_data[rf_columns[:12]].values  # Use first 12 months as features

    # Create target based on average annual departure
    annual_departures = rajasthan_data[dep_columns[:12]].values
    avg_departure = np.mean(annual_departures, axis=1)

    # Create binary classification: Above/Below normal rainfall
    y = (avg_departure > 0).astype(int)

    print(f"Features shape: {X.shape}")
    print(f"Target distribution:")
    print(f"  Below Normal (0): {np.sum(y == 0)} samples")
    print(f"  Above Normal (1): {np.sum(y == 1)} samples")

    print(f"Districts included:")
    for i, district in enumerate(rajasthan_data['District'].unique()):
        print(f"  {i+1}. {district}")

    # Handle missing values
    X = np.nan_to_num(X, nan=0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train MLP with different configurations
    print("\nTraining MLP Classifiers...")

    mlp_configs = {
        'Basic': MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000, random_state=42),
        'Deep': MLPClassifier(hidden_layer_sizes=(50, 25, 10), max_iter=1000, random_state=42),
        'Wide': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }

    results = {}

    for name, mlp in mlp_configs.items():
        print(f"\nTesting {name} MLP configuration...")

        # Train
        mlp.fit(X_train, y_train)

        # Predict
        y_pred = mlp.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(mlp, X_scaled, y, cv=5)

        results[name] = {
            'model': mlp,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_test': y_test
        }

        print(f"  Test Accuracy: {accuracy:.3f}")
        print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"  Iterations to converge: {mlp.n_iter_}")

    return results, X_scaled, y, scaler, rajasthan_data

def main():
    print("=" * 80)
    print("A12: PROJECT DATASET ANALYSIS USING MLPClassifier")
    print("=" * 80)

    try:
        # Load data
        alwar_data, rajasthan_data = load_and_prepare_data()

        # Analyze Alwar data
        alwar_results, alwar_X, alwar_y, alwar_scaler = analyze_alwar_data(alwar_data)

        # Analyze Rajasthan data
        rajasthan_results, raj_X, raj_y, raj_scaler, rajasthan_data = analyze_rajasthan_data(rajasthan_data)

        print("\n" + "=" * 80)
        print("FINAL ANALYSIS SUMMARY")
        print("=" * 80)

        print("\nALWAR DATASET RESULTS:")
        print("-" * 30)
        for name, result in alwar_results.items():
            print(f"{name} MLP:")
            print(f"  Test Accuracy: {result['accuracy']:.3f}")
            print(f"  CV Accuracy: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")
            print(f"  Iterations: {result['model'].n_iter_}")

        print("\nRAJASTHAN DATASET RESULTS:")
        print("-" * 30)
        for name, result in rajasthan_results.items():
            print(f"{name} MLP:")
            print(f"  Test Accuracy: {result['accuracy']:.3f}")
            print(f"  CV Accuracy: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")
            print(f"  Iterations: {result['model'].n_iter_}")

        # Best models
        best_alwar = max(alwar_results.keys(), key=lambda k: alwar_results[k]['cv_mean'])
        best_raj = max(rajasthan_results.keys(), key=lambda k: rajasthan_results[k]['cv_mean'])

        print(f"\nBEST MODELS:")
        print(f"  Alwar: {best_alwar} (CV Accuracy: {alwar_results[best_alwar]['cv_mean']:.3f})")
        print(f"  Rajasthan: {best_raj} (CV Accuracy: {rajasthan_results[best_raj]['cv_mean']:.3f})")

    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nCreating synthetic example for demonstration...")

        # Create synthetic data for demonstration
        np.random.seed(42)
        n_samples = 100

        # Synthetic Alwar-like data
        X_alwar = np.random.normal(50, 20, (n_samples, 1))
        y_alwar = (X_alwar[:, 0] > 45).astype(int)

        # Synthetic Rajasthan-like data
        X_raj = np.random.normal(30, 15, (n_samples, 12))  # 12 months
        y_raj = (np.mean(X_raj, axis=1) > 30).astype(int)

        print("Using synthetic data for demonstration...")

        # Train MLPs on synthetic data
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # Alwar synthetic
        X_train, X_test, y_train, y_test = train_test_split(X_alwar, y_alwar, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        mlp = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=1000, random_state=42)
        mlp.fit(X_train_scaled, y_train)
        accuracy = mlp.score(X_test_scaled, y_test)

        print(f"Synthetic Alwar MLP Accuracy: {accuracy:.3f}")

        # Rajasthan synthetic
        X_train, X_test, y_train, y_test = train_test_split(X_raj, y_raj, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        mlp = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
        mlp.fit(X_train_scaled, y_train)
        accuracy = mlp.score(X_test_scaled, y_test)

        print(f"Synthetic Rajasthan MLP Accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    main()
