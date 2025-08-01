import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_groundwater_data(csv_path, feature1, feature2, target_column):
    """
    Load and preprocess groundwater data for kNN classification.
    Handles non-numeric values like 'Dry', 'Filled up', etc.
    """
    df = pd.read_csv(csv_path, encoding='latin1')

    # Define invalid values
    invalid_values = ['Dry', 'Filled up', 'Not Measured', 'NA', 'N/A', '-', '', None]

    # Remove rows with invalid or missing entries
    for col in [feature1, feature2]:
        df = df[~df[col].isin(invalid_values)]
    df = df.dropna(subset=[feature1, feature2, target_column])

    # Convert to float
    df[feature1] = pd.to_numeric(df[feature1], errors='coerce')
    df[feature2] = pd.to_numeric(df[feature2], errors='coerce')

    # Drop rows where conversion failed
    df = df.dropna(subset=[feature1, feature2])

    return df[[feature1, feature2, target_column]]


def split_data(df, feature1, feature2, target_column, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets.
    """
    X = df[[feature1, feature2]]
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def perform_grid_search(X_train, y_train, param_grid=None):
    """
    Perform hyperparameter tuning using GridSearchCV.
    """
    if param_grid is None:
        param_grid = {'n_neighbors': list(range(1, 21))}

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using accuracy score.
    """
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    # Input Parameters
    csv_path = "Atal_Jal_Disclosed_Ground_Water_Level-2015-2022.csv"  # Change to your actual file
    feature1 = "Pre-monsoon_2015 (meters below ground level)"
    feature2 = "Post-monsoon_2022 (meters below ground level)"
    target_column = "Aquifer"  # You can change based on classification goal

    # Load and split the data
    df = load_groundwater_data(csv_path, feature1, feature2, target_column)
    X_train, X_test, y_train, y_test = split_data(df, feature1, feature2, target_column)

    # Perform Grid Search to find best k
    best_model, best_params, best_score = perform_grid_search(X_train, y_train)

    # Evaluate model on test data
    test_accuracy = evaluate_model(best_model, X_test, y_test)

    # Print results
    print(f"Best Parameters (k): {best_params}")
    print(f"Best Cross-validation Accuracy: {best_score:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
