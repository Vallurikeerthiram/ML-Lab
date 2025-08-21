import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# ---------------------- DATA LOADER ---------------------- #
def load_dataset(file_path, target_col, exclude_cols=None, test_size=0.2, random_state=42):
    """
    Load dataset from Excel file, separate features and target, and split into train/test.
    """
    df = pd.read_excel(file_path)

    if exclude_cols is None:
        exclude_cols = [target_col]

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, feature_cols


# ---------------------- MODEL TUNER ---------------------- #
def tune_model_with_random_search(model, param_grid, X_train, y_train,
                                  cv=5, n_iter=20, scoring="r2", random_state=42):
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    Returns the best model, best parameters, and best cross-validation score.
    """
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    return {
        "best_model": random_search.best_estimator_,
        "best_params": random_search.best_params_,
        "best_cv_score": random_search.best_score_
    }


# ---------------------- MAIN ---------------------- #
if __name__ == "__main__":
    file_path = r"C:\Users\keert\OneDrive - Amrita vishwa vidyapeetham\Amrita\Sem5\ML\Lab6\Alwar.xlsx"
    target_col = "DEP"   # target column
    exclude_cols = [target_col]  # features are all except target

    # Load data
    X_train, X_test, y_train, y_test, features = load_dataset(file_path, target_col, exclude_cols)

    # Define model + param grid
    dt_model = DecisionTreeRegressor(random_state=42)
    param_grid = {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "criterion": ["squared_error", "friedman_mse", "absolute_error"]  # removed poisson
    }


    # Tune model
    results = tune_model_with_random_search(dt_model, param_grid, X_train, y_train)

    # Evaluate on test data
    best_model = results["best_model"]
    y_pred = best_model.predict(X_test)

    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("Best Parameters:", results["best_params"])
    print("Best CV R2 Score:", results["best_cv_score"])
    print("Test R2 Score:", test_r2)
    print("Test RMSE:", test_rmse)
