import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------- DATA LOADER ---------------------- #
def load_dataset(file_path, feature_col="RF", target_col="DEP", test_size=0.2, random_state=42):
    df = pd.read_excel(file_path)
    X = df[[feature_col]].fillna(0)   # Features
    y = df[target_col].fillna(0)      # Target
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# ---------------------- EVALUATOR ---------------------- #
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }

# ---------------------- MAIN ---------------------- #
if __name__ == "__main__":
    file_path = r"C:\Users\keert\OneDrive - Amrita vishwa vidyapeetham\Amrita\Sem5\ML\Lab6\Alwar.xlsx"
    X_train, X_test, y_train, y_test = load_dataset(file_path)

    models = {
        "Linear Regression": LinearRegression(),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(random_state=42, n_estimators=100),
        "SVR": SVR(kernel="rbf")
    }

    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)

    results_df = pd.DataFrame(results).T
    print("\n📊 Regression Results on Alwar.xlsx\n")
    print(results_df)
    results_df.to_excel("regression_results.xlsx", index=True)  # Save results if needed
