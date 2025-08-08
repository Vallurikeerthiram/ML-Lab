import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np

# Function to load and preprocess data
def load_dataset(file_path):
    df = pd.read_excel(file_path)
    df = df[['RF', 'DEP']].dropna()
    return df

# Function to train linear regression
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to make predictions
def make_predictions(model, X):
    return model.predict(X)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # in %
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

# Main program
if __name__ == "__main__":
    file_path = r"C:\Users\keert\OneDrive - Amrita vishwa vidyapeetham\Amrita\Sem5\ML\Lab5\Alwar.xlsx"
    
    df = load_dataset(file_path)
    
    X = df[['RF']]
    y = df['DEP']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_linear_regression(X_train, y_train)
    
    y_train_pred = make_predictions(model, X_train)
    y_test_pred = make_predictions(model, X_test)
    
    # Calculate metrics
    train_mse, train_rmse, train_mape, train_r2 = calculate_metrics(y_train, y_train_pred)
    test_mse, test_rmse, test_mape, test_r2 = calculate_metrics(y_test, y_test_pred)
    
    print("---- Training Set ----")
    print(f"MSE:  {train_mse:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"MAPE: {train_mape:.2f}%")
    print(f"R²:   {train_r2:.4f}")
    
    print("\n---- Test Set ----")
    print(f"MSE:  {test_mse:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAPE: {test_mape:.2f}%")
    print(f"R²:   {test_r2:.4f}")

##R² score:
##Training R² = 0.0436 → The model explains only 4.36% of the variation in DEP from RF in the training data.

##Test R² = -0.1111 → Negative value means the model performs worse than simply predicting the mean DEP for all test points.

##Error Metrics:

##RMSE on the test set (406.50) is much larger than the training RMSE (94.64), showing that the model generalizes poorly.

##MAPE values are very high (>100%), indicating predictions are often far from the actual values in relative terms.

##Possible Causes:

##Rainfall alone is insufficient to accurately predict groundwater depth; other factors like seasonality, soil type, extraction rate, and recharge time likely have strong influence.

## relationship between RF and DEP is not strictly linear, so a simple linear regression cannot capture it well.

##Model Fit Conclusion:

##The model is underfitting — both train and test accuracy are low, meaning the model fails to learn the relationship well, even on the training data.

##To improve performance, we could:

##Add more relevant predictors (month, year, soil parameters, pumping data, etc.).

##Try polynomial regression or non-linear models.

##Use feature engineering to represent seasonal trends.

