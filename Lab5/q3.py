import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np

# Function to load and preprocess data
def load_dataset(file_path):
    df = pd.read_excel(file_path)
    df = df[['RF', 'DEP']].dropna().reset_index(drop=True)

    # Create month index feature (1, 2, ..., n)
    df['Month_Index'] = range(1, len(df) + 1)
    
    # Derive Year and Month from Month_Index if needed
    start_year = 2018
    df['Year'] = start_year + (df['Month_Index'] - 1) // 12
    df['Month'] = ((df['Month_Index'] - 1) % 12) + 1
    
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
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

# Main program
if __name__ == "__main__":
    file_path = r"C:\Users\keert\OneDrive - Amrita vishwa vidyapeetham\Amrita\Sem5\ML\Lab5\Alwar.xlsx"
    
    df = load_dataset(file_path)
    
    # Multiple features: RF + Month_Index + Year + Month
    X = df[['RF', 'Month_Index', 'Year', 'Month']]
    y = df['DEP']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_linear_regression(X_train, y_train)
    
    y_train_pred = make_predictions(model, X_train)
    y_test_pred = make_predictions(model, X_test)
    
    # Metrics
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
    
    print("\nModel Coefficients:", model.coef_)
    print("Model Intercept:", model.intercept_)
