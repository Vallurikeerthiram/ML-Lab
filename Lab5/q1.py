import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Function to load and preprocess data
def load_dataset(file_path):
    df = pd.read_excel(file_path)
    # Assuming 'RF' and 'DEP' are numeric already
    df = df[['RF', 'DEP']].dropna()  # Keep only RF and DEP, remove NaNs
    return df

# Function to train linear regression
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to make predictions
def make_predictions(model, X):
    return model.predict(X)

# Main program
if __name__ == "__main__":
    file_path = r"C:\Users\keert\OneDrive - Amrita vishwa vidyapeetham\Amrita\Sem5\ML\Lab5\Alwar.xlsx"
    
    df = load_dataset(file_path)
    
    # Feature: RF, Target: DEP
    X = df[['RF']]
    y = df['DEP']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_linear_regression(X_train, y_train)
    y_train_pred = make_predictions(model, X_train)
    y_test_pred = make_predictions(model, X_test)
    
    print("Training Predictions:", y_train_pred)
    print("Test Predictions:", y_test_pred)
    print("Model Coefficients:", model.coef_)
    print("Model Intercept:", model.intercept_)

"Goal: We used the RF (Rainfall) column as the single predictor (X_train) and the DEP (Depth) column as the target (y_train) to train a Linear Regression model."
"Coefficient (0.2979): This indicates that for every 1 mm increase in rainfall, the groundwater depth is predicted to rise by approximately 0.2979 units, assuming other factors remain constant."

"Intercept (-22.34): If the rainfall value is zero, the model predicts a depth of about –22.34. This negative value has no direct physical meaning but represents the model’s baseline offset."
