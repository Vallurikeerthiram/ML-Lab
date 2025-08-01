import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# -------------------- Function Definitions --------------------

def load_purchase_data(file_path):
    """
    Load the purchase data from the Excel file.
    Returns the feature matrix A and actual payment vector C.
    """
    df = pd.read_excel(file_path)  # Reads the first/default sheet
    feature_matrix = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
    actual_payment = df['Payment (Rs)'].values
    return feature_matrix, actual_payment

def estimate_prices(feature_matrix, actual_payment):
    """
    Estimate product prices using pseudo-inverse of the matrix.
    Returns the estimated price vector and predicted payments.
    """
    pseudo_inverse = np.linalg.pinv(feature_matrix)
    estimated_prices = pseudo_inverse @ actual_payment
    predicted_payment = feature_matrix @ estimated_prices
    return estimated_prices, predicted_payment

def evaluate_regression_metrics(actual_payment, predicted_payment):
    """
    Compute regression metrics: MSE, RMSE, MAPE, R².
    Returns them as a dictionary.
    """
    mse = mean_squared_error(actual_payment, predicted_payment)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_payment - predicted_payment) / actual_payment)) * 100
    r2 = r2_score(actual_payment, predicted_payment)
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

# -------------------- Main Program --------------------

# Set the file path
file_path = r"C:\Users\keert\OneDrive - Amrita vishwa vidyapeetham\Amrita\Sem5\ML\Lab4\Lab Session Data (1).xlsx"

# Load features and labels
A, C_actual = load_purchase_data(file_path)

# Estimate prices and predict payments
X_estimated, C_pred = estimate_prices(A, C_actual)

# Evaluate model
metrics = evaluate_regression_metrics(C_actual, C_pred)

# Print outputs
print(f"Estimated Prices: {X_estimated}")
print(f"MSE: {metrics['MSE']:.2f}") #Tells how wrong your predictions are on average. The lower, the better. Big errors get punished more because it squares them. If MSE is 0, predictions are exactly right.
print(f"RMSE: {metrics['RMSE']:.2f}") #It's just the square root of MSE. Tells you the average error in the same unit as your output (like ₹ or steps). Easier to understand than MSE.
print(f"MAPE: {metrics['MAPE']:.2f}%") #Shows how far off your predictions are in percent. Example: MAPE = 5% means you are wrong by 5% on average. Smaller % = better model.
print(f"R² Score: {metrics['R2']:.4f}") #Tells how well your model fits the data. if 1 Perfect predictions if 0 useless