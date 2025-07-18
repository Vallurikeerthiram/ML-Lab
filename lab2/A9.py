import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def load_data(filepath, sheet_name):
    return pd.read_excel(filepath, sheet_name=sheet_name)

def identify_scaling_columns(df):
    """Return a list of numeric columns (excluding binary and categorical)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    scaling_cols = []
    for col in numeric_cols:
        unique_vals = df[col].nunique()
        if unique_vals > 2:  # Exclude binary
            scaling_cols.append(col)
    return scaling_cols

def has_outliers(series):
    """Detect outliers using IQR."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).any()

def normalize_columns(df, columns):
    """Apply appropriate scaling techniques based on outliers."""
    df_scaled = df.copy()
    methods_used = {}

    for col in columns:
        series = df[col].dropna()

        if has_outliers(series):
            scaler = RobustScaler()
            methods_used[col] = "Robust Scaling (handles outliers)"
        else:
            scaler = MinMaxScaler()
            methods_used[col] = "Min-Max Scaling (0–1 range)"

        reshaped = df[[col]].values
        scaled_values = scaler.fit_transform(reshaped)
        df_scaled[col] = scaled_values

    return df_scaled, methods_used

def A9():
    filepath = "Lab Session Data.xlsx"
    sheet_name = "thyroid0387_UCI"

    df = load_data(filepath, sheet_name)

    scaling_cols = identify_scaling_columns(df)
    print(" Columns selected for normalization:", scaling_cols)

    df_scaled, methods = normalize_columns(df, scaling_cols)

    print("\nNormalization Methods Used:")
    for col, method in methods.items():
        print(f"{col}: {method}")


    return df_scaled

if __name__ == "__main__":
    normalized_df = A9()
