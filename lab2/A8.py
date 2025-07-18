import pandas as pd
import numpy as np

def load_data(filepath, sheet_name):
    return pd.read_excel(filepath, sheet_name=sheet_name)

def has_outliers(series):
    """Check if a numeric series has outliers using IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((series < lower_bound) | (series > upper_bound)).any()

def impute_data(df):
    df_imputed = df.copy()

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue

        if df[col].dtype == 'object':
            mode_val = df[col].mode().iloc[0]
            df_imputed[col].fillna(mode_val, inplace=True)

        elif np.issubdtype(df[col].dtype, np.number):
            if has_outliers(df[col].dropna()):
                median_val = df[col].median()
                df_imputed[col].fillna(median_val, inplace=True)
                print(f"{col}: Imputed with Median (due to outliers)")
            else:
                mean_val = df[col].mean()
                df_imputed[col].fillna(mean_val, inplace=True)
                print(f"{col}: Imputed with Mean (no outliers)")

    return df_imputed

def A8():
    filepath = "Lab Session Data.xlsx"
    sheet_name = "thyroid0387_UCI"
    
    df = load_data(filepath, sheet_name)
    print("🔍 Missing values before imputation:\n", df.isnull().sum())

    df_imputed = impute_data(df)

    print("\n✅ Missing values after imputation:\n", df_imputed.isnull().sum())

if __name__ == "__main__":
    A8()
