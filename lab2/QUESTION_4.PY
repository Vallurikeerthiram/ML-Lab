import pandas as pd

def load_excel(path, sheet):
    return pd.read_excel(path, sheet_name=sheet)

def analyze_attributes(df):
    print("\nAttributes Overview:")
    for col in df.columns:
        print(f"{col} - Type: {df[col].dtype}, Unique: {df[col].nunique()}")

def suggest_encoding(df):
    print("\nEncoding Suggestion:")
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"{col}: One-Hot Encoding")

def numeric_stats(df):
    print("\nNumeric Attributes:")
    numeric = df.select_dtypes(include='number')
    for col in numeric.columns:
        stats = df[col].dropna()
        print(f"{col}: Min={stats.min()}, Max={stats.max()}, Mean={stats.mean():.2f}, Std={stats.std():.2f}")

def check_missing(df):
    print("\nMissing Values:")
    print(df.isnull().sum())

def detect_outliers(df):
    print("\nOutlier Detection (IQR):")
    numeric = df.select_dtypes(include='number')
    for col in numeric.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        print(f"{col}: {len(outliers)} outliers")

# Main execution
df_data = load_excel("Lab Session Data", "thyroid0387_UCI")
analyze_attributes(df_data)
suggest_encoding(df_data)
numeric_stats(df_data)
check_missing(df_data)
detect_outliers(df_data)
