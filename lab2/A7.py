import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def load_first_20_vectors(filepath, sheet_name):
    """Load first 20 rows from the given worksheet."""
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df.head(20)

def preprocess_binary(df):
    """Convert categorical binary-like columns into 0/1 form."""
    new_df = df.copy()
    for col in df.columns:
        if new_df[col].dtype == 'object':
            unique_vals = set(new_df[col].dropna().unique())
            if unique_vals <= {"t", "f"}:
                new_df[col] = new_df[col].map({"t": 1, "f": 0})
            elif unique_vals <= {"y", "n"}:
                new_df[col] = new_df[col].map({"y": 1, "n": 0})
            elif unique_vals <= {"yes", "no"}:
                new_df[col] = new_df[col].map({"yes": 1, "no": 0})
    return new_df

def get_binary_columns(df):
    """Return only binary columns (0/1 values)."""
    return [col for col in df.columns if df[col].dropna().isin([0, 1]).all()]

def compute_jc_smc(df):
    """Compute pairwise JC and SMC similarity for binary columns only."""
    binary_cols = get_binary_columns(df)
    if not binary_cols:
        print("⚠️ No binary columns found after preprocessing!")
        return None, None

    binary_data = df[binary_cols]
    n = binary_data.shape[0]
    jc_matrix = np.zeros((n, n))
    smc_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            vec1 = binary_data.iloc[i]
            vec2 = binary_data.iloc[j]
            f11 = ((vec1 == 1) & (vec2 == 1)).sum()
            f00 = ((vec1 == 0) & (vec2 == 0)).sum()
            f10 = ((vec1 == 1) & (vec2 == 0)).sum()
            f01 = ((vec1 == 0) & (vec2 == 1)).sum()
            jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) > 0 else 0
            smc = (f11 + f00) / (f11 + f00 + f10 + f01) if (f11 + f00 + f10 + f01) > 0 else 0
            jc_matrix[i][j] = jc
            smc_matrix[i][j] = smc

    return jc_matrix, smc_matrix

def compute_cosine(df):
    """Compute cosine similarity on numeric data."""
    numeric_df = df.select_dtypes(include=[np.number])
    return cosine_similarity(numeric_df)

def plot_heatmap(matrix, title):
    """Display a heatmap for the given similarity matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, cmap='coolwarm', square=True)
    plt.title(title)
    plt.xlabel("Observation Index")
    plt.ylabel("Observation Index")
    plt.show()

def A7():
    filepath = "Lab Session Data.xlsx"
    sheet_name = "thyroid0387_UCI"

    # Load and preprocess
    df20 = load_first_20_vectors(filepath, sheet_name)
    df20 = preprocess_binary(df20)

    print("Calculating JC and SMC on binary features...")
    jc_matrix, smc_matrix = compute_jc_smc(df20)

    if jc_matrix is not None and smc_matrix is not None:
        plot_heatmap(jc_matrix, "Jaccard Coefficient Heatmap (First 20 Observations)")
        plot_heatmap(smc_matrix, "Simple Matching Coefficient Heatmap (First 20 Observations)")

    print("Calculating Cosine similarity on full numeric data...")
    cos_matrix = compute_cosine(df20)
    plot_heatmap(cos_matrix, "Cosine Similarity Heatmap (First 20 Observations)")

if __name__ == "__main__":
    A7()
