import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def load_first_20_vectors(filepath, sheet_name):
    """Load first 20 rows from the given worksheet."""
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df.head(20)

def get_binary_columns(df):
    """Return only binary columns (0 or 1 values)."""
    return [col for col in df.columns if df[col].dropna().isin([0, 1]).all()]

def compute_jc_smc(df):
    """Compute pairwise JC and SMC similarity for binary columns only."""
    binary_cols = get_binary_columns(df)
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
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title(title)
    plt.xlabel("Observation Index")
    plt.ylabel("Observation Index")
    plt.show()

def A7():
    filepath = "Lab Session Data.xlsx"
    sheet_name = "thyroid0387_UCI"

    df20 = load_first_20_vectors(filepath, sheet_name)

    print(" Calculating JC and SMC on binary features...")
    jc_matrix, smc_matrix = compute_jc_smc(df20)

    print(" Calculating Cosine similarity on full numeric data...")
    cos_matrix = compute_cosine(df20)

    print(" Plotting heatmaps...")
    plot_heatmap(jc_matrix, "Jaccard Coefficient Heatmap (First 20 Observations)")
    plot_heatmap(smc_matrix, "Simple Matching Coefficient Heatmap (First 20 Observations)")
    plot_heatmap(cos_matrix, "Cosine Similarity Heatmap (First 20 Observations)")

if __name__ == "__main__":
    A7()
