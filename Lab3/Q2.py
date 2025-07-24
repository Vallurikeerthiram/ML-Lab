import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_clean_feature(file_path, feature_column):
    df = pd.read_csv(file_path, encoding='latin1')
    data = pd.to_numeric(df[feature_column], errors='coerce').dropna()
    return data

def analyze_distribution(data, bins=10):
    counts, bin_edges = np.histogram(data, bins=bins)
    return counts, bin_edges, np.mean(data), np.var(data)

def plot_histogram(data, feature_column, bins=10):
    plt.hist(data, bins=bins, color='lightblue', edgecolor='black')
    plt.title(f"Histogram of {feature_column}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# Main
file_path = 'Atal_Jal_Disclosed_Ground_Water_Level-2015-2022.csv'
feature_column = 'Pre-monsoon_2022 (meters below ground level)'
data = load_and_clean_feature(file_path, feature_column)
counts, bin_edges, mean_val, var_val = analyze_distribution(data)

print("Histogram counts per bucket:", counts)
print("Bin ranges:", bin_edges)
print(f"Mean: {mean_val:.2f}, Variance: {var_val:.2f}")
plot_histogram(data, feature_column)
