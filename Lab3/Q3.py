import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_vectors(file_path, feature_columns, index1=0, index2=1):
    df = pd.read_csv(file_path)
    df = df[feature_columns].dropna().reset_index(drop=True)
    return df.loc[index1].values, df.loc[index2].values

def compute_minkowski_distances(x, y, r_range=range(1, 11)):
    distances = [np.sum(np.abs(x - y) ** r) ** (1 / r) for r in r_range]
    return r_range, distances

def plot_minkowski(r_vals, distances):
    plt.plot(r_vals, distances, marker='o')
    plt.title("Minkowski Distance vs r")
    plt.xlabel("r")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()

# Main
file_path = "Atal_Jal_Disclosed_Ground_Water_Level-2015-2022.csv"
feature_columns = ['GWL (in Mtr)', 'Latitude', 'Longitude']
x, y = load_vectors(file_path, feature_columns)
r_vals, distances = compute_minkowski_distances(x, y)
plot_minkowski(r_vals, distances)
