import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Function to load dataset ----
def load_data(file_path):
    df = pd.read_csv(file_path, encoding="latin1")
    return df

# ---- Function to calculate Minkowski distance ----
def minkowski_distance(x, y, r):
    return np.sum(np.abs(x - y) ** r) ** (1 / r)

# ---- Main ----
file_path = "Atal_Jal_Disclosed_Ground_Water_Level-2015-2022.csv"
df = load_data(file_path)

# Pick two feature vectors (rows) and select numeric columns only
numeric_df = df.select_dtypes(include=[np.number]).dropna()

# Take first two rows as example feature vectors
x = numeric_df.iloc[0].values
y = numeric_df.iloc[1].values

distances = []
r_values = range(1, 11)

for r in r_values:
    d = minkowski_distance(x, y, r)
    distances.append(d)

# Print distances
for r, d in zip(r_values, distances):
    print(f"r={r}, Minkowski distance={d:.4f}")

# Plot
plt.plot(r_values, distances, marker="o", linestyle="-", color="blue")
plt.title("Minkowski Distance vs r (1 to 10)")
plt.xlabel("r value")
plt.ylabel("Minkowski Distance")
plt.grid(True)
plt.show()
