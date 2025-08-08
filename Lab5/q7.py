import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Function to load and preprocess dataset
def load_dataset(file_path):
    df = pd.read_excel(file_path)
    df = df[['RF', 'DEP']].dropna().reset_index(drop=True)

    # Add time-related features
    df['Month_Index'] = range(1, len(df) + 1)
    start_year = 2018
    df['Year'] = start_year + (df['Month_Index'] - 1) // 12
    df['Month'] = ((df['Month_Index'] - 1) % 12) + 1

    # Remove target variable 'DEP'
    X = df.drop(columns=['DEP'])
    return X

# Function to compute distortions for different k values
def elbow_method(X, k_range):
    distortions = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
        distortions.append(kmeans.inertia_)
    return distortions

# Function to plot elbow curve
def plot_elbow(k_values, distortions):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, distortions, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Distortion (Inertia)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)
    plt.show()

# Main Program
if __name__ == "__main__":
    file_path = r"C:\Users\keert\OneDrive - Amrita vishwa vidyapeetham\Amrita\Sem5\ML\Lab5\Alwar.xlsx"

    X = load_dataset(file_path)
    k_values = range(2, 20)

    distortions = elbow_method(X, k_values)
    plot_elbow(k_values, distortions)

    # Print values for reference
    for k, d in zip(k_values, distortions):
        print(f"k={k}, Distortion={d:.4f}")
