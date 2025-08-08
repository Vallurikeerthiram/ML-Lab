import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Function to load and preprocess data (without target variable)
def load_dataset_for_clustering(file_path):
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

# Function to perform k-means and calculate metrics
def evaluate_kmeans_for_k_values(X, k_values):
    metrics = {"k": [], "Silhouette": [], "CH_Score": [], "DB_Index": []}

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)

        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)

        metrics["k"].append(k)
        metrics["Silhouette"].append(sil)
        metrics["CH_Score"].append(ch)
        metrics["DB_Index"].append(db)

    return metrics

# Function to plot metrics
def plot_metrics(metrics):
    plt.figure(figsize=(14, 4))

    # Silhouette Score
    plt.subplot(1, 3, 1)
    plt.plot(metrics["k"], metrics["Silhouette"], marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs k")

    # Calinski-Harabasz Score
    plt.subplot(1, 3, 2)
    plt.plot(metrics["k"], metrics["CH_Score"], marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Calinski-Harabasz Score")
    plt.title("CH Score vs k")

    # Davies-Bouldin Index
    plt.subplot(1, 3, 3)
    plt.plot(metrics["k"], metrics["DB_Index"], marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Davies-Bouldin Index")
    plt.title("DB Index vs k")

    plt.tight_layout()
    plt.show()

# Main Program
if __name__ == "__main__":
    file_path = r"C:\Users\keert\OneDrive - Amrita vishwa vidyapeetham\Amrita\Sem5\ML\Lab5\Alwar.xlsx"

    X = load_dataset_for_clustering(file_path)
    k_values = range(2, 10)  # Testing k from 2 to 9

    metrics = evaluate_kmeans_for_k_values(X, k_values)
    plot_metrics(metrics)

    # Print metrics table
    print("\n--- K-Means Clustering Evaluation ---")
    for i in range(len(metrics["k"])):
        print(f"k={metrics['k'][i]}: "
              f"Silhouette={metrics['Silhouette'][i]:.4f}, "
              f"CH_Score={metrics['CH_Score'][i]:.4f}, "
              f"DB_Index={metrics['DB_Index'][i]:.4f}")
