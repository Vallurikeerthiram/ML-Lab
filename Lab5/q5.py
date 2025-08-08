import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Function to load and preprocess data (without target column)
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

# Function to perform K-Means clustering
def perform_kmeans(X, k=2):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    return kmeans

# Function to calculate clustering metrics
def calculate_cluster_metrics(X, labels):
    sil_score = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_index = davies_bouldin_score(X, labels)
    return sil_score, ch_score, db_index

# Main program
if __name__ == "__main__":
    file_path = r"C:\Users\keert\OneDrive - Amrita vishwa vidyapeetham\Amrita\Sem5\ML\Lab5\Alwar.xlsx"

    X = load_dataset_for_clustering(file_path)
    kmeans_model = perform_kmeans(X, k=2)
    
    sil, ch, db = calculate_cluster_metrics(X, kmeans_model.labels_)

    print("Cluster Labels:", kmeans_model.labels_)
    print("Cluster Centers:\n", kmeans_model.cluster_centers_)
    print("\n--- Clustering Evaluation Metrics ---")
    print(f"Silhouette Score:       {sil:.4f}")
    print(f"Calinski-Harabasz Score:{ch:.4f}")
    print(f"Davies-Bouldin Index:   {db:.4f}")
