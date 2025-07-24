import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

def train_knn_and_save(X_train, y_train, k=3, model_file='knn_model.pkl'):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    joblib.dump(knn, model_file)

# Main
train_df = pd.read_csv('groundwater_train.csv')
X_train = train_df.drop('Class', axis=1)
y_train = train_df['Class']
train_knn_and_save(X_train, y_train)
print("Model saved as 'knn_model.pkl'")
