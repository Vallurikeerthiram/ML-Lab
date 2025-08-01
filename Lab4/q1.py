import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y

# Train and return the model
def train_knn_model(X_train, y_train, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

# Evaluate predictions and return metrics
def evaluate_classification(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return cm, precision, recall, f1

# Main program
data_path = 'groundwater_train.csv'
X, y = load_data(data_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = train_knn_model(X_train, y_train, k=3)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluation
cm_train, precision_train, recall_train, f1_train = evaluate_classification(y_train, y_train_pred)
cm_test, precision_test, recall_test, f1_test = evaluate_classification(y_test, y_test_pred)

# Print results
print("Train Confusion Matrix:\n", cm_train)
print(f"Train Precision: {precision_train:.2f}, Recall: {recall_train:.2f}, F1-Score: {f1_train:.2f}")

print("\nTest Confusion Matrix:\n", cm_test)
print(f"Test Precision: {precision_test:.2f}, Recall: {recall_test:.2f}, F1-Score: {f1_test:.2f}")

# Interpretation
if f1_train > f1_test + 0.1:
    print("\nObservation: Model is overfitting (high train, low test performance).")
elif f1_test > f1_train + 0.1:
    print("\nObservation: Model is underfitting (low train, higher test performance).")
else:
    print("\nObservation: Model is fitting well (regular fit).")
