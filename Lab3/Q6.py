import pandas as pd
import joblib

def evaluate_model_accuracy(X_test, y_test, model_file='knn_model.pkl'):
    model = joblib.load(model_file)
    return model.score(X_test, y_test)

# Main
test_df = pd.read_csv('groundwater_test.csv')
X_test = test_df.drop('Class', axis=1)
y_test = test_df['Class']
accuracy = evaluate_model_accuracy(X_test, y_test)
print(f"Accuracy of kNN (k=3): {round(accuracy * 100, 2)}%")
