import pandas as pd
import joblib

def predict_all_and_single(X_test, model_file='knn_model.pkl'):
    model = joblib.load(model_file)
    y_pred = model.predict(X_test)
    single_pred = model.predict([X_test.iloc[0].values])[0]
    return y_pred, single_pred

# Main
test_df = pd.read_csv('groundwater_test.csv')
X_test = test_df.drop('Class', axis=1)
y_test = test_df['Class']
y_pred_all, single_pred = predict_all_and_single(X_test)

print("First 10 predictions vs actual:")
for i in range(10):
    print(f"Predicted: {y_pred_all[i]} | Actual: {y_test.iloc[i]}")
print("\nPrediction for first test vector only:", single_pred)
