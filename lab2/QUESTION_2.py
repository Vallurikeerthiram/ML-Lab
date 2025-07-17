import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Data setup
customer_data = {
    'candy': [20, 16, 27, 19, 24, 22, 15, 18, 21, 16],
    'mango': [6, 3, 6, 1, 4, 1, 4, 4, 1, 2],
    'milk': [2, 6, 2, 2, 2, 5, 2, 2, 4, 4],
    'payment': [386, 289, 393, 110, 280, 167, 271, 274, 148, 198]
}

df3 = pd.DataFrame(customer_data)

# Function to tag customer as rich/poor
def tag_income(pay):
    return 'RICH' if pay > 200 else 'POOR'

df3['income_group'] = df3['payment'].apply(tag_income)

X3 = df3[['candy', 'mango', 'milk']].to_numpy()
y3 = np.where(df3['income_group'] == 'RICH', 1, 0)

log_reg = LogisticRegression()
log_reg.fit(X3, y3)
y_pred = log_reg.predict(X3)

print("---- Classification Report ----")
print(classification_report(y3, y_pred, target_names=['POOR', 'RICH']))
print(f"Accuracy: {accuracy_score(y3, y_pred):.2f}")

df3['predicted_income'] = ['RICH' if val == 1 else 'POOR' for val in y_pred]
print("\n---- Customer Classification ----")
print(df3[['candy', 'mango', 'milk', 'payment', 'income_group', 'predicted_income']])
