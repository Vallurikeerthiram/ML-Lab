import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def evaluate_model(train_df, test_df):
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    print("Training Accuracy:", round(accuracy_score(y_train, y_train_pred) * 100, 2), "%")
    print("Testing Accuracy:", round(accuracy_score(y_test, y_test_pred) * 100, 2), "%")

    print("\nConfusion Matrix (Train):")
    print(confusion_matrix(y_train, y_train_pred))

    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, y_test_pred))

    print("\nClassification Report (Train):")
    print(classification_report(y_train, y_train_pred))

    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred))

# Main
train_df = pd.read_csv('groundwater_train.csv')
test_df = pd.read_csv('groundwater_test.csv')
evaluate_model(train_df, test_df)
