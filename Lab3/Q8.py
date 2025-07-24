import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def plot_accuracy_vs_k(X_train, y_train, X_test, y_test, k_range=range(1, 12)):
    accuracies = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test))
        accuracies.append(acc)
        print(f"k = {k}, Accuracy: {acc * 100:.2f}%")
    plt.plot(k_range, [a * 100 for a in accuracies], marker='o')
    plt.xlabel("k")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs k")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("knn_accuracy_plot.png")
    plt.show()

# Main
train_df = pd.read_csv('groundwater_train.csv')
test_df = pd.read_csv('groundwater_test.csv')
X_train = train_df.drop('Class', axis=1)
y_train = train_df['Class']
X_test = test_df.drop('Class', axis=1)
y_test = test_df['Class']
plot_accuracy_vs_k(X_train, y_train, X_test, y_test)
