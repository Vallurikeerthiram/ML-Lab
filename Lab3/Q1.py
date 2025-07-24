import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

def load_and_filter_data(file_path, class_column, feature_column, class1, class2):
    df = pd.read_csv(file_path)
    df = df[[class_column, feature_column]].dropna()
    data1 = df[df[class_column] == class1][feature_column].values
    data2 = df[df[class_column] == class2][feature_column].values
    return data1, data2

def calculate_centroid_and_spread(data):
    return np.mean(data), np.std(data)

def calculate_interclass_distance(c1, c2):
    return norm(c1 - c2)

def plot_class_distributions(data1, data2, class1, class2, feature_column):
    plt.hist(data1, alpha=0.5, label=class1)
    plt.hist(data2, alpha=0.5, label=class2)
    plt.xlabel(feature_column)
    plt.ylabel("Frequency")
    plt.title("Class-wise Feature Distribution")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
file_path = "Atal_Jal_Disclosed_Ground_Water_Level-2015-2022.csv"
class_column = 'State Name'
feature_column = 'GWL (in Mtr)'
class1 = 'Maharashtra'
class2 = 'Karnataka'

data1, data2 = load_and_filter_data(file_path, class_column, feature_column, class1, class2)
centroid1, spread1 = calculate_centroid_and_spread(data1)
centroid2, spread2 = calculate_centroid_and_spread(data2)
interclass_dist = calculate_interclass_distance(centroid1, centroid2)

print(f"Centroid of {class1}: {centroid1:.3f}")
print(f"Spread of {class1}: {spread1:.3f}")
print(f"Centroid of {class2}: {centroid2:.3f}")
print(f"Spread of {class2}: {spread2:.3f}")
print(f"Interclass Distance: {interclass_dist:.3f}")

plot_class_distributions(data1, data2, class1, class2, feature_column)
