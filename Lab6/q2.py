# q2.py - Gini Calculator (A2)
"""
A2. Gini Index Calculation Module
This module provides functions to calculate Gini impurity for decision tree construction.
"""

import numpy as np
from collections import Counter
import pandas as pd


def calculate_gini_index(labels):
    """
    calculate gini index for a list of labels gini=1- sigma(pi^2)
    """
    if len(labels) == 0:
        return 0.0
    
    # Count occurrences of each label
    label_counts = Counter(labels)
    total_count = len(labels)
    
    # Calculate Gini index
    gini = 1.0
    for count in label_counts.values():
        probability = count / total_count
        gini -= probability ** 2
    
    return gini

def equal_width_binning (data, bins=4):
    """perform equal with bining on continous data
    parameters are data: array like continous numerical data and bins: int number of bins
    returns list binned labels as categorical values"""
    data=pd.Series(data).dropna()
    bin_edges=np.linspace(data.min(),data.max(),bins+1)
    binned=np.digitize(data,bin_edges,right=False)-1
    return [f"Bin_{i}" for i in binned]

if __name__ =="__main__":
    file=r"C:\Users\keert\OneDrive - Amrita vishwa vidyapeetham\Amrita\Sem5\ML\Lab6\rajasthan.xlsx"
    df = pd.read_excel(file)

    # Select rainfall columns
    rainfall_cols = [col for col in df.columns if "R/F" in col]

    # Loop over each month column
    for col in rainfall_cols:
        values = df[col].dropna().values
        binned_labels = equal_width_binning(values, bins=4)
        gini = calculate_gini_index(binned_labels)

        print(f"Gini Index for {col}: {gini:.4f}")
