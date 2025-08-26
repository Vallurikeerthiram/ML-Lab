import pandas as pd
import numpy as np

def load_data(filepath, sheet_name="Purchase data"):
    # Read Excel
    df = pd.read_excel(filepath, sheet_name=sheet_name)

    # Drop first column (Customer IDs)
    df = df.drop(df.columns[0], axis=1)

    # Convert everything to numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Separate into A (all but last column) and C (last column)
    A = df.iloc[:, :-1].values.astype(float)
    C = df.iloc[:, -1].values.astype(float)

    return A, C

def get_dimensionality(A):
    return A.shape[1]   # number of columns = 3 (Candies, Mangoes, Milk)

def get_num_vectors(A):
    return A.shape[0]   # number of rows = 10 (customers)

def get_rank(A):
    return np.linalg.matrix_rank(A)

def get_product_cost(A, C):
    return np.linalg.pinv(A).dot(C)   # pseudo-inverse solution

# ------------------ Example Usage ------------------
if __name__ == "__main__":
    A, C = load_data(r"C:\Users\keert\OneDrive - Amrita vishwa vidyapeetham\Amrita\Sem5\ML\lab2\Lab Session Data.xlsx", 
                     sheet_name="Purchase data")

    print("Dimensionality:", get_dimensionality(A))
    print("Number of vectors:", get_num_vectors(A))
    print("Rank of A:", get_rank(A))
    print("Cost per product (Candy, Mango, Milk):", get_product_cost(A, C))
