import pandas as pd

def load_data(filepath, sheet_name="Purchase data"):
    # Read Excel
    df = pd.read_excel(filepath, sheet_name=sheet_name)

    # Drop first col (Customer IDs)
    df = df.drop(df.columns[0], axis=1)

    return df

def classify_customers(df):
    # Mark customers as RICH if Payment > 200 else POOR
    df["Class"] = df.iloc[:, -1].apply(lambda x: "RICH" if x > 200 else "POOR")
    return df

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    df = load_data(r"C:\Users\keert\OneDrive - Amrita vishwa vidyapeetham\Amrita\Sem5\ML\lab2\Lab Session Data.xlsx",
                   sheet_name="Purchase data")

    classified_df = classify_customers(df)
    print(classified_df)
