import pandas as pd
import statistics
import matplotlib.pyplot as plt

def load_stock_data(filepath, sheet_name="IRCTC Stock Price"):
    return pd.read_excel(filepath, sheet_name=sheet_name)

def mean_variance_price(df):
    prices = df.iloc[:, 3].dropna().astype(float)  # Column D = 4th column
    return statistics.mean(prices), statistics.variance(prices)

def wednesday_mean(df):
    df["Date"] = pd.to_datetime(df.iloc[:, 0])  # assuming 1st col is Date
    wed_prices = df[df["Date"].dt.day_name() == "Wednesday"].iloc[:, 3].astype(float)
    return statistics.mean(wed_prices)

def april_mean(df):
    df["Date"] = pd.to_datetime(df.iloc[:, 0])
    april_prices = df[df["Date"].dt.month == 4].iloc[:, 3].astype(float)
    return statistics.mean(april_prices)

def loss_probability(df):
    chg = df.iloc[:, 8].astype(float)  # Chg% in column I (9th col)
    return (chg < 0).mean()

def profit_wednesday_probability(df):
    df["Date"] = pd.to_datetime(df.iloc[:, 0])
    chg = df.iloc[:, 8].astype(float)
    wed_chg = chg[df["Date"].dt.day_name() == "Wednesday"]
    return (wed_chg > 0).mean()

def conditional_profit_probability(df):
    # P(Profit | Wednesday) = (Profit ∩ Wednesday) / P(Wednesday)
    df["Date"] = pd.to_datetime(df.iloc[:, 0])
    chg = df.iloc[:, 8].astype(float)

    wed_mask = df["Date"].dt.day_name() == "Wednesday"
    profit_mask = chg > 0

    joint = (wed_mask & profit_mask).sum()
    wed_total = wed_mask.sum()

    return joint / wed_total if wed_total > 0 else 0

def scatter_chg_weekday(df):
    df["Date"] = pd.to_datetime(df.iloc[:, 0])
    df["Weekday"] = df["Date"].dt.day_name()
    chg = df.iloc[:, 8].astype(float)

    plt.figure(figsize=(8,5))
    plt.scatter(df["Weekday"], chg)
    plt.xlabel("Day of Week")
    plt.ylabel("Chg%")
    plt.title("Chg% vs Day of Week")
    plt.show()

# ------------------ MAIN ------------------
if __name__ == "__main__":
    df = load_stock_data(r"C:\Users\keert\OneDrive - Amrita vishwa vidyapeetham\Amrita\Sem5\ML\lab2\Lab Session Data.xlsx")

    mean_val, var_val = mean_variance_price(df)
    print("Mean Price:", mean_val, "Variance:", var_val)

    print("Wednesday Mean:", wednesday_mean(df))
    print("April Mean:", april_mean(df))
    print("Loss Probability:", loss_probability(df))
    print("Profit on Wednesday Probability:", profit_wednesday_probability(df))
    print("Conditional Profit | Wednesday:", conditional_profit_probability(df))

    scatter_chg_weekday(df)
