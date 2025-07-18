import statistics as stats

records = [
    {"dt": "Jul 14, 2020", "mo": "Jul", "dy": "Tue", "p": 1362.15, "c": -2.52},
    {"dt": "Jul 13, 2020", "mo": "Jul", "dy": "Mon", "p": 1397.35, "c": -0.26},
    {"dt": "Jul 10, 2020", "mo": "Jul", "dy": "Fri", "p": 1400.95, "c": 1.15},
    {"dt": "Jul 09, 2020", "mo": "Jul", "dy": "Thu", "p": 1385.05, "c": -0.36},
    {"dt": "Jul 08, 2020", "mo": "Jul", "dy": "Wed", "p": 1390.10, "c": -0.52},
    {"dt": "Jul 07, 2020", "mo": "Jul", "dy": "Tue", "p": 1397.40, "c": -0.24},
    {"dt": "Jul 06, 2020", "mo": "Jul", "dy": "Mon", "p": 1400.75, "c": -0.31},
    {"dt": "Jul 03, 2020", "mo": "Jul", "dy": "Fri", "p": 1405.10, "c": -0.51},
    {"dt": "Jul 02, 2020", "mo": "Jul", "dy": "Thu", "p": 1412.35, "c": 3.62},
    {"dt": "Jul 01, 2020", "mo": "Jul", "dy": "Wed", "p": 1363.05, "c": 0.32},
]

vals = [r["p"] for r in records]
diffs = [r["c"] for r in records]

mean_val = stats.mean(vals)
var_val = stats.variance(vals)
print(f"Mean: {mean_val:.2f}, Variance: {var_val:.2f}")

wed_vals = [r["p"] for r in records if r["dy"] == "Wed"]
mean_wed = stats.mean(wed_vals)
print(f"\nWed Mean: {mean_wed:.2f} | Diff: {mean_wed - mean_val:.2f}")

apr_vals = [r["p"] for r in records if r["mo"] == "Apr"]
if apr_vals:
    mean_apr = stats.mean(apr_vals)
    print(f"\nApril Mean: {mean_apr:.2f} | Diff: {mean_apr - mean_val:.2f}")
else:
    print("\nNo April records.")

loss = len([c for c in diffs if c < 0]) / len(records)
print(f"\nLoss Chance: {loss:.2%}")

wed_changes = [r["c"] for r in records if r["dy"] == "Wed"]
wed_profit = len([c for c in wed_changes if c > 0]) / len(wed_changes)
print(f"\nProfit on Wed: {wed_profit:.2%}")
print(f"Conditional Profit|Wed: {wed_profit:.2%}")

print("\nDay : Change%")
for r in records:
    print(f"{r['dy']:>3} : {r['c']:>6.2f}%")
