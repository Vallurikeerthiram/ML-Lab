import numpy as np

data_matrix = np.array([
    [20, 6, 2],
    [16, 3, 6],
    [27, 6, 2],
    [19, 1, 2],
    [24, 4, 2],
    [22, 1, 5],
    [15, 4, 2],
    [18, 4, 2],
    [21, 1, 4],
    [16, 2, 4]
])

total_paid = np.array([
    [386],
    [289],
    [393],
    [110],
    [280],
    [167],
    [271],
    [274],
    [148],
    [198]
])

features = data_matrix.shape[1]
customers = data_matrix.shape[0]

U1, S1, V1 = np.linalg.svd(data_matrix)
true_rank = sum(val > 1e-10 for val in S1)

pinv_data = np.linalg.pinv(data_matrix)
item_prices = pinv_data @ total_paid

print("---- RESULTS ----")
print(f"Dimensionality of vector space : {features}")
print(f"Number of vectors              : {customers}")
print(f"Rank of Matrix A               : {true_rank}")
print("\nEstimated cost of each product:")
print(f"Cost of 1 Candy       : Rs {item_prices[0][0]:.2f}")
print(f"Cost of 1 Kg Mango    : Rs {item_prices[1][0]:.2f}")
print(f"Cost of 1 Milk Packet : Rs {item_prices[2][0]:.2f}")
