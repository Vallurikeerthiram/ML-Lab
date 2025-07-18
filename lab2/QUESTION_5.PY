# Full dataset (excluding Customer IDs)
data = [
    [20, 6, 2, 386],
    [16, 3, 6, 289],
    [27, 6, 2, 393],
    [19, 1, 2, 110],
    [24, 4, 2, 280],
    [22, 1, 5, 167],
    [15, 4, 2, 271],
    [18, 4, 2, 274],
    [21, 1, 4, 148],
    [16, 2, 4, 198]
]

# Get medians for each attribute
def get_median(values):
    values = sorted(values)
    n = len(values)
    return values[n//2] if n % 2 == 1 else (values[n//2 - 1] + values[n//2]) / 2

medians = []
for i in range(4):
    column = [row[i] for row in data]
    medians.append(get_median(column))

# Use median thresholds to binarize C1 and C2
C1 = data[0]
C2 = data[1]
binary_C1 = [1 if C1[i] > medians[i] else 0 for i in range(4)]
binary_C2 = [1 if C2[i] > medians[i] else 0 for i in range(4)]

# Compute f-values
f11 = f10 = f01 = f00 = 0
for i in range(4):
    if binary_C1[i] == 1 and binary_C2[i] == 1:
        f11 += 1
    elif binary_C1[i] == 1 and binary_C2[i] == 0:
        f10 += 1
    elif binary_C1[i] == 0 and binary_C2[i] == 1:
        f01 += 1
    else:
        f00 += 1

jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) else 0
smc = (f11 + f00) / (f11 + f10 + f01 + f00) if (f11 + f10 + f01 + f00) else 0

print("Binary C1:", binary_C1)
print("Binary C2:", binary_C2)
print("JC:", round(jc, 4), "SMC:", round(smc, 4))
