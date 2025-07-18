# Use only first 3 attributes: Candies, Mangoes, Milk Packets
C1 = [20, 6, 2]
C2 = [16, 3, 6]

# Compute dot product
dot = 0
for i in range(len(C1)):
    dot += C1[i] * C2[i]

# Compute magnitudes
mag1 = sum([x*x for x in C1]) ** 0.5
mag2 = sum([x*x for x in C2]) ** 0.5

# Compute cosine similarity
cos_sim = dot / (mag1 * mag2) if mag1 != 0 and mag2 != 0 else 0

print("Version 3: Ignoring Payment (scaled attribute)")
print("Cosine Similarity:", round(cos_sim, 4))
