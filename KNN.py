# %%
import math


# %% 
def knn(x, k):
    neighbours = {}
    for i in range(k):
        # closests = #min of distance
        closest = None
        neighbours[closest] = neighbours.get(closest, 0) + 1

    return max(neighbours, key=neighbours.get)


# %%
def distance(a, b):
    distance_sq = 0
    i = 0
    while i < len(a):
        distance_sq += (a[i] - b[i])**2
        i += 1
    return math.sqrt(distance_sq)
# %%
