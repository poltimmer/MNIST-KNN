# %%
import math


#TODO define datapoints as tuples with (point in space, class)

# %% 
def knn_iter(x, k):
    neighbours = {}
    for i in range(k):
        ith_closest = None
        neighbours[ith_closest[1]] = neighbours.get(ith_closest[1], 0) + 1 

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
