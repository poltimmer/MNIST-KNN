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



def distance(a, b):
    # TODO implement eucledian distance
    pass

