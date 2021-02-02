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



def distance(a, b):
    # TODO implement eucledian distance
    pass

