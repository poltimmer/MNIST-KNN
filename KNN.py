# %%
import math


# %% 
def knn_iter(x, k):
    neighbours = {}
    for i in range(k):
        closest = None
        neighbours[closest] = neighbours.get(closest, 0) + 1 #TODO this should be class of neighbour, not neighbour itself. 

    return max(neighbours, key=neighbours.get)



def distance(a, b):
    # TODO implement eucledian distance
    pass

