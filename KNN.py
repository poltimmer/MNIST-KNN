# %%
import math
import pandas as pd


train = pd.read_csv("input/MNIST_train_small.csv", header=None)
y_train = train[0]
x_train = train.drop(columns=0)

test = pd.read_csv("input/MNIST_test_small.csv", header=None)
y_test = test[0]
x_test = test.drop(columns=0)

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
