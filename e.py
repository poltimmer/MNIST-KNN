
# %%
import pandas as pd
import numpy as np
from KNN import KNN
from matplotlib import pyplot as plt
from func import acc_score, leave_one_out
from sklearn.decomposition import PCA


# %%
train = pd.read_csv("input/MNIST_train.csv", header=None)
y_train = train[0]
x_train = train.drop(columns=0)


# %%
### Preprocessing: PCA to increase speed
pca = PCA(n_components=25) #safe side of accurraccy
x_train = pd.DataFrame(pca.fit(x_train).transform(x_train))


# %%
k = 5

score = leave_one_out(x_train, y_train, k)





# %%
