# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from func import leave_one_out
# %%
# load training data
train = pd.read_csv("input/MNIST_train_small.csv", header=None)
y_train = train[0]
x_train = train.drop(columns=0)

# load test data
test = pd.read_csv("input/MNIST_test_small.csv", header=None)
y_test = test[0]
x_test = test.drop(columns=0)
# %%
Xs, ys = x_train[:500], y_train[:500] #smaller sets to test soluton
k_range = np.linspace(1,20,20, dtype=int)
result = [{'risk':1-leave_one_out(Xs, ys, k), 'k':k} for k in tqdm(k_range)]

# %% 
line = [r.get('risk') for r in result]
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot()
ax.plot(range(1, 21), line, label="leave-one-out", marker=".")
ax.legend()
ax.set_xticks(range(1,21))
ax.set_ylabel("empirical risk")
ax.set_xlabel("k")
ax.set_ylim(bottom=0, top=0.35)
ax.grid()
plt.show()
# %%
