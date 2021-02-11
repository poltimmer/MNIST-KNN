# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
from KNN import KNN
from matplotlib import pyplot as plt

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
# Upper limit for range iterator k. Set to maximum k-1. (21 gives max k = 20)
k_upper = 21
# %%
train_preds = []
test_preds = []

for k in tqdm(range(1, k_upper)):
    model = KNN(x_train, y_train, k)
    train_preds.append(model.predict(x_train))
    test_preds.append(model.predict(x_test))

# %%
# train_risk = [loss/len(y_train) for loss in train_loss]
# test_risk = [loss/len(y_test) for loss in test_loss]

train_risk = [(y_train != pred).mean() for pred in train_preds]
test_risk = [(y_test != pred).mean() for pred in test_preds]

train_loss = [sum((y_train != pred)) for pred in train_preds]
test_loss= [sum((y_test != pred)) for pred in test_preds]

fig = plt.figure(figsize=(10,10))
ax_risk = fig.add_subplot(221)
ax_loss = fig.add_subplot(222)

ax_risk.plot(range(1, k_upper), train_risk, label="train", marker=".")
ax_risk.plot(range(1, k_upper), test_risk, label="test", marker=".")
ax_risk.legend()
ax_risk.set_xticks(range(1, k_upper))
ax_risk.set_ylabel("empirical risk")
ax_risk.set_xlabel("k")
ax_risk.grid()

ax_loss.plot(range(1, k_upper), train_loss, label="train", marker=".")
ax_loss.plot(range(1, k_upper), test_loss, label="test", marker=".")
ax_loss.legend()
ax_loss.set_xticks(range(1, k_upper))
ax_loss.set_ylabel("empirical loss")
ax_loss.set_xlabel("k")
ax_loss.grid()


plt.show()
# %%
