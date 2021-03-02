
#%%
from sklearn.decomposition import PCA
import pandas as pd
from crossval import acc_score
from KNN import KNN
import time
import numpy as np
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
pca = PCA(n_components=25)
x_train_tr = pd.DataFrame(pca.fit(x_train).transform(x_train))
x_test_tr = pd.DataFrame(pca.transform(x_test))

# %%
model_1 = KNN(x_train, y_train, 5)
model_2 = KNN(x_train_tr, y_train, 5)
scores = []
for i in range(5):
    # Time benchmark
    now = time.time()
    score = acc_score(model_2.predict(x_test_tr), y_test)
    scores.append({'time PCA' : time.time() - now, 'score PCA': score})
    now = time.time()
    score = acc_score(model_1.predict(x_test), y_test)
    scores.append({'time non-PCA' : time.time() - now, 'score non-PCA': score})

# %%
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot()
ax.plot(range(5), [s.get('time PCA') for s in scores if s.get('time PCA')] , label="PCA prediction time", marker=".")
ax.plot(range(5), [s.get('time non-PCA') for s in scores if s.get('time non-PCA')], label="train", marker=".")
#ax.plot(range(1, k_upper), test_risk, label="test", marker=".")
ax.legend()
ax.set_xticks(range(5))
ax.set_ylabel("prediction time on small training set (seconds)")
ax.set_xlabel("iteration")


## testing N components
# %%
def get_score(components):
    pca = PCA(n_components=components)
    x_train_tr = pd.DataFrame(pca.fit(x_train).transform(x_train))
    x_test_tr = pd.DataFrame(pca.transform(x_test))
    score = acc_score(KNN(x_train_tr, y_train, 5).predict(x_test_tr), y_test)
    return score

comp_scores = [{comp: get_score(comp)} for comp in component_range]


# %%
