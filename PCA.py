
#%%
from sklearn.decomposition import PCA
import pandas as pd
from func import acc_score
from KNN import KNN
import time
import numpy as np

# load training data
train = pd.read_csv("input/MNIST_train_small.csv", header=None)
y_train = train[0]
x_train = train.drop(columns=0)

# load test data
test = pd.read_csv("input/MNIST_test_small.csv", header=None)
y_test = test[0]
x_test = test.drop(columns=0)

# %%
pca = PCA(n_components=100)
x_train_tr = pd.DataFrame(pca.fit(x_train).transform(x_train))
x_test_tr = pd.DataFrame(pca.transform(x_test))

# %%
model_1 = KNN(x_train, y_train, 5)
model_2 = KNN(x_train_tr, y_train, 5)


# Time benchmark
# %%
now = time.time()
score = acc_score(model_2.predict(x_test_tr), y_test)
print('time:',time.time() - now, score)
now = time.time()
score = acc_score(model_1.predict(x_test), y_test)
print('time:',time.time() - now, score)


## testing N components
# %%
def get_score(components):
    pca = PCA(n_components=components)
    x_train_tr = pd.DataFrame(pca.fit(x_train).transform(x_train))
    x_test_tr = pd.DataFrame(pca.transform(x_test))
    score = acc_score(KNN(x_train_tr, y_train, 5).predict(x_test_tr), y_test)
    return score

scores = [{comp: get_score(comp)} for comp in np.linspace(25,250,10, dtype=int)]
scores2 = [{comp: get_score(comp)} for comp in np.linspace(1, 25, 25, dtype=int)]


# %%
