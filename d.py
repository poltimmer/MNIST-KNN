#%%
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from KNN import *
from sklearn.preprocessing import MinMaxScaler

# load training data
train = pd.read_csv("input/MNIST_train_small.csv", header=None)
y_train = train[0]
x_train = train.drop(columns=0)

# load test data
test = pd.read_csv("input/MNIST_test_small.csv", header=None)
y_test = test[0]
x_test = test.drop(columns=0)

#%%
Xs_tr, ys_tr = x_train[:500], y_train[:500] #smaller train sets to test soluton
Xs_te, ys_te = x_test[:500], y_test[:500] #smaller test sets to test soluton

distance_results_small = []

def acc_score(y_test, y_pred):
        return (y_test == y_pred).mean()

# minowski, weighted minowski and mahalanobis left out
for metric in tqdm(['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'matching', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']):
    clf = KNN(Xs_tr, ys_tr, 5)
    y_pred = clf.predict(Xs_te, metric)
    accuracy = acc_score(ys_te, y_pred)
    distance_results_small.append([metric, accuracy])

#%%
#Normalize data
Xs_tr_norm = pd.DataFrame(MinMaxScaler().fit_transform(Xs_tr)) #immediately convert back to df
Xs_te_norm = pd.DataFrame(MinMaxScaler().fit_transform(Xs_te))

#%%
distance_results_small_scaled = []

for metric in tqdm(['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'matching', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']):
    clf = KNN(Xs_tr_norm, ys_tr, 5)
    y_pred = clf.predict(Xs_te_norm, metric)
    accuracy = acc_score(ys_te, y_pred)
    distance_results_small_scaled.append([metric, accuracy])

#%%
# plot distance metric vs loss
plt.bar(*zip(*distance_results_small), label='Unprocessed')
plt.bar(*zip(*distance_results_small_scaled), label='Normalized')
plt.xlabel("Metric")
plt.xticks(rotation=90)
plt.ylabel("Accuracy")
plt.legend(loc='best')
plt.show()
# %%
