#%%
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from KNN import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import time


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




# %%
#######
#### PCA
#######

# Find significance in speedup:
pca = PCA(n_components=25)
x_train_tr = pd.DataFrame(pca.fit(x_train).transform(x_train))
x_test_tr = pd.DataFrame(pca.transform(x_test))
model_1 = KNN(x_train, y_train, 5)
model_2 = KNN(x_train_tr, y_train, 5)

#Speed comparison:
now = time.time()
score = acc_score(model_2.predict(x_test_tr), y_test)
print('time PCA set, score:',time.time() - now, score)
now = time.time()
score = acc_score(model_1.predict(x_test), y_test)
print('time full set, score:',time.time() - now, score)


# Find best number of components:

# %%
def get_score(components):
    pca = PCA(n_components=components)
    x_train_tr = pd.DataFrame(pca.fit(x_train).transform(x_train))
    x_test_tr = pd.DataFrame(pca.transform(x_test))
    score = acc_score(KNN(x_train_tr, y_train, 5).predict(x_test_tr), y_test)
    return score

component_range = np.append(np.linspace(5, 25, 20, dtype=int),np.linspace(50,100,3, dtype=int))
comp_scores = {comp: get_score(comp) for comp in component_range}

# %%
plt.plot(comp_scores.keys(), comp_scores.values(), linewidth=1, marker='.')
plt.grid()
plt.text(25, 0.91, 'Score at n=25: ' + str(comp_scores.get(25)))
plt.ylabel("Accuracy")
plt.xlabel("N components in PCA")
plt.title('Comparing accuraccy scores of KNN using PCA with N components')
# %%
