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
########################### DISTANCE ON FULL DATASET ###############################
accuracy_barchart = []
time_barchart = []

def acc_score(y_test, y_pred):
        return (y_test == y_pred).mean()

# minowski, weighted minowski and mahalanobis left out
labels = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'matching', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
labels2 = ['correlation', 'cosine', 'dice', 'euclidean', 'jensenshannon', 'rogerstanimoto', 'sokalmichener', 'sokalsneath']
for metric in tqdm(labels):
    clf = KNN(x_train, y_train, 5)
    now = time.time()
    y_pred = clf.predict(x_test, metric)
    time_barchart.append(time.time() - now)
    accuracy = acc_score(y_test, y_pred)
    accuracy_barchart.append(accuracy)

#%%
# Retreive 3 best scoring metrics based on accuracy
indices = np.argpartition(accuracy_barchart, -3)[-3:]
best_metrices = []
for i in indices:
    #best_metrices.append([labels[i], accuracy_barchart[i], time_barchart[i]])
    print("Metric: {m}. Accuracy: {a}. Time: {t}".format(m=labels[i], a=accuracy_barchart[i], t=time_barchart[i]))

# Retreive values for euclidean
i = labels.index('euclidean')
print("For euclidean distance: Accuracy: {a}. Time: {t}".format(m=labels[i], a=accuracy_barchart[i], t=time_barchart[i]))

#%% 
# Normalize accuracy scores and training times for better comparison
time_barchart_scaled = pd.DataFrame(MinMaxScaler().fit_transform(pd.DataFrame(time_barchart)))
accuracy_barchart_scaled = pd.DataFrame(MinMaxScaler().fit_transform(pd.DataFrame(accuracy_barchart)))

# Compute some kind of weighted score for accuracy and training time combined
combined_score = []
w = 2/3
for i in range(len(labels)):
    combined_score.append(w*accuracy_barchart_scaled[0][i] + (1-w)*(1-time_barchart_scaled[0][i]))

# Retreive 3 best scoring metrics on the new score
indices = np.argpartition(combined_score, -3)[-3:]
best_metrices = []
for i in indices:
    #best_metrices.append([labels[i], accuracy_barchart[i], time_barchart[i]])
    print("Metric: {m}. Combined score: {c}. Accuracy: {a}. Time: {t}".format(m=labels[i], a=accuracy_barchart[i], c=combined_score[i], t=time_barchart[i]))

# Retreive values for euclidean
i = labels.index('euclidean')
print("For euclidean distance: Combined score: {c}. Accuracy: {a}. Time: {t}".format(m=labels[i], a=accuracy_barchart[i], c=combined_score[i], t=time_barchart[i]))


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

def acc_score(y_test, y_pred):
        return (y_test == y_pred).mean()

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
#%%
########################### DISTANCES USING PCA ###############################
accuracy_barchart = []
time_barchart = []

def acc_score(y_test, y_pred):
        return (y_test == y_pred).mean()

# minowski, weighted minowski and mahalanobis left out
labels = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'matching', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
labels2 = ['euclidean', 'sqeuclidean', 'dice', 'matching']
for metric in tqdm(labels):
    pca = PCA(n_components=25)
    x_train_tr = pd.DataFrame(pca.fit(x_train).transform(x_train))
    x_test_tr = pd.DataFrame(pca.transform(x_test))
    now = time.time()
    accuracy = acc_score(KNN(x_train_tr, y_train, 5).predict(x_test_tr), y_test)
    time_barchart.append(time.time() - now)
    accuracy_barchart.append(accuracy)

#%%
# Retreive 3 best scoring metrics based on accuracy
indices = np.argpartition(accuracy_barchart, -3)[-3:]
best_metrices = []
for i in indices:
    #best_metrices.append([labels[i], accuracy_barchart[i], time_barchart[i]])
    print("Metric: {m}. Accuracy: {a}. Time: {t}".format(m=labels[i], a=accuracy_barchart[i], t=time_barchart[i]))

# Retreive values for euclidean
i = labels.index('euclidean')
print("For euclidean distance: Accuracy: {a}. Time: {t}".format(m=labels[i], a=accuracy_barchart[i], t=time_barchart[i]))
#%% 
# Normalize accuracy scores and training times for better comparison
time_barchart_scaled = pd.DataFrame(MinMaxScaler().fit_transform(pd.DataFrame(time_barchart)))
accuracy_barchart_scaled = pd.DataFrame(MinMaxScaler().fit_transform(pd.DataFrame(accuracy_barchart)))

#%%
# Compute some kind of weighted score for accuracy and training time combined
combined_score = []
w = 2/3
for i in range(len(labels)):
    combined_score.append(w*accuracy_barchart_scaled[0][i] + (1-w)*(1-time_barchart_scaled[0][i]))

# Retreive 3 best scoring metrics on the new score
indices = np.argpartition(combined_score, -3)[-3:]
best_metrices = []
for i in indices:
    #best_metrices.append([labels[i], accuracy_barchart[i], time_barchart[i]])
    print("Metric: {m}. Combined score: {c}. Accuracy: {a}. Time: {t}".format(m=labels[i], c=combined_score[i], a=accuracy_barchart[i], t=time_barchart[i]))

# Retreive values for euclidean
i = labels.index('euclidean')
print("For euclidean distance: Metric: {m}. Combined score: {c}. Accuracy: {a}. Time: {t}".format(m=labels[i], c=combined_score[i], a=accuracy_barchart[i], t=time_barchart[i]))
# %%
clf = KNN(x_train, y_train, 5)
now = time.time()
y_pred = clf.predict(x_test, 'minkowski', )
time_barchart.append(time.time() - now)
accuracy = acc_score(y_test, y_pred)
accuracy_barchart.append(accuracy)