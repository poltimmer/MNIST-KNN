import pandas as pd
import numpy as np
from KNN import KNN
from matplotlib import pyplot as plt
from functools import partial
from crossval import acc_score, leave_one_out, leave_one_out_legacy
from tqdm.contrib.concurrent import process_map
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle



def loo_rewrite(k, x_train, y_train, metric='euclidean', p=None):
    return leave_one_out_legacy(x_train, y_train, k, metric=metric, p=p)


def main():
    p = 7
    metric = 'minkowski'

    train = pd.read_csv("input/MNIST_train.csv", header=None)
    y_train = train[0]
    x_train = train.drop(columns=0)
    print('Done loading')

    # x_train = x_train[:500]
    # y_train = y_train[:500]

    k_upper = 21

    #Preprocessing/optimization
    #SCALING
    scaler = MinMaxScaler().fit(x_train)
    # x_test = scaler.transform(x_test)
    x_train = scaler.transform(x_train)

    #PCA
    pca = PCA(n_components=25).fit(x_train) #safe side of accurraccy
    x_train = pd.DataFrame(pca.transform(x_train))
    print('Done preprocessing, starting estimations')

    #PREDICTING
    # preds = []
    # for k in range(1, k_upper):
    #     print('scoring for k =' + str(k))
    #     train_pred = leave_one_out(x_train, y_train, metric=metric, k=k, p=p)
    #     preds.append({"train": train_pred})#, "test": test_pred})

    scores = process_map(partial(loo_rewrite, x_train=x_train, y_train=y_train, metric=metric, p=p), range(1, k_upper), max_workers=10)
    print(scores)

    pickle.dump(scores, open("scores_e.pickle", "wb"))
    pd.DataFrame(scores).to_csv('scores_e.csv')


    train_risk = [1-score for score in scores]


    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot()
    ax.plot(range(1, k_upper), train_risk, label="train_risk", marker=".")
    ax.legend()
    ax.set_xticks(range(1, k_upper))
    ax.set_ylabel("empirical risk using LOOCV")
    ax.set_xlabel("k")
    ax.grid()

    plt.show()

if __name__ == "__main__":
    main()