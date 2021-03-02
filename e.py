
# %%
import pandas as pd
import numpy as np
from KNN import KNN
from matplotlib import pyplot as plt
from crossval import acc_score, leave_one_out
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle

p = 5 #TODO DEZE AANPASSEN
metric = 'minkowski'

# %%
def main():


    train = pd.read_csv("input/MNIST_train.csv", header=None)
    y_train = train[0]
    x_train = train.drop(columns=0)
    print('Done loading')

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
    preds = []
    for k in range(1, k_upper):
        print('scoring for k =' + str(k))
        train_pred = leave_one_out(x_train, y_train, metric=metric, k=k, p=p)
        preds.append({"train": train_pred})#, "test": test_pred})



    pickle.dump(preds, open("preds_e.pickle", "wb"))


    train_risk = [1-p.get('train') for p in preds]


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
# %%
