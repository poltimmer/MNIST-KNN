
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


def main():

    train = pd.read_csv("input/MNIST_train.csv", header=None)
    y_train = train[0]
    x_train = train.drop(columns=0)

    test = pd.read_csv("input/MNIST_test.csv", header=None)
    y_test = test[0]
    x_test = test.drop(columns=0)
    print('Done loading')

    k_upper = 10

    #Preprocessing/optimization
    #SCALING
    scaler = MinMaxScaler().fit(x_train)
    x_test = scaler.transform(x_test)
    x_train = scaler.transform(x_train)

    #PCA
    pca = PCA(n_components=25).fit(x_train) #safe side of accurraccy
    x_train = pd.DataFrame(pca.transform(x_train))
    x_test = pd.DataFrame(pca.transform(x_test))
    print('Done preprocessing, starting estimations')



    preds = []
    for k in range(1, k_upper):
        print('scoring for k =' + str(k))
        model = KNN(x_train, y_train, k)
        test_pred = model.predict_concurrent(x_test, metric=metric, p=p)
        test_risk = (y_test != test_pred).mean()
        test_loss = sum((y_test != test_pred))
        preds.append({"k": k, 'test_risk': test_risk, 'test_loss': test_loss})

    pickle.dump(preds, open("preds_f.pickle", "wb"))


    fig = plt.figure(figsize=(10,6))
    ax_risk = fig.add_subplot(122)
    ax_loss = fig.add_subplot(121)

    ax_risk.plot(range(1, k_upper), [p.get('test_risk') for p in preds], label="risk", marker=".")
    ax_risk.legend()
    ax_risk.set_xticks(range(1, k_upper))
    ax_risk.set_ylabel("empirical risk")
    ax_risk.set_xlabel("k")
    ax_risk.grid()

    ax_loss.plot(range(1, k_upper), [p.get('test_loss') for p in preds], label="loss", marker=".")
    ax_loss.legend()
    ax_loss.set_xticks(range(1, k_upper))
    ax_loss.set_ylabel("empirical loss")
    ax_loss.set_xlabel("k")
    ax_loss.grid()

    plt.show()

if __name__ == "__main__":
    main()
# %%
