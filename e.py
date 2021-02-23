
# %%
import pandas as pd
import numpy as np
from KNN import KNN
from matplotlib import pyplot as plt
from func import acc_score, leave_one_out
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

#Commented out all testing data for the sake of question D.

def main():

    train = pd.read_csv("input/MNIST_train.csv", header=None)
    y_train = train[0]
    x_train = train.drop(columns=0)

    # test = pd.read_csv("input/MNIST_train.csv", header=None)
    # y_test = test[0]
    # x_test = test.drop(columns=0)
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
    # x_test = pd.DataFrame(pca.transform(x_test))
    print('Done preprocessing, starting estimations')

    #PREDICTING
    preds = []
    for k in range(1, k_upper):
        print('scoring for k =' + str(k))
        model = KNN(x_train, y_train, k)
        train_pred = model.predict_concurrent(x_train, metric='rogerstanimoto')
        #test_pred = model.predict_concurrent(x_test, metric='rogerstanimoto')
        preds.append({"train": train_pred})#, "test": test_pred})

    train_risk = [(y_train != pred["train"]).mean() for pred in preds]
    #test_risk = [(y_test != pred["test"]).mean() for pred in preds]

    train_loss = [sum((y_train != pred["train"])) for pred in preds]
    #test_loss= [sum((y_test != pred["test"])) for pred in preds]

    fig = plt.figure(figsize=(10,6))
    ax_risk = fig.add_subplot(122)
    ax_loss = fig.add_subplot(121)

    ax_risk.plot(range(1, k_upper), train_risk, label="train", marker=".")
    #ax_risk.plot(range(1, k_upper), test_risk, label="test", marker=".")
    ax_risk.legend()
    ax_risk.set_xticks(range(1, k_upper))
    ax_risk.set_ylabel("empirical risk")
    ax_risk.set_xlabel("k")
    ax_risk.grid()

    ax_loss.plot(range(1, k_upper), train_loss, label="train", marker=".")
    #ax_loss.plot(range(1, k_upper), test_loss, label="test", marker=".")
    ax_loss.legend()
    ax_loss.set_xticks(range(1, k_upper))
    ax_loss.set_ylabel("empirical loss")
    ax_loss.set_xlabel("k")
    ax_loss.grid()

    plt.show()

if __name__ == "__main__":
    main()
# %%
