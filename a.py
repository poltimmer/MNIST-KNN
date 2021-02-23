import pandas as pd
import numpy as np
from tqdm.contrib.concurrent import process_map, cpu_count
from functools import partial
from KNN import KNN
from matplotlib import pyplot as plt


def get_preds(k, x_train, y_train, x_test):
    model = KNN(x_train, y_train, k)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    return {"train": train_pred, "test": test_pred}

def main():
    # load training data
    train = pd.read_csv("input/MNIST_train_small.csv", header=None)
    y_train = train[0]
    x_train = train.drop(columns=0)

    # load test data
    test = pd.read_csv("input/MNIST_test_small.csv", header=None)
    y_test = test[0]
    x_test = test.drop(columns=0)

    # Upper limit for range iterator k. Set to maximum k-1. (21 gives max k = 20)
    k_upper = 21

    # Limit data size for debugging/development
    # x_train = x_train[:500]
    # y_train = y_train[:500]
    # x_test = x_test[:500]
    # y_test = y_test[:500]


    # TWO OPTIONS FOR CONCURRENCY:
    # get_preds_partial = partial(get_preds, x_train=x_train, y_train=y_train, x_test=x_test)
    # Prediction sets parallel (less overhead but doesn't divide as nicely due to lower amount of workers)
    # preds = process_map(get_preds_partial, range(1, k_upper), max_workers=cpu_count()-2)

    # Individual predictions parallel (divides more nicely due to larger amount of workers, but more overhead due to process pool creation)
    preds = []
    for k in range(1, k_upper):
        model = KNN(x_train, y_train, k)
        train_pred = model.predict_concurrent(x_train)
        test_pred = model.predict_concurrent(x_test)
        preds.append({"train": train_pred, "test": test_pred})


    train_risk = [(y_train != pred["train"]).mean() for pred in preds]
    test_risk = [(y_test != pred["test"]).mean() for pred in preds]

    train_loss = [sum((y_train != pred["train"])) for pred in preds]
    test_loss= [sum((y_test != pred["test"])) for pred in preds]

    fig = plt.figure(figsize=(10,6))
    ax_risk = fig.add_subplot(122)
    ax_loss = fig.add_subplot(121)

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

if __name__ == "__main__":
    main()