import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, x_train, y_train, k):
        self.x_train = x_train
        self.y_train = y_train
        self.k = k

    def predict(self, x_test):
        y_pred = []
        # iterate through every test entry to be predicted
        for _, xi in x_test.iterrows():
            # calculate the euclidean distance between each row in x_train and xi
            dist_mat = cdist(np.expand_dims(xi.to_numpy(), axis=0), self.x_train, metric='euclidean')
            # partition the array such that the smallest k elements are in [:self.k]
            ismallest = np.argpartition(dist_mat, self.k, axis=None)[:self.k]
            # take the label of the smallest indexes of these k closest points
            smallest = self.y_train.iloc[ismallest]
            # take the label that appears most often or in case of a tie take one of these most frequent ones at random
            y = smallest.mode().sample()
            y_pred.append(y.values[0])
        return pd.DataFrame(y_pred)
