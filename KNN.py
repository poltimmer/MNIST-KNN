import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree


class KNN:
    def __init__(self, x_train, y_train, k, kd_tree=False, ball_tree=False):
        self.x_train = x_train
        self.y_train = y_train
        self.k = k
        self.kd_tree = KDTree(x_train) if kd_tree else None
        self.ball_tree = BallTree(x_train) if ball_tree else None

    def predictBallTree(self, x_test):
        _, indices = self.ball_tree.query(x_test, k=self.k)
        y_pred = [self.y_train.iloc[nearest].mode().sample().values[0] for nearest in indices]
        return pd.DataFrame(y_pred)

    def predictKDTree(self, x_test):
        _, indices = self.kd_tree.query(x_test, k=self.k)
        y_pred = [self.y_train.iloc[nearest].mode().sample().values[0] for nearest in indices]
        return pd.DataFrame(y_pred)

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
