import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
from tqdm.contrib.concurrent import process_map, cpu_count
from functools import partial

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

    def predict(self, x_test, metric='euclidean'):
        y_pred = []
        # iterate through every test entry to be predicted
        for _, xi in x_test.iterrows():
            # calculate the euclidean distance between each row in x_train and xi
            dist_mat = cdist(np.expand_dims(xi.to_numpy(), axis=0), self.x_train, metric=metric)[0]
            # partition the array such that the smallest k elements are in [:self.k]
            ismallest = np.argpartition(dist_mat, self.k)[:self.k]
            while True:
                # take the label of the smallest indexes of these k closest points
                smallest = self.y_train.iloc[ismallest]
                # take the labels that appear most often amongst these k closest points
                y = smallest.mode()
                # in case there is one, we use this label as the prediction
                if len(y) == 1:
                    y_pred.append(y.values[0])
                    break
                # else we ignore the point that was furthest away
                rem = max(ismallest, key=lambda i: dist_mat[i])
                ismallest = np.delete(ismallest, np.argwhere(ismallest == rem))
        return pd.Series(y_pred)

    def predict_concurrent(self, x_test, metric='euclidean'):
        predict_single_partial = partial(self.predict_single, metric=metric)
        y_pred = process_map(predict_single_partial, [xi for _, xi in x_test.iterrows()], max_workers=cpu_count()-2, chunksize=max(50, int(x_test.shape[0]/100)))
        # iterate through every test entry to be predicted

        return pd.Series(y_pred)

    def predict_single(self, xi, metric='euclidean', p=None):
        # calculate the euclidean distance between each row in x_train and xi
        if p:
            dist_mat = cdist(np.expand_dims(xi.to_numpy(), axis=0), self.x_train, metric=metric, p=p)[0]
        else: 
            dist_mat = cdist(np.expand_dims(xi.to_numpy(), axis=0), self.x_train, metric=metric)[0]
        # partition the array such that the smallest k elements are in [:self.k]
        ismallest = np.argpartition(dist_mat, self.k)[:self.k]
        while True:
            # take the label of the smallest indexes of these k closest points
            smallest = self.y_train.iloc[ismallest]
            # take the labels that appear most often amongst these k closest points
            y = smallest.mode()
            # in case there is one, we use this label as the prediction
            if len(y) == 1:
                return y.values[0]
            # else we ignore the point that was furthest away
            rem = max(ismallest, key=lambda i: dist_mat[i])
            ismallest = np.delete(ismallest, np.argwhere(ismallest == rem))

    def leave_one_out(self, i, metric='euclidean', p=None):
        y_train = self.y_train.drop(index=i)
        # calculate the euclidean distance between each row in x_train and xi
        if p:
            dist_mat = cdist(np.expand_dims(self.x_train.iloc[i].to_numpy(), axis=0), self.x_train.drop(index=i), metric=metric, p=p)[0]
        else: 
            dist_mat = cdist(np.expand_dims(self.x_train.iloc[i].to_numpy(), axis=0), self.x_train.drop(index=i), metric=metric)[0]
        # partition the array such that the smallest k elements are in [:self.k]
        ismallest = np.argpartition(dist_mat, self.k)[:self.k]
        while True:
            # take the label of the smallest indexes of these k closest points
            smallest = y_train.iloc[ismallest]
            # take the labels that appear most often amongst these k closest points
            y = smallest.mode()
            # in case there is one, we use this label as the prediction
            if len(y) == 1:
                return y.values[0]
            # else we ignore the point that was furthest away
            rem = max(ismallest, key=lambda i: dist_mat[i])
            ismallest = np.delete(ismallest, np.argwhere(ismallest == rem))
        


