from KNN import KNN
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, cpu_count
from functools import partial
from itertools import chain
from crossval import leave_one_out, acc_score, leave_one_out_smart, leave_one_out_legacy
from matplotlib import pyplot as plt

def worker(k, x_train, y_train):
    return [{'accuracy': leave_one_out_legacy(x_train, y_train, k, metric='minkowski', p=p), 'k':k, 'p':p} for p in tqdm(range(1, 16))]

def main():

        # load training data
        train = pd.read_csv("input/MNIST_train_small.csv", header=None)
        y_train = train[0]
        x_train = train.drop(columns=0)

        # y_train = y_train[:200]
        # x_train = x_train[:200]

        # train model
        # scores = [{'accuracy':leave_one_out_smart(x_train, y_train, k, metric='minkowski', p=p), 'k':k, 'p':p} for p in range(5, 7) for k in range(1, 3)]
        scores2d = process_map(partial(worker, x_train=x_train, y_train=y_train), range(1, 11), max_workers=10)
        # print(leave_one_out_legacy(x_train, y_train, 3, metric='minkowski', p=1))
        print(scores2d)

        results = pd.DataFrame([[score['accuracy'] for score in scores] for scores in scores2d])

        print(results)

        results.to_csv('./c_results.csv')


        scores2d_2 = process_map(partial(worker, x_train=x_train, y_train=y_train), range(11, 21), max_workers=10)

        print(scores2d_2)

        results2 = pd.DataFrame([[score['accuracy'] for score in scores] for scores in scores2d_2])

        print(results2)

        results.to_csv('./c_results2.csv')


        # print k and p with highest accuracy
        # best_score = max(scores, key=lambda x:x['accuracy'])
        # print(best_score)                           

if __name__ == "__main__":
    main()  

