#%%
from KNN import KNN
import pandas as pd
import numpy as np
from tqdm import tqdm
from crossval import leave_one_out, acc_score

def main():

        # load training data
        train = pd.read_csv("input/MNIST_train_small.csv", header=None)
        y_train = train[0]
        x_train = train.drop(columns=0)

        # train model
        scores = [{'accuracy':leave_one_out(x_train, y_train, k, metric='minkowski', p=p), 'k':k, 'p':p} for p in range(1, 16) for k in range(1, 21)]

        # print k and p with highest accuracy
        best_score = max(scores, key=lambda x:x['accuracy'])
        print(best_score)                           

if __name__ == "__main__":
    main()  

# %%
