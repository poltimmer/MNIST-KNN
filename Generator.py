import numpy as np
import pandas as pd


def genRandom(n):
    df = pd.DataFrame(np.random.randint(256, size=(n, 785)))
    df[0] = np.random.randint(10, size=(n, 785))
    return df

#%%
import pandas as pd
import Generator
import time
from KNN import KNN

# generate random data
random1 = Generator.genRandom(60000)
y_random1 = random1[0]
x_random1 = random1.drop(columns=0)
random2 = Generator.genRandom(10000)
y_random2 = random2[0]
x_random2 = random2.drop(columns=0)

print("start predicting")
start = time.time()
y_pred = KNN(x_random1, y_random1, 11, kd_tree=True).predictKDTree(x_random2)
end = time.time()
print("stop predicting", end - start)
