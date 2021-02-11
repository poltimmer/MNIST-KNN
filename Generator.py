import numpy as np
import pandas as pd


def genRandom(n):
    df = pd.DataFrame(np.random.randint(256, size=(n, 785)))
    df[0] = np.random.randint(10, size=(n, 785))
    return df
