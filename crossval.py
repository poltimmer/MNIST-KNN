import pandas as pd
from KNN import KNN
from tqdm.contrib.concurrent import process_map, cpu_count
from functools import partial
# simple accuracy score
def acc_score(y_test, y_pred):
    return (y_test == y_pred).mean()

def leave_one_out_legacy(X, y, k):
    out = []
    for ix in range(len(X)):
        # leave one out of X and y, this becomes training
        X_t, y_t = X.drop(index=ix), y.drop(index=ix)
        # the one left out is used to validate
        X_v, y_v = X.iloc[ix], y.iloc[ix]
        # create model
        model = KNN(X_t, y_t, k)
        #predict the value using the validation row
        val = model.predict_single(X_v)
        out.append(
            {
                # "index":ix,
                # 'predicted': val,
                # 'true': y_v.values[0],
                'correct': (y_v==val)*1
            }
        )
    #return out #returns full output
    return sum(o.get('correct') for o in out)/len(out) # returns accuraccy 

def leave_one_out(X, y, k, metric='euclidean', p=None):
    crossval_partial = partial(leave_one_out_worker, X=X, y=y, k=k, metric=metric, p=p)
    out = process_map(crossval_partial, range(len(X)), max_workers=cpu_count()-2, chunksize=max(50, int(len(X)/(cpu_count()*2))))
    return sum(o.get('correct') for o in out)/len(out) # returns accuraccy 

def leave_one_out_worker(ix, X, y, k, metric, p=None):
    # leave one out of X and y, this becomes training
    X_t, y_t = X.drop(index=ix), y.drop(index=ix)
    # the one left out is used to validate
    X_v, y_v = X.iloc[ix], y.iloc[ix]
    # create model
    model = KNN(X_t, y_t, k)
    #predict the value using the validation row
    val = model.predict_single(X_v, metric=metric, p=p)
    return {
        # "index":ix,
        # 'predicted': val,
        # 'true': y_v.values[0],
        'correct': (y_v==val)*1
    }
    #return out #returns full output

def leave_one_out_smart(X, y, k, metric='euclidean', p=None):
    # Just as slow as leave_one_out(). 
    model = KNN(X, y, k)
    out = process_map(partial(model.leave_one_out, metric=metric, p=p), range(len(X)), max_workers=cpu_count()-2, chunksize=max(50, int(len(X)/(cpu_count()*2))))
    return (out == y).mean() # returns accuraccy 