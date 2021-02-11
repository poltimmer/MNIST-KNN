import pandas as pd
from KNN import KNN
# simple accuracy score
def acc_score(y_test, y_pred):
    return (y_test == y_pred).mean()

def leave_one_out(X, y, k):
    out = []
    for ix, row in X.iterrows():
        # leave one out of X and y, this becomes training
        X_t, y_t = X.drop(index=ix), y.drop(index=ix)
        # the one left out is used to validate
        X_v, y_v = X.iloc[[ix]], y.iloc[[ix]]
        # create model
        model = KNN(X_t, y_t, 5)
        #predict the value using the validation row
        val = model.predict(X_v).values[0]
        out.append(
            {
                "index":ix,
                'predicted': val,
                'true': y_v.values[0],
                'correct': (y_v.values[0]==val)*1
            }
        )
    #return out #returns full output
    return sum(o.get('correct') for o in out)/len(out) # returns accuraccy 



