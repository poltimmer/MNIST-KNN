#%%
import pandas as pd
from KNN import KNN

# load training data
train = pd.read_csv("input/MNIST_train_small.csv", header=None)
y_train = train[0]
x_train = train.drop(columns=0)

# load test data
test = pd.read_csv("input/MNIST_test_small.csv", header=None)
y_test = test[0]
x_test = test.drop(columns=0)

# perform predictions
y_pred = KNN(x_train, y_train, 5).predict(x_test)

# write output to file, such that metric calculations can be implemented using this file without running KNN
y_pred.to_csv("output/pred_small.csv", header=False)

#%%
# simple 0/1 loss function
loss = 0
for i, yi in y_test.iteritems():
    if yi != y_pred[0][i]:
        loss += 1
print("Loss: " + str(loss))

#%%
# empirical risk
emp_risk = loss/len(y_pred)
print("Empirical risk: " + str(emp_risk))

