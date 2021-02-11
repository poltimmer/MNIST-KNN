#%%
import pandas as pd
from KNN import KNN
from matplotlib import pyplot as plt

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
def get_loss(y_real, y_pred):
    loss = 0
    for i, yi in y_real.iteritems():
        if yi != y_pred[0][i]:
            loss += 1
    print("Loss: " + str(loss))
    return loss

#%%
# empirical risk
emp_risk = loss/len(y_pred)
print("Empirical risk: " + str(emp_risk))


# %%
train_loss = []
test_loss = []
train_preds = []
test_preds = []
for k in range(1, 21):
    model = KNN(x_train, y_train, k)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    train_preds.append(train_pred)
    test_preds.append(test_pred)
    train_loss.append(get_loss(y_train, train_pred))
    test_loss.append(get_loss(y_test, test_pred))

# %%
# train_risk = [loss/len(y_train) for loss in train_loss]
# test_risk = [loss/len(y_test) for loss in test_loss]

train_risk = [(y_train != pred).mean() for pred in train_preds]
test_risk = [(y_test != pred).mean() for pred in test_preds]

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot()
ax.plot(range(1, 21), train_risk, label="train", marker=".")
ax.plot(range(1, 21), test_risk, label="test", marker=".")
ax.legend()
ax.set_xticks(range(1,21))
ax.set_ylabel("empirical risk")
ax.set_xlabel("k")
ax.grid()
plt.show()
# %%
