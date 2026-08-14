import pandas as pd
import numpy as np
import random
from model import Model

def train_test_split(X, y, test_size):
    X_train, X_test, y_train, y_test = [], [], [], []

    n = len(X)
    num_ints = int((1 - test_size) * n)
    random_indices = random.sample(range(n), num_ints)

    for i in range(n):
        if(i in random_indices):
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])
    return X_train, X_test, y_train, y_test

def one_hot(label, n=3):
    vec = [0.0] * n
    vec[int(label)] = 1.0
    return vec

def standardize(X):
    X = np.array(X)

    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)

    return ((X - means) / stds).tolist()

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
df = pd.read_csv(url)

df['species'] = df['species'].replace("setosa", 0.0)
df['species'] = df['species'].replace("versicolor", 1.0)
df['species'] = df['species'].replace("virginica", 2.0)

X = df.drop('species', axis = 1)
y = df['species']

X = X.values.tolist()
y = y.values.tolist()
X = standardize(X)
y = [one_hot(v) for v in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = Model(4, 8, 3)
model.train(X_train, y_train, lr=0.01, num_epochs=200, visualize=True)
