"""
Train the from-scratch MLP on the Iris (flower) dataset and save a loss curve.

This script does not modify library modules. It imports `Model` and reuses the
same training recipe as `dataloader.py`, then writes:

    assets/iris_loss_curve.png
"""

from __future__ import annotations

import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running as: python examples/train_iris.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model import Model


IRIS_URL = (
    "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/"
    "0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
)


def one_hot(label: float, n: int = 3) -> list[float]:
    vec = [0.0] * n
    vec[int(label)] = 1.0
    return vec


def train_test_split(X, y, test_size: float = 0.2, seed: int = 42):
    random.seed(seed)
    n = len(X)
    n_train = int((1 - test_size) * n)
    train_idx = set(random.sample(range(n), n_train))

    X_train, X_test, y_train, y_test = [], [], [], []
    for i in range(n):
        if i in train_idx:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])
    return X_train, X_test, y_train, y_test


def standardize_fit_transform(X_train, X_test):
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds == 0] = 1.0
    return ((X_train - means) / stds).tolist(), ((X_test - means) / stds).tolist()


def accuracy(model: Model, X, y) -> float:
    correct = 0
    for i, x in enumerate(X):
        pred = model.forward(x).as_list()
        if pred.index(max(pred)) == y[i].index(max(y[i])):
            correct += 1
    return correct / len(X)


def load_iris():
    df = pd.read_csv(IRIS_URL)
    species_map = {"setosa": 0.0, "versicolor": 1.0, "virginica": 2.0}
    df["species"] = df["species"].map(species_map)
    X = df.drop("species", axis=1).values.tolist()
    y = [one_hot(v) for v in df["species"].tolist()]
    return X, y


def main():
    X, y = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test = standardize_fit_transform(X_train, X_test)

    model = Model(4, 8, 3)
    epochs, losses = [], []
    num_epochs = 200
    lr = 0.01

    for epoch in range(num_epochs):
        net_loss = 0.0
        for i, x in enumerate(X_train):
            net_loss += model.backward(x, y_train[i], lr)
        avg = net_loss / len(X_train)
        epochs.append(epoch + 1)
        losses.append(avg)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}: Loss = {avg:.6f}")

    train_acc = accuracy(model, X_train, y_train)
    test_acc = accuracy(model, X_test, y_test)
    print(f"Train accuracy: {train_acc:.2%}")
    print(f"Test accuracy:  {test_acc:.2%}")

    assets = os.path.join(ROOT, "assets")
    os.makedirs(assets, exist_ok=True)
    out_path = os.path.join(assets, "iris_loss_curve.png")

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, losses, color="#1f4e79", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Iris training loss (from-scratch MLP)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved loss curve to {out_path}")


if __name__ == "__main__":
    main()
