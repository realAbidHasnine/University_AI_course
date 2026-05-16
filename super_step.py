import csv
import math
import random
from collections import Counter


# ===== Step 1: Load dataset into a 2D Python list =====

def load_iris(path="IRIS.csv"):
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        data = []
        for row in reader:
            sepal_len = float(row[0])
            sepal_wid = float(row[1])
            petal_len = float(row[2])
            petal_wid = float(row[3])
            species = row[4]
            data.append([sepal_len, sepal_wid, petal_len, petal_wid, species])
    return data


# ===== Step 2: Random 70/15/15 split =====

def split_data(data):
    train, val, test = [], [], []
    for sample in data:
        r = random.random()
        if r <= 0.7:
            train.append(sample)
        elif r <= 0.85:
            val.append(sample)
        else:
            test.append(sample)
    return train, val, test


def separate_features_labels(rows):
    X = [row[:4] for row in rows]
    y = [row[4] for row in rows]
    return X, y


# ===== Step 3: KNN implementation from scratch =====

def euclidean_distance(a, b):
    total = 0.0
    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2
    return math.sqrt(total)


def knn_predict(X_train, y_train, x, k):
    distances = []
    for i, x_train in enumerate(X_train):
        dist = euclidean_distance(x, x_train)
        distances.append((dist, y_train[i]))
    distances.sort(key=lambda pair: pair[0])
    k_nearest = distances[:k]
    k_labels = [label for _, label in k_nearest]
    majority = Counter(k_labels).most_common(1)[0][0]
    return majority


def compute_accuracy(X_train, y_train, X_val, y_val, k):
    correct = 0
    for i, x in enumerate(X_val):
        pred = knn_predict(X_train, y_train, x, k)
        if pred == y_val[i]:
            correct += 1
    return (correct / len(y_val)) * 100


# ===== Class-based KNN (ported from basic_KNN.py) =====

class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# ===== Step 4 & 5: Tune k, pick best, evaluate on test =====

def main():
    k_values = [1, 3, 5, 10, 15]

    random.seed(42)
    data = load_iris()
    train, val, test = split_data(data)

    X_train, y_train = separate_features_labels(train)
    X_val, y_val = separate_features_labels(val)
    X_test, y_test = separate_features_labels(test)

    print(f"Dataset: {len(data)} samples")
    print(f"Splits -> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print()

    print("K Value   Validation Accuracy (%)")
    print("------   -----------------------")
    best_k = k_values[0]
    best_acc = 0.0

    for k in k_values:
        acc = compute_accuracy(X_train, y_train, X_val, y_val, k)
        print(f"  {k}            {acc:.2f}")
        if acc > best_acc:
            best_acc = acc
            best_k = k

    print(f"\nBest k = {best_k} (validation accuracy: {best_acc:.2f}%)")

    correct = 0
    for i, x in enumerate(X_test):
        pred = knn_predict(X_train, y_train, x, best_k)
        if pred == y_test[i]:
            correct += 1

    test_acc = (correct / len(X_test)) * 100
    print(f"Test Accuracy with k={best_k}: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
