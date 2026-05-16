import math
from collections import Counter


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


if __name__ == "__main__":
    from step_1 import load_iris
    from step_2 import split_data, separate_features_labels
    import random

    random.seed(42)
    data = load_iris()
    train, val, test = split_data(data)
    X_train, y_train = separate_features_labels(train)
    X_val, y_val = separate_features_labels(val)

    acc = compute_accuracy(X_train, y_train, X_val, y_val, k=5)
    print(f"Validation accuracy with k=5: {acc:.2f}%")
