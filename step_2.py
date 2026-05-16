import random
from step_1 import load_iris


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


if __name__ == "__main__":
    random.seed(42)
    iris = load_iris()
    train, val, test = split_data(iris)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    X_train, y_train = separate_features_labels(train)
    X_val, y_val = separate_features_labels(val)
    X_test, y_test = separate_features_labels(test)
