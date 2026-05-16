import numpy as np
import pandas as pd
from basic_KNN import KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("IRIS.csv")

# Rename columns (optional if your CSV already has headers)
data.columns = ["f1", "f2", "f3", "f4", "label"]

# Features and target
X = data[["f1", "f2", "f3", "f4"]].values
y = data["label"].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Test different k values
for k in [10, 10, 10]:
    knn = KNN(k=k)
    knn.fit(X_train, y_train)

    preds = knn.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"basic_KNN.py | k={k} | accuracy={acc:.4f}")