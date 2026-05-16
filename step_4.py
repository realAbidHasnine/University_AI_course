from step_1 import load_iris
from step_2 import split_data, separate_features_labels
from step_3 import compute_accuracy
import random


k_values = [1, 3, 5, 10, 15]

random.seed(42)
data = load_iris()
train, val, test = split_data(data)
X_train, y_train = separate_features_labels(train)
X_val, y_val = separate_features_labels(val)

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

print(f"\nBest k: {best_k} with validation accuracy: {best_acc:.2f}%")
