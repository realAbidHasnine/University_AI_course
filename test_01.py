import numpy as np
from basic_KNN import KNN
from basic_data_splitter import train_test_split

data = np.loadtxt("IRIS.csv", delimiter=",", skiprows=1, dtype=str)
X = data[:, :4].astype(float)
y = data[:, 4]

classes = np.unique(y)
y_encoded = np.zeros(len(y), dtype=int)
for i, c in enumerate(classes):
    y_encoded[y == c] = i

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

knn = KNN(k=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.4f} ({np.sum(predictions == y_test)}/{len(y_test)})")
