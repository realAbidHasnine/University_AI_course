import numpy as np
from collections import Counter
from typing import Literal, Optional


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.sum(np.abs(x1 - x2))


class KNN:

    def __init__(
        self,
        k: int = 3,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
    ):
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k
        self.metric = metric
        self._distance_fn = (
            euclidean_distance if metric == "euclidean" else manhattan_distance
        )
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have mismatched row counts: {X.shape[0]} vs {y.shape[0]}"
            )
        if self.k > X.shape[0]:
            raise ValueError(
                f"k={self.k} exceeds number of training samples ({X.shape[0]})"
            )
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Must call fit before predict")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained on {self.X_train.shape[1]}"
            )
        return np.array([self._predict(x) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Must call fit before predict_proba")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained on {self.X_train.shape[1]}"
            )
        return np.array([self._predict_proba(x) for x in X])

    def _predict(self, x: np.ndarray) -> int:
        distances = [self._distance_fn(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _predict_proba(self, x: np.ndarray) -> dict:
        distances = [self._distance_fn(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = self.y_train[k_indices]
        counts = Counter(k_nearest_labels)
        total = sum(counts.values())
        return {cls: cnt / total for cls, cnt in counts.items()}
