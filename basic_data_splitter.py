import numpy as np
import numbers
import math
import random


def _approximate_mode(class_counts, n_draws, rng):
    continuous = class_counts / class_counts.sum() * n_draws
    floored = np.floor(continuous)
    need_to_add = int(n_draws - floored.sum())
    if need_to_add > 0:
        remainder = continuous - floored
        values = np.sort(np.unique(remainder))[::-1]
        for value in values:
            (inds,) = np.where(remainder == value)
            add_now = min(len(inds), need_to_add)
            inds = rng.choice(inds, size=add_now, replace=False)
            floored[inds] += 1
            need_to_add -= add_now
            if need_to_add == 0:
                break
    return floored.astype(int)


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


def train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
):
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as a train/test split")

    n_samples = len(arrays[0])
    for arr in arrays:
        if len(arr) != n_samples:
            raise ValueError("All arrays must have the same length")

    if test_size is None and train_size is None:
        test_size = 0.25

    if isinstance(test_size, numbers.Integral):
        n_test = int(test_size)
    elif test_size is None:
        n_test = 0
    else:
        n_test = int(math.ceil(n_samples * test_size))

    if isinstance(train_size, numbers.Integral):
        n_train = int(train_size)
    elif train_size is None:
        n_train = n_samples - n_test
    else:
        n_train = int(n_samples * train_size)

    if shuffle:
        if stratify is not None:
            stratify = np.asarray(stratify)
            if stratify.ndim == 2:
                stratify = np.array([" ".join(row.astype("str")) for row in stratify])
            classes, y_indices, class_counts = np.unique(
                stratify, return_inverse=True, return_counts=True
            )
            n_classes = len(classes)

            if np.min(class_counts) < 2:
                raise ValueError(
                    "The least populated classes in y have only 1"
                    " member, which is too few."
                )
            if n_train < n_classes:
                raise ValueError(
                    "The train_size = %d should be greater or "
                    "equal to the number of classes = %d" % (n_train, n_classes)
                )
            if n_test < n_classes:
                raise ValueError(
                    "The test_size = %d should be greater or "
                    "equal to the number of classes = %d" % (n_test, n_classes)
                )

            class_indices = np.split(
                np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
            )

            rng = np.random.RandomState(random_state)

            n_i = _approximate_mode(class_counts, n_train, rng)
            class_counts_remaining = class_counts - n_i
            t_i = _approximate_mode(class_counts_remaining, n_test, rng)

            train = []
            test = []
            for i in range(n_classes):
                permutation = rng.permutation(class_counts[i])
                perm_indices_class_i = class_indices[i].take(permutation, mode="clip")
                train.extend(perm_indices_class_i[: n_i[i]])
                test.extend(perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]])

            train_idx = rng.permutation(train)
            test_idx = rng.permutation(test)
        else:
            rng = np.random.RandomState(random_state)
            permutation = rng.permutation(n_samples)
            test_idx = permutation[:n_test]
            train_idx = permutation[n_test : n_test + n_train]
    else:
        if stratify is not None:
            raise ValueError(
                "Stratified train/test split is not implemented for shuffle=False"
            )
        train_idx = np.arange(n_train)
        test_idx = np.arange(n_train, n_train + n_test)

    result = []
    for arr in arrays:
        arr = np.asarray(arr)
        result.append(arr[train_idx])
        result.append(arr[test_idx])

    return result
