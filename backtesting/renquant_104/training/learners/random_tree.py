"""Random Tree learner for classification."""

import numpy as np


class RTLearner:
    """Random decision tree that selects split features randomly."""

    def __init__(self, leaf_size: int = 1):
        self.leaf_size = leaf_size
        self.tree: np.ndarray | None = None

    def add_evidence(self, data_x: np.ndarray, data_y: np.ndarray) -> None:
        self.tree = self._build_tree(data_x, data_y)

    def _build_tree(self, data_x: np.ndarray, data_y: np.ndarray) -> np.ndarray:
        if data_x.shape[0] <= self.leaf_size or np.all(data_y == data_y[0]):
            return np.array([[-1, self._mode(data_y), -1, -1]])

        feat = np.random.randint(data_x.shape[1])
        split_val = np.median(data_x[:, feat])

        left_mask = data_x[:, feat] <= split_val
        if np.all(left_mask) or np.all(~left_mask):
            return np.array([[-1, self._mode(data_y), -1, -1]])

        left = self._build_tree(data_x[left_mask], data_y[left_mask])
        right = self._build_tree(data_x[~left_mask], data_y[~left_mask])
        root = np.array([[feat, split_val, 1, left.shape[0] + 1]])
        return np.vstack((root, left, right))

    @staticmethod
    def _mode(data_y: np.ndarray) -> float:
        vals, counts = np.unique(data_y, return_counts=True)
        return float(vals[np.argmax(counts)])

    def query(self, data_x: np.ndarray) -> np.ndarray:
        return np.array([self._traverse(row) for row in data_x])

    def _traverse(self, row: np.ndarray) -> float:
        idx = 0
        while True:
            feat, split_val, left_off, right_off = self.tree[idx]
            if feat == -1:
                return split_val
            idx += int(left_off) if row[int(feat)] <= split_val else int(right_off)
