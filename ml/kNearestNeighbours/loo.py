from kNN import kNearestNeighbors
from typing import Optional
import numpy as np


class LeaveOneOut:
    def __init__(self):
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X.copy()
        self.y = y.copy()

    def findk(self) -> int:
        n = len(self.y)
        max_k = min(49, n - 1)
        losses = {}

        for k in range(1, max_k + 1, 2):
            predicted = []

            for i in range(len(self.y)):
                x_deleted = self.X[i]

                new_X = np.delete(self.X, i, axis=0)
                new_y = np.delete(self.y, i)

                model = kNearestNeighbors(k=k)
                model.fit(X_train=new_X, y_train=new_y)
                predicted.append(model.predict(x_deleted)[0])

            predicted = np.array(predicted)
            losses[k] = np.mean(predicted != self.y)

        return min(losses, key=losses.get)
