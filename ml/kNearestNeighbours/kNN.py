import numpy as np
from typing import Optional
from sklearn.datasets import make_blobs


class kNearestNeighbors:
    def __init__(self, k = 5, weights = 'uniform'):
        self.k = k
        self.weights = weights

        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None

        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    @staticmethod
    def _distance(x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self._mu) / self._sigma

    def _fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        self._mu = np.mean(X_train, axis=0)
        self._sigma = np.std(X_train, axis=0)
        self._sigma[self._sigma == 0] = 1.0

        return (X_train - self._mu) / self._sigma

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.y = y_train.copy()
        self.X = self._fit_transform(X_train)

    def _predict_one(self, X_test: np.ndarray) -> int:
        distances = np.array([self._distance(X_test, x_train) for x_train in self.X])
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = self.y[nearest_indices]

        if self.weights == 'uniform':
            classes, counts = np.unique(nearest_labels, return_counts=True)
            return classes[np.argmax(counts)]

        nearest_distances = distances[nearest_indices]
        eps = 1e-8
        weights = 1 / (nearest_distances + eps)
        classes = np.unique(nearest_labels)
        classes_score = []

        for cls in classes:
            score = np.sum(weights[nearest_labels == cls])
            classes_score.append(score)

        return classes[np.argmax(classes_score)]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_test = np.atleast_2d(X_test)
        x_test_used = self._transform(X_test)

        return np.array([self._predict_one(x) for x in x_test_used])

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        y_pred = self.predict(X_test)
        y_test = np.asarray(y_test)

        return np.mean(y_pred == y_test)

    def confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_test = np.asarray(y_test)
        y_pred = self.predict(X_test)

        labels = np.unique(np.concatenate([y_test, y_pred]))
        matrix = np.zeros((len(labels), len(labels)), dtype=int)

        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                matrix[i, j] = np.sum((y_test == true_label) & (y_pred == pred_label))

        return matrix, labels

    def precision_recall_f1(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_test = np.asarray(y_test)
        matrix, labels = self.confusion_matrix(X_test, y_test)

        precisions = []
        recalls = []
        f1_scores = []
        supports = []

        for i in range(len(labels)):
            tp = matrix[i, i]
            fp = np.sum(matrix[:, i]) - tp
            fn = np.sum(matrix[i, :]) - tp
            support = np.sum(matrix[i, :])

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            supports.append(support)

        return {
            "labels": labels,
            "precision_per_class": np.array(precisions),
            "recall_per_class": np.array(recalls),
            "f1_per_class": np.array(f1_scores),
            "support_per_class": np.array(supports),
            "precision_macro": float(np.mean(precisions)),
            "recall_macro": float(np.mean(recalls)),
            "f1_macro": float(np.mean(f1_scores)),
            "accuracy": self.score(X_test, y_test)
        }

    def classification_report(self, X_test: np.ndarray, y_test: np.ndarray) -> str:
        metrics = self.precision_recall_f1(X_test, y_test)

        labels = metrics["labels"]
        precisions = metrics["precision_per_class"]
        recalls = metrics["recall_per_class"]
        f1_scores = metrics["f1_per_class"]
        supports = metrics["support_per_class"]

        total_support = int(np.sum(supports))
        accuracy = metrics["accuracy"]
        precision_macro = metrics["precision_macro"]
        recall_macro = metrics["recall_macro"]
        f1_macro = metrics["f1_macro"]

        lines = []
        lines.append(f"{'class':>10} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
        lines.append("")

        for label, p, r, f1, s in zip(labels, precisions, recalls, f1_scores, supports):
            lines.append(f"{str(label):>10} {p:>10.3f} {r:>10.3f} {f1:>10.3f} {s:>10}")

        lines.append("")
        lines.append(f"{'accuracy':>10} {'':>10} {'':>10} {accuracy:>10.3f} {total_support:>10}")
        lines.append(
            f"{'macro avg':>10} {precision_macro:>10.3f} {recall_macro:>10.3f} {f1_macro:>10.3f} {total_support:>10}")

        return "\n".join(lines)
