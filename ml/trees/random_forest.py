import numpy as np
from typing import Optional
from decision_tree_rsm import DecisionTreeRSM

class RandomForest:
    def __init__(self, n_estimators: int = 100):
        self.trees: Optional[list[DecisionTreeRSM]] = None
        self.n_estimators = n_estimators
        self.X_train: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.trees = []

        for _ in range(self.n_estimators):
            indices = np.random.choice(
                X_train.shape[0],
                size=X_train.shape[0],
                replace=True
            )

            X_bootstrap = X_train[indices]
            y_bootstrap = y_train[indices]

            tree = DecisionTreeRSM()
            tree.fit(X_bootstrap, y_bootstrap)

            self.trees.append(tree)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        raw_predictions = np.array([tree.predict(X_test) for tree in self.trees])
        predictions = []

        for prediction in raw_predictions.T:
            values, counts = np.unique(prediction, return_counts=True)
            predictions.append(values[np.argmax(counts)])

        return np.array(predictions)

    def classification_report(self, X_test: np.ndarray, y_true: np.ndarray) -> str:
        if len(self.trees) == 0:
            raise ValueError("Forest is not fitted yet")

        y_true = np.asarray(y_true).ravel()
        y_pred = self.predict(X_test).ravel()

        if len(y_true) != len(y_pred):
            raise ValueError("X_test and y_true must contain the same number of objects")

        labels = np.unique(np.concatenate([y_true, y_pred]))

        def safe_div(a, b):
            return a / b if b != 0 else 0.0

        rows = []

        for label in labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fp = np.sum((y_true != label) & (y_pred == label))
            fn = np.sum((y_true == label) & (y_pred != label))

            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            f1 = safe_div(2 * precision * recall, precision + recall)
            support = np.sum(y_true == label)

            rows.append((label, precision, recall, f1, support))

        accuracy = np.mean(y_true == y_pred)

        macro_precision = np.mean([row[1] for row in rows])
        macro_recall = np.mean([row[2] for row in rows])
        macro_f1 = np.mean([row[3] for row in rows])

        total_support = np.sum([row[4] for row in rows])

        weighted_precision = safe_div(
            np.sum([row[1] * row[4] for row in rows]),
            total_support
        )
        weighted_recall = safe_div(
            np.sum([row[2] * row[4] for row in rows]),
            total_support
        )
        weighted_f1 = safe_div(
            np.sum([row[3] * row[4] for row in rows]),
            total_support
        )

        report = []
        report.append(
            f"{'class':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
        )
        report.append("-" * 60)

        for label, precision, recall, f1, support in rows:
            report.append(
                f"{str(label):>12} "
                f"{precision:>10.3f} "
                f"{recall:>10.3f} "
                f"{f1:>10.3f} "
                f"{support:>10}"
            )

        report.append("-" * 60)
        report.append(
            f"{'accuracy':>12} "
            f"{'':>10} "
            f"{'':>10} "
            f"{accuracy:>10.3f} "
            f"{total_support:>10}"
        )
        report.append(
            f"{'macro avg':>12} "
            f"{macro_precision:>10.3f} "
            f"{macro_recall:>10.3f} "
            f"{macro_f1:>10.3f} "
            f"{total_support:>10}"
        )
        report.append(
            f"{'weighted avg':>12} "
            f"{weighted_precision:>10.3f} "
            f"{weighted_recall:>10.3f} "
            f"{weighted_f1:>10.3f} "
            f"{total_support:>10}"
        )

        return "\n".join(report)





