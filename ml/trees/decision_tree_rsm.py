import numpy as np
from typing import Optional

class Node:
    def __init__(
            self,
            threshold: Optional[float] = None,
            feature: Optional[int] = None,
            is_leaf: bool = False,
        ):
        self.threshold = threshold
        self.feature = feature
        self.is_leaf = is_leaf
        self.right: Optional[Node] = None
        self.left: Optional[Node] = None
        self._class: int = None


class DecisionTreeRSM:
    def __init__(self):
        self.root: Optional[Node] = None
        self.X_train: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    @staticmethod
    def H(x: np.ndarray) -> float:
        n = len(x)
        _, counts = np.unique(x, return_counts=True)

        return -np.sum(counts / n * np.log(counts / n))

    @staticmethod
    def _weightedEntropy(L: np.ndarray, R: np.ndarray) -> float:
        return len(L) / (len(L) + len(R)) * DecisionTreeRSM.H(L) + len(R) / (len(L) + len(R)) * DecisionTreeRSM.H(R)

    @staticmethod
    def _findBestParams(X: np.ndarray, y: np.ndarray, max_features: int) -> tuple:
        min_criterion = np.inf
        best_threshold: float = None
        best_feature: int = None

        feature_indices = np.random.choice(
            X.shape[1],
            size=max_features,
            replace=False
        )

        for feature_i in feature_indices:
            features = X.T[feature_i]
            indices = np.argsort(features)
            features = np.sort(features)
            labels = y[indices]

            for i in range(len(features) - 1):
                if features[i] == features[i + 1]:
                    continue
                threshold = (features[i] + features[i + 1]) / 2

                left = labels[features < threshold]
                right = labels[features >= threshold]

                information = DecisionTreeRSM._weightedEntropy(left, right)
                if information < min_criterion:
                    best_feature = feature_i
                    best_threshold = threshold
                    min_criterion = information

        return (best_feature, best_threshold)

    def buildTree(self, X, y: np.ndarray, current_node: Node):
        if len(np.unique(y)) == 1:
            current_node.is_leaf = True
            vals, counts = np.unique(y, return_counts=True)
            current_node._class = vals[np.argmax(counts)]
            return

        k = max(1, int(np.sqrt(X.shape[1])))
        best_params = DecisionTreeRSM._findBestParams(X, y, k)

        if best_params[0] is None:
            current_node.is_leaf = True
            vals, counts = np.unique(y, return_counts=True)
            current_node._class = vals[np.argmax(counts)]
            return

        mask_l = X.T[best_params[0]] < best_params[1]
        mask_r = X.T[best_params[0]] >= best_params[1]
        X_L = X[mask_l]
        X_R = X[mask_r]
        y_L = y[mask_l]
        y_R = y[mask_r]

        new_left = Node()
        new_right = Node()

        current_node.left = new_left
        current_node.right = new_right
        current_node.feature = best_params[0]
        current_node.threshold = best_params[1]

        self.buildTree(X_L, y_L, new_left)
        self.buildTree(X_R, y_R, new_right)

    def fit(self, X_train: np.ndarray, y: np.ndarray):
        self.X_train = X_train.copy()
        self.y = y.copy()
        self.root = Node()
        self.buildTree(self.X_train, self.y, self.root)

    def _predict_one(self, X_test: np.ndarray) -> int:
        node = self.root

        while not node.is_leaf:
            feature = node.feature
            threshold = node.threshold

            if X_test[feature] < threshold:
                node = node.left
            else:
                node = node.right

        return node._class

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_test = np.atleast_2d(X_test)

        return np.array([self._predict_one(x) for x in X_test])

    def classification_report(self, X_test: np.ndarray, y_true: np.ndarray) -> str:
        if self.root is None:
            raise ValueError("Tree is not fitted yet")

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

