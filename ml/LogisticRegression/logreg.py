import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Iterable

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class LogReg:
    @dataclass
    class Report:
        train_loss: list[float] = field(default_factory=list)
        val_loss: list[float] = field(default_factory=list)
        epochs_run: int = 0
        stopped_early: bool = False

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_epochs: int = 1000,
        tol: float = 1e-6,
        patience: int = 20,
        regularization: Optional[str] = "l2",
        reg_lambda: float = 0.0,
    ):
        reg = None if regularization is None else str(regularization).lower()
        if reg == "none":
            reg = None
        if reg not in (None, "l1", "l2"):
            raise ValueError("regularization must be one of: None, 'l1', 'l2'")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be > 0")
        if tol < 0:
            raise ValueError("tol must be >= 0")
        if patience <= 0:
            raise ValueError("patience must be > 0")
        if reg_lambda < 0:
            raise ValueError("reg_lambda must be >= 0")

        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.tol = float(tol)
        self.patience = int(patience)
        self.regularization = reg
        self.reg_lambda = float(reg_lambda)

        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.is_fitted: bool = False
        self.report = LogReg.Report()

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _check_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_features)")
        return X

    @staticmethod
    def _check_y(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        unique = np.unique(y)
        if not np.all(np.isin(unique, [0.0, 1.0])):
            raise ValueError("y must contain only 0 and 1")
        return y

    def _fit_scaler(self, X: np.ndarray) -> None:
        self._mu = np.mean(X, axis=0)
        self._sigma = np.std(X, axis=0)
        self._sigma[self._sigma == 0] = 1.0

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._mu is None or self._sigma is None:
            raise RuntimeError("Scaler is not fitted yet")
        return (X - self._mu) / self._sigma

    @staticmethod
    def _binary_cross_entropy(y: np.ndarray, p: np.ndarray) -> float:
        eps = np.finfo(float).eps
        p = np.clip(p, eps, 1.0 - eps)
        return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    def _regularization_loss(self) -> float:
        if self.regularization is None or self.reg_lambda == 0.0:
            return 0.0
        if self.regularization == "l2":
            return self.reg_lambda * np.sum(self.w ** 2)
        if self.regularization == "l1":
            return self.reg_lambda * np.sum(np.abs(self.w))
        raise RuntimeError("Unknown regularization mode")

    def _regularization_grad(self) -> np.ndarray:
        if self.regularization is None or self.reg_lambda == 0.0:
            return np.zeros_like(self.w)
        if self.regularization == "l2":
            return 2.0 * self.reg_lambda * self.w
        if self.regularization == "l1":
            return self.reg_lambda * np.sign(self.w)
        raise RuntimeError("Unknown regularization mode")

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        p = self._sigmoid(X @ self.w + self.b)
        return self._binary_cross_entropy(y, p) + self._regularization_loss()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LogReg":
        X_train = self._check_X(X_train)
        y_train = self._check_y(y_train)

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train have incompatible shapes")

        use_val = X_val is not None and y_val is not None
        if use_val:
            X_val = self._check_X(X_val)
            y_val = self._check_y(y_val)
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError("X_val and y_val have incompatible shapes")
            if X_val.shape[1] != X_train.shape[1]:
                raise ValueError("X_val must have same number of features as X_train")

        self._fit_scaler(X_train)
        X_train_scaled = self._transform(X_train)
        X_val_scaled = self._transform(X_val) if use_val else None

        n_samples, n_features = X_train_scaled.shape
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.report = LogReg.Report()

        best_monitor_loss = float("inf")
        best_w = self.w.copy()
        best_b = self.b
        epochs_without_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            p_train = self._sigmoid(X_train_scaled @ self.w + self.b)

            dw = (X_train_scaled.T @ (p_train - y_train)) / n_samples
            dw += self._regularization_grad()
            db = np.mean(p_train - y_train)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            train_loss = self._compute_loss(X_train_scaled, y_train)
            self.report.train_loss.append(train_loss)

            if use_val:
                val_loss = self._compute_loss(X_val_scaled, y_val)
                self.report.val_loss.append(val_loss)
                monitor_loss = val_loss
            else:
                monitor_loss = train_loss

            if best_monitor_loss - monitor_loss > self.tol:
                best_monitor_loss = monitor_loss
                best_w = self.w.copy()
                best_b = self.b
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            self.report.epochs_run = epoch

            if epochs_without_improvement >= self.patience:
                self.report.stopped_early = True
                break

        self.w = best_w
        self.b = best_b
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet")
        X = self._check_X(X)
        X_scaled = self._transform(X)
        return self._sigmoid(X_scaled @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = self._check_y(y)
        pred = self.predict(X)
        return float(np.mean(pred == y))

    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        y_true = np.asarray(y_true, dtype=int).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=int).reshape(-1)

        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true and y_pred must have same length")
        if not np.all(np.isin(np.unique(y_true), [0, 1])):
            raise ValueError("y_true must contain only 0 and 1")
        if not np.all(np.isin(np.unique(y_pred), [0, 1])):
            raise ValueError("y_pred must contain only 0 and 1")

        idx = 2 * y_true + y_pred
        counts = np.bincount(idx, minlength=4)
        tn, fp, fn, tp = counts

        accuracy = (tp + tn) / (tn + fp + fn + tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    @staticmethod
    def format_classification_report(metrics: dict) -> str:
        return (
            f"Accuracy : {metrics['accuracy']:.6f}\n"
            f"Precision: {metrics['precision']:.6f}\n"
            f"Recall   : {metrics['recall']:.6f}\n"
            f"F1-score : {metrics['f1']:.6f}\n"
            f"TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}"
        )


def select_best_lambda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lambdas: Iterable[float],
    *,
    regularization: Optional[str] = "l2",
    learning_rate: float = 0.05,
    max_epochs: int = 1000,
    tol: float = 1e-6,
    patience: int = 20,
    metric: str = "accuracy",
    validation_size: float = 0.25,
    random_state: int = 42,
):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)

    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train,
    )

    best_lambda = None
    best_score = -np.inf
    best_model = None
    all_results = []

    for lam in lambdas:
        model = LogReg(
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            tol=tol,
            patience=patience,
            regularization=regularization,
            reg_lambda=float(lam),
        )
        model.fit(X_subtrain, y_subtrain, X_val, y_val)
        pred = model.predict(X_val)
        metrics = model.classification_metrics(y_val, pred)

        if metric not in metrics:
            raise ValueError(f"Unknown metric: {metric}")

        score = metrics[metric]
        all_results.append({"lambda": float(lam), **metrics})

        if score > best_score:
            best_score = score
            best_lambda = float(lam)
            best_model = model

    return best_lambda, best_model, all_results


if __name__ == "__main__":
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    lambdas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    best_lambda, _, tuning_results = select_best_lambda(
        X_train,
        y_train,
        lambdas=lambdas,
        regularization="l2",
        learning_rate=0.05,
        max_epochs=2000,
        tol=1e-7,
        patience=30,
        metric="accuracy",
        validation_size=0.25,
        random_state=42,
    )

    print("Tuning results:")
    for row in tuning_results:
        print(
            f"lambda={row['lambda']:.4g} | "
            f"accuracy={row['accuracy']:.6f} | "
            f"precision={row['precision']:.6f} | "
            f"recall={row['recall']:.6f} | "
            f"f1={row['f1']:.6f}"
        )

    print(f"\nBest lambda: {best_lambda}\n")

    final_model = LogReg(
        learning_rate=0.05,
        max_epochs=3000,
        tol=1e-7,
        patience=50,
        regularization="l2",
        reg_lambda=best_lambda,
    )
    final_model.fit(X_train, y_train)

    my_pred = final_model.predict(X_test)
    my_metrics = final_model.classification_metrics(y_test, my_pred)

    print("--- My model ---")
    print(final_model.format_classification_report(my_metrics))

    sk_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000)
    )
    sk_model.fit(X_train, y_train)
    sk_pred = sk_model.predict(X_test)

    print("\n--- sklearn model ---")
    print(classification_report(y_test, sk_pred))