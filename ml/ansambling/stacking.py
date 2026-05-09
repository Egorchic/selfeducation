import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold


class Stacking:
    def __init__(self, meta_model = None):
        self.models = []
        self.meta_model = meta_model or LogisticRegression(max_iter=1000)

    def addModel(self, model):
        self.models.append(model)
        return self

    def _makeFeatures(self, X: np.ndarray) -> np.ndarray:
        predictions = []

        for model in self.models:
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)

        return np.column_stack(predictions)

    def fit(self, X_base: np.ndarray, y_base:np.ndarray, n_folds: int = 5):
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        meta_features = np.zeros((X_base.shape[0], len(self.models)))
        true_vals = []

        for model_i, model in enumerate(self.models):
            for train_i, valid_i in kf.split(X_base, y_base):
                X_train_fold = X_base[train_i]
                y_train_fold = y_base[train_i]
                X_valid_fold = X_base[valid_i]
                y_valid_fold = y_base[valid_i]

                cloned_model = clone(model)
                cloned_model.fit(X_train_fold, y_train_fold)
                pred = cloned_model.predict_proba(X_valid_fold)[:, 1]

                meta_features[valid_i, model_i] = pred

        self.meta_model.fit(meta_features, y_base)

        for model in self.models:
            model.fit(X_base, y_base)

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        features = self._makeFeatures(X_test)
        return self.meta_model.predict(features)

    def predictProba(self, X_test: np.ndarray) -> np.ndarray:
        features = self._makeFeatures(X_test)
        return self.meta_model.predict_proba(features)





