import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class Blending:
    def __init__(self, meta_model = None):
        self.model_collection = []
        self.meta_model = meta_model or LogisticRegression(max_iter=1000)

    def addModel(self, model):
        self.model_collection.append(model)

    def _makeFeatures(self, X) -> np.ndarray:
        predictions = []

        for model in self.model_collection:
            prediction = model.predict_proba(X)
            predictions.append(prediction)

        return np.hstack(predictions)

    def fit(self, X: np.ndarray, y: np.ndarray, blend_size: float = 0.3):
        X_base, X_blend, y_base, y_blend = train_test_split(X, y, test_size=blend_size)

        for model in self.model_collection:
            model.fit(X_base, y_base)

        X_meta = self._makeFeatures(X_blend)

        self.meta_model.fit(X_meta, y_blend)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_meta = self._makeFeatures(X_test)

        return self.meta_model.predict(X_meta)

    def predictProba(self, X_test: np.ndarray) -> np.ndarray:
        X_meta = self._makeFeatures(X_test)

        return self.meta_model.predict_proba(X_meta)





