from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Independent sigmoid calibration fitted to held-out probabilities."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2_000)
        self._fitted = False

    def _logits(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), self.epsilon, 1 - self.epsilon)
        return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        y = np.asarray(y_true, dtype=int)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Calibration data must contain both classes")
        self.model.fit(self._logits(probabilities), y)
        self._fitted = True
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Calibrator must be fitted before predict")
        calibrated = self.model.predict_proba(self._logits(probabilities))[:, 1]
        return np.clip(calibrated, self.epsilon, 1 - self.epsilon)

