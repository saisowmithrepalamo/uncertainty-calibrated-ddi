from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Conservative split-conformal quantile with finite-sample correction."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    level = min(1.0, np.ceil((len(values) + 1) * (1.0 - alpha)) / len(values))
    try:
        return float(np.quantile(values, level, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility
        return float(np.quantile(values, level, interpolation="higher"))


@dataclass(frozen=True)
class MondrianThresholds:
    alpha: float
    class_zero: float
    class_one: float


class MondrianConformalClassifier:
    """Class-conditional split conformal sets for imbalanced binary outcomes."""

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self.thresholds: MondrianThresholds | None = None

    def fit(
        self, failure_probabilities: np.ndarray, y_true: np.ndarray
    ) -> "MondrianConformalClassifier":
        p = np.asarray(failure_probabilities, dtype=float)
        y = np.asarray(y_true, dtype=int)
        if len(p) != len(y):
            raise ValueError("probabilities and labels must have equal length")
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Conformal calibration data must contain both classes")
        probability_matrix = np.column_stack([1.0 - p, p])
        true_class_scores = 1.0 - probability_matrix[np.arange(len(y)), y]
        q0 = finite_sample_quantile(true_class_scores[y == 0], self.alpha)
        q1 = finite_sample_quantile(true_class_scores[y == 1], self.alpha)
        self.thresholds = MondrianThresholds(self.alpha, q0, q1)
        return self

    def prediction_sets(self, failure_probabilities: np.ndarray) -> np.ndarray:
        if self.thresholds is None:
            raise RuntimeError("Conformal classifier must be fitted first")
        p = np.asarray(failure_probabilities, dtype=float)
        include_zero = p <= self.thresholds.class_zero
        include_one = (1.0 - p) <= self.thresholds.class_one
        return np.column_stack([include_zero, include_one])

    def predict_or_defer(self, failure_probabilities: np.ndarray) -> np.ndarray:
        """Predict only when calibrated argmax and conformal singleton agree."""

        p = np.asarray(failure_probabilities, dtype=float)
        sets = self.prediction_sets(p)
        base_prediction = (p >= 0.5).astype(int)
        decisions = np.full(len(sets), -1, dtype=int)
        safe_zero = sets[:, 0] & ~sets[:, 1] & (base_prediction == 0)
        safe_one = sets[:, 1] & ~sets[:, 0] & (base_prediction == 1)
        decisions[safe_zero] = 0
        decisions[safe_one] = 1
        return decisions
