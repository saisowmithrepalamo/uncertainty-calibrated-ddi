import numpy as np

from ddi.conformal import MondrianConformalClassifier, finite_sample_quantile


def test_finite_sample_quantile_is_observed_score() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    quantile = finite_sample_quantile(scores, alpha=0.25)
    assert quantile in scores
    assert quantile == 0.4


def test_mondrian_classifier_predicts_or_defers() -> None:
    calibration_probabilities = np.array([0.02, 0.05, 0.10, 0.20, 0.70, 0.80, 0.90, 0.95])
    calibration_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    conformal = MondrianConformalClassifier(alpha=0.25).fit(
        calibration_probabilities, calibration_labels
    )
    decisions = conformal.predict_or_defer(np.array([0.01, 0.50, 0.99]))
    assert decisions[0] == 0
    assert decisions[1] == -1
    assert decisions[-1] == 1


def test_conformal_singleton_that_disagrees_with_model_is_deferred() -> None:
    probabilities = np.array([0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    conformal = MondrianConformalClassifier(alpha=0.25).fit(probabilities, labels)
    assert conformal.predict_or_defer(np.array([0.20]))[0] == -1
