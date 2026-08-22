from __future__ import annotations

import numpy as np
import pandas as pd

from ddi.data import ID_COLUMNS, LEAKAGE_COLUMNS, TARGET


RAW_SENSOR_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

KNOWLEDGE_FEATURES = [
    "Temperature gap [K]",
    "Mechanical power [W]",
    "Wear-load interaction",
    "Speed deviation from nominal [rpm]",
]


def assert_no_leakage(columns: list[str]) -> None:
    forbidden = set(LEAKAGE_COLUMNS + ID_COLUMNS + [TARGET])
    present = forbidden.intersection(columns)
    if present:
        raise ValueError(f"Leakage/identifier columns found in model features: {present}")


def build_feature_matrix(
    frame: pd.DataFrame, include_knowledge: bool
) -> pd.DataFrame:
    """Return numeric features without IDs, target, or failure-mode labels."""

    missing = set(RAW_SENSOR_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing raw sensor columns: {sorted(missing)}")

    features = frame[RAW_SENSOR_COLUMNS].copy()
    if include_knowledge:
        features["Temperature gap [K]"] = (
            features["Process temperature [K]"]
            - features["Air temperature [K]"]
        )
        features["Mechanical power [W]"] = (
            2.0
            * np.pi
            * features["Rotational speed [rpm]"]
            * features["Torque [Nm]"]
            / 60.0
        )
        features["Wear-load interaction"] = (
            features["Tool wear [min]"] * features["Torque [Nm]"]
        )
        features["Speed deviation from nominal [rpm]"] = (
            features["Rotational speed [rpm]"] - 1500.0
        ).abs()

    features = pd.get_dummies(features, columns=["Type"], dtype=float)
    features = features.astype(float)
    assert_no_leakage(features.columns.tolist())
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Feature matrix contains non-finite values")
    return features

