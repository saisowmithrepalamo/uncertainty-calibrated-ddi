import numpy as np
import pandas as pd

from ddi.features import build_feature_matrix


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UDI": [1, 2],
            "Product ID": ["L1", "M2"],
            "Type": ["L", "M"],
            "Air temperature [K]": [300.0, 301.0],
            "Process temperature [K]": [310.0, 312.0],
            "Rotational speed [rpm]": [1500, 1800],
            "Torque [Nm]": [40.0, 30.0],
            "Tool wear [min]": [100, 50],
            "Machine failure": [0, 1],
            "TWF": [0, 0],
            "HDF": [0, 0],
            "PWF": [0, 1],
            "OSF": [0, 0],
            "RNF": [0, 0],
        }
    )


def test_features_remove_identifiers_targets_and_failure_modes() -> None:
    features = build_feature_matrix(sample_frame(), include_knowledge=True)
    forbidden = {"UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"}
    assert forbidden.isdisjoint(features.columns)
    assert np.isfinite(features.to_numpy()).all()


def test_knowledge_features_are_computed_from_sensors() -> None:
    features = build_feature_matrix(sample_frame(), include_knowledge=True)
    assert features.loc[0, "Temperature gap [K]"] == 10.0
    assert features.loc[0, "Wear-load interaction"] == 4000.0
    expected_power = 2 * np.pi * 1500 * 40 / 60
    assert np.isclose(features.loc[0, "Mechanical power [W]"], expected_power)

