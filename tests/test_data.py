import numpy as np
import pandas as pd

from ddi.data import dataset_quality_audit, make_four_way_split


def test_four_way_split_is_disjoint_complete_and_stratified() -> None:
    y = np.array([0] * 900 + [1] * 100)
    splits = make_four_way_split(y, 42, 0.55, 0.15, 0.15, 0.15)
    all_indices = np.concatenate(list(splits.as_dict().values()))
    assert len(all_indices) == 1000
    assert len(np.unique(all_indices)) == 1000
    assert set(all_indices) == set(range(1000))
    for indices in splits.as_dict().values():
        assert abs(float(y[indices].mean()) - 0.10) < 0.02


def test_data_quality_audit_counts_documented_rule_mismatches() -> None:
    frame = pd.DataFrame(
        {
            "UDI": range(10_000),
            "Product ID": [f"L{i}" for i in range(10_000)],
            "Type": ["L"] * 10_000,
            "Air temperature [K]": [300.0] * 10_000,
            "Process temperature [K]": [310.0] * 10_000,
            "Rotational speed [rpm]": [1500] * 10_000,
            "Torque [Nm]": [40.0] * 10_000,
            "Tool wear [min]": [100] * 10_000,
            "Machine failure": [1, 0] + [0] * 9_998,
            "TWF": [0, 1] + [0] * 9_998,
            "HDF": [0] * 10_000,
            "PWF": [0] * 10_000,
            "OSF": [0] * 10_000,
            "RNF": [0] * 10_000,
        }
    )
    audit = dataset_quality_audit(frame)
    assert audit["target_positive_without_failure_mode_rows"] == 1
    assert audit["failure_mode_positive_with_target_zero_rows"] == 1
    assert audit["target_failure_mode_or_rule_mismatches"] == 2
    assert audit["target_matches_failure_mode_or_rule"] is False
