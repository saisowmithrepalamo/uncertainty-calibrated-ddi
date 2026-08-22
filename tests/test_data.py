import numpy as np

from ddi.data import make_four_way_split


def test_four_way_split_is_disjoint_complete_and_stratified() -> None:
    y = np.array([0] * 900 + [1] * 100)
    splits = make_four_way_split(y, 42, 0.55, 0.15, 0.15, 0.15)
    all_indices = np.concatenate(list(splits.as_dict().values()))
    assert len(all_indices) == 1000
    assert len(np.unique(all_indices)) == 1000
    assert set(all_indices) == set(range(1000))
    for indices in splits.as_dict().values():
        assert abs(float(y[indices].mean()) - 0.10) < 0.02

