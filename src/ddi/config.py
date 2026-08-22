from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    """All experiment choices in one serializable object."""

    mode: str = "paper"
    seeds: list[int] = field(default_factory=lambda: [11, 23, 42, 67, 101])
    train_fraction: float = 0.55
    probability_calibration_fraction: float = 0.15
    conformal_calibration_fraction: float = 0.15
    test_fraction: float = 0.15
    primary_alpha: float = 0.10
    alpha_grid: list[float] = field(
        default_factory=lambda: [0.02, 0.05, 0.10, 0.15, 0.20]
    )
    false_positive_cost: float = 1.0
    false_negative_cost: float = 20.0
    deferral_cost: float = 0.5
    xgb_estimators: int = 350
    bootstrap_repetitions: int = 500
    output_dir: str = "results"
    data_dir: str = "data"

    def __post_init__(self) -> None:
        split_total = (
            self.train_fraction
            + self.probability_calibration_fraction
            + self.conformal_calibration_fraction
            + self.test_fraction
        )
        if abs(split_total - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1.0, got {split_total:.6f}")
        if not 0.0 < self.primary_alpha < 1.0:
            raise ValueError("primary_alpha must be between 0 and 1")
        if self.mode not in {"quick", "paper"}:
            raise ValueError("mode must be 'quick' or 'paper'")

    @classmethod
    def for_mode(
        cls, mode: str, output_dir: str = "results", data_dir: str = "data"
    ) -> "ExperimentConfig":
        if mode == "quick":
            return cls(
                mode="quick",
                seeds=[42],
                xgb_estimators=160,
                bootstrap_repetitions=150,
                output_dir=output_dir,
                data_dir=data_dir,
            )
        return cls(mode="paper", output_dir=output_dir, data_dir=data_dir)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

