from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ModelMonitor:
    """Track model health and highlight concept drift in live trading."""

    drift_threshold: float = 0.1
    retrain_interval_days: int = 7
    artifact_dir: str | Path = "data/model_releases"

    def __post_init__(self) -> None:
        self.artifact_dir = Path(self.artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def check_drift(self, reference: pd.Series, live: pd.Series) -> float:
        """Return the mean absolute difference between the reference and live feature distributions."""
        if reference.empty or live.empty:
            raise ValueError("Reference and live series must not be empty.")
        return float((reference - live).abs().mean())

    def should_retrain(self, last_retrain_date: pd.Timestamp, current_date: pd.Timestamp | None = None) -> bool:
        """Return whether a retraining cycle is due based on elapsed time."""
        if current_date is None:
            current_date = pd.Timestamp.utcnow()
        elapsed_days = (current_date - last_retrain_date).days
        return elapsed_days >= self.retrain_interval_days

    def save_snapshot(self, name: str, payload: dict[str, float]) -> str:
        """Persist a JSON-like summary to the artifact directory."""
        file_path = self.artifact_dir / f"{name}.txt"
        lines = [f"{key}={value}" for key, value in payload.items()]
        file_path.write_text("\n".join(lines), encoding="utf-8")
        return str(file_path)
