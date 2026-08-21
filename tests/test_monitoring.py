import pandas as pd

from smarttred.monitoring import ModelMonitor


def test_model_monitor_drift_and_retrain() -> None:
    monitor = ModelMonitor(drift_threshold=0.1, retrain_interval_days=7)

    reference = pd.Series([1.0, 2.0, 3.0])
    live = pd.Series([1.1, 2.2, 3.4])
    drift = monitor.check_drift(reference, live)
    assert drift > 0

    last_retrain = pd.Timestamp("2024-01-01")
    current = pd.Timestamp("2024-01-10")
    assert monitor.should_retrain(last_retrain, current) is True

    snapshot_path = monitor.save_snapshot("healthcheck", {"drift": drift})
    assert snapshot_path.endswith("healthcheck.txt")
