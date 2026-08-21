import pandas as pd
import numpy as np

from smarttred.features.target_labeling import generate_labels_from_features


def test_generate_labels_basic():
    rng = pd.date_range(end=pd.Timestamp.now(), periods=50, freq='T')
    open = np.linspace(1.0, 1.05, 50) + np.random.randn(50)*0.0003
    high = open + np.abs(np.random.rand(50)*0.001)
    low = open - np.abs(np.random.rand(50)*0.001)
    close = open + np.random.randn(50)*0.0002
    df = pd.DataFrame({'timestamp': rng, 'open': open, 'high': high, 'low': low, 'close': close})

    labeled = generate_labels_from_features(df, horizons=(1,5), threshold=0.0001, sl=0.005, tp=0.01)
    # Should contain label_1, label_5, future_ret_5 (largest horizon)
    assert 'label_1' in labeled.columns
    assert 'label_5' in labeled.columns
    assert 'future_ret_5' in labeled.columns
    assert 'event_5' in labeled.columns
