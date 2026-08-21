import pandas as pd
import numpy as np

from smarttred.features.feature_engine import FeatureEngine


def test_feature_engine_basic():
    # Create simple synthetic OHLCV
    rng = pd.date_range(end=pd.Timestamp.now(), periods=60, freq='min')
    open = np.linspace(1.0, 1.1, 60) + np.random.randn(60) * 0.0005
    high = open + np.abs(np.random.rand(60) * 0.001)
    low = open - np.abs(np.random.rand(60) * 0.001)
    close = open + np.random.randn(60) * 0.0003
    df = pd.DataFrame({'timestamp': rng, 'open': open, 'high': high, 'low': low, 'close': close})

    fe = FeatureEngine()
    df_ind = fe.add_technical_indicators(df)
    assert not df_ind.empty
    # Check that columns exist
    for col in ['rsi_14', 'macd', 'macd_signal', 'macd_hist', 'bb_lower', 'bb_middle', 'bb_upper', 'atr_14']:
        assert col in df_ind.columns

    df_stats = fe.compute_statistical_features(df_ind, window=10)
    assert f'vol_roll_10' in df_stats.columns

    fd = fe.fractional_differentiation(df_stats, column='close', d=0.4, window=20)
    assert isinstance(fd, pd.Series)
    assert not fd.empty
