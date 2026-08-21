import pandas as pd

from smarttred.features.data_split import time_split


def test_time_split_basic():
    df = pd.DataFrame(
        {
            'timestamp': pd.date_range('2024-01-01', periods=20, freq='h'),
            'close': range(20),
            'label_1': [0, 1] * 10,
        }
    )

    split = time_split(df, time_col='timestamp', train_ratio=0.7)
    assert len(split.train) > 0
    assert len(split.validation) > 0
    assert split.train['timestamp'].min() <= split.validation['timestamp'].min()
