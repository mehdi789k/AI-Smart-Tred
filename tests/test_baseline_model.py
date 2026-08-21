import pandas as pd
import numpy as np

from smarttred.models.baseline_model import BaselineModelTrainer, prepare_model_matrix


def test_prepare_model_matrix_and_training():
    rng = pd.date_range(end=pd.Timestamp.now(), periods=50, freq='min')
    df = pd.DataFrame({
        'timestamp': rng,
        'close': np.linspace(100, 120, 50),
        'feature_a': np.linspace(0, 1, 50),
        'feature_b': np.linspace(1, 0, 50),
        'label_5': [1, -1, 1, -1] * 12 + [1, -1],
    })

    X, y = prepare_model_matrix(df, target_col='label_5')
    # prepare_model_matrix keeps all numeric columns except timestamp and target
    expected_features = ['close', 'feature_a', 'feature_b']
    assert set(X.columns) == set(expected_features)
    assert len(X) == len(y)

    trainer = BaselineModelTrainer(model_name='logistic')
    model, metrics, _, _, _, _ = trainer.train(X, y, test_size=0.2)
    assert model is not None
    assert 'accuracy' in metrics
