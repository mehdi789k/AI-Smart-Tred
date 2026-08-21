import pandas as pd

from smarttred.backtest import VectorBTBacktester


def test_vectorbt_backtester_runs() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0, 100.5, 101.0, 100.7, 101.5, 101.2],
            "signal": [0, 1, 1, -1, 0, 1],
        }
    )

    tester = VectorBTBacktester(initial_balance=10000.0)
    result = tester.run(df, signal_col="signal")

    assert not result.empty
    assert "balance" in result.columns
    assert "equity_change" in result.columns
