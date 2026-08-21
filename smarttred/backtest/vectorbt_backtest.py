from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    import vectorbt as vbt
except ImportError:  # pragma: no cover - optional dependency for live/advanced usage
    vbt = None


@dataclass
class VectorBTBacktester:
    """Vectorized backtesting engine for signal-based trading strategies.

    This implementation is designed to be realistic for a research pipeline:
    - it accepts OHLCV data with a discrete signal column (-1, 0, 1)
    - it uses a vectorized return formulation to avoid Python loops
    - it includes a simple slippage and commission model
    - it optionally wraps VectorBT when available, while keeping a pandas fallback
      so tests and local runs still work without the heavy dependency installed.
    """

    initial_balance: float = 10000.0
    commission_per_trade: float = 0.0005
    slippage_bps: float = 5.0
    risk_per_trade: float = 0.01

    def _prepare(self, df: pd.DataFrame, signal_col: str) -> pd.DataFrame:
        """Validate and normalize the input DataFrame before backtesting."""
        if signal_col not in df.columns:
            raise ValueError(f"Signal column '{signal_col}' not found in dataframe.")
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column for backtesting.")

        out = df.copy()
        out["signal"] = pd.to_numeric(out[signal_col], errors="coerce").fillna(0).clip(-1, 1)
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
        out["returns"] = out["close"].pct_change().fillna(0.0)
        return out

    def _vectorbt_optional(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use VectorBT when available; otherwise return a pandas-based result."""
        if vbt is None:
            return self._pandas_backtest(df)

        # Using vectorbt for signal-based strategy evaluation when the optional
        # dependency is installed. We still keep the fallback path so the project
        # remains testable in lightweight environments.
        signals = df["signal"].astype(float)
        price = df["close"].astype(float)
        entries = signals == 1
        exits = signals == -1

        # In vectorbt v1.0+, commission and slippage are passed directly to from_signals
        portfolio = vbt.Portfolio.from_signals(
            close=price,
            entries=entries,
            exits=exits,
            init_cash=self.initial_balance,
            fees=self.commission_per_trade,
            slippage=self.slippage_bps / 10_000
        )

        result = portfolio.value().to_frame(name="balance")
        result["equity_change"] = result["balance"].pct_change().fillna(0.0)
        result["position"] = signals.fillna(0)
        result["signal"] = signals
        return result

    def _pandas_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute a vectorized backtest using pandas arithmetic only."""
        signals = df["signal"].fillna(0).astype(float).clip(-1, 1)
        positions = signals.shift(1).fillna(0).clip(-1, 1)
        daily_returns = df["returns"].fillna(0.0)

        # Position drift approximates strategy exposure over the trading period.
        strategy_returns = positions * daily_returns

        # Slippage and commission are applied to position changes, which is a
        # practical approximation for a realistic total-cost model.
        position_changes = positions.diff().abs().fillna(0.0)
        notional = position_changes * df["close"].fillna(0.0)
        costs = notional * (self.commission_per_trade + (self.slippage_bps / 10_000))

        net_returns = strategy_returns - (costs / self.initial_balance)
        equity = (1.0 + net_returns).cumprod() * self.initial_balance

        result = pd.DataFrame(
            {
                "signal": signals,
                "position": positions,
                "returns": daily_returns,
                "net_returns": net_returns,
                "balance": equity,
            }
        )
        result["equity_change"] = result["balance"].pct_change().fillna(0.0)
        return result

    def run(self, df: pd.DataFrame, signal_col: str = "signal") -> pd.DataFrame:
        """Run a vectorized signal-based backtest and return equity progression."""
        prepared = self._prepare(df, signal_col)
        return self._vectorbt_optional(prepared)

    def save_report(self, result: pd.DataFrame, path: str | Path) -> str:
        """Persist backtest output to a CSV file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(target, index=False)
        return str(target)
