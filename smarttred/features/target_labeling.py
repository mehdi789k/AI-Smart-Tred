"""Target labeling utilities for trading ML tasks.

Provides vectorized label generation for classification and regression targets,
including multi-horizon future returns, thresholded directional labels, and
stop-loss/take-profit event labeling over a fixed horizon.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class LabelConfig:
    horizons: List[int] = None
    thresholds: List[float] = None
    sl: float = 0.0  # stop-loss in relative units (e.g., 0.01 = 1%)
    tp: float = 0.0  # take-profit in relative units


class LabelGenerator:
    """Generate supervised targets from OHLCV or feature-enriched DataFrames.

    Conventions:
    - Input DataFrame must be sorted by timestamp ascending and contain a 'close' column.
    - Future returns: (future_close / current_close) - 1
    - Directional label: 1 (long) if future_ret > threshold, -1 (short) if future_ret < -threshold, 0 otherwise.
    - SL/TP labeling: checks within the next `horizon` bars whether price reaches SL or TP,
      returns event column with values: 'tp', 'sl', 'none' and the bar index when event occurred.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def future_returns(series: pd.Series, horizon: int) -> pd.Series:
        """Compute future returns for a given horizon.

        Returns a Series aligned to the input index where value at i is the return over the
        next `horizon` bars: (close[i+horizon] / close[i]) - 1. Values for which future
        data is missing are set to NaN.
        """
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        future = series.shift(-horizon)
        return future.div(series).subtract(1)

    def directional_labels(
        self, close: pd.Series, horizon: int, threshold: float
    ) -> pd.Series:
        """Create directional labels based on future returns and a threshold.

        Label encoding: 1 -> long, -1 -> short, 0 -> neutral.
        """
        fut = self.future_returns(close, horizon)
        labels = pd.Series(0, index=close.index)
        labels[fut > threshold] = 1
        labels[fut < -threshold] = -1
        labels = labels.astype(int)
        # Put NaN for rows where future is not available
        labels[fut.isna()] = np.nan
        return labels

    def multi_horizon_directional(
        self, df: pd.DataFrame, horizons: Iterable[int], threshold: float
    ) -> pd.DataFrame:
        """Generate directional labels for multiple horizons.

        Produces columns: future_ret_{h}, label_{h}
        where {h} is the horizon integer.
        """
        out = df.copy()
        for h in horizons:
            out[f"future_ret_{h}"] = self.future_returns(out["close"], h)
            out[f"label_{h}"] = self.directional_labels(out["close"], h, threshold)
        return out

    def sl_tp_events(self, df: pd.DataFrame, horizon: int, sl: float, tp: float) -> pd.DataFrame:
        """Compute stop-loss / take-profit events over a forward-looking window.

        For each bar i, examine the subsequent `horizon` bars and determine whether
        the high/low reaches the TP or SL levels first. Levels are computed relative
        to the entry price (close at i):
            tp_level = entry * (1 + tp)
            sl_level = entry * (1 - sl)

        Returns a DataFrame with columns:
        - event_{h}: 'tp', 'sl', or 'none'
        - event_idx_{h}: index (integer position) of the bar within the horizon where the event happened (0-based), NaN if none
        """
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if "high" not in df.columns or "low" not in df.columns:
            raise ValueError("DataFrame must contain 'high' and 'low' columns for SL/TP event detection")

        prices_high = df["high"].to_numpy()
        prices_low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(df)

        events = np.array(["none"] * n, dtype=object)
        event_pos = np.full(n, np.nan)

        for i in range(n):
            entry = close[i]
            tp_level = entry * (1.0 + tp)
            sl_level = entry * (1.0 - sl)
            end = min(n, i + horizon + 1)
            happened = "none"
            pos = np.nan
            # scan forward to see which happens first
            for j in range(i + 1, end):
                # check TP first (more conservative for long positions)
                if prices_high[j] >= tp_level:
                    happened = "tp"
                    pos = j - i
                    break
                if prices_low[j] <= sl_level:
                    happened = "sl"
                    pos = j - i
                    break
            events[i] = happened
            event_pos[i] = pos

        out = df.copy()
        out[f"event_{horizon}"] = events
        out[f"event_pos_{horizon}"] = pd.Series(event_pos, index=df.index)
        return out


def generate_labels_from_features(
    features_df: pd.DataFrame,
    horizons: Iterable[int] = (1, 5, 10, 20),
    threshold: float = 0.0005,
    sl: float = 0.01,
    tp: float = 0.02,
) -> pd.DataFrame:
    """Convenience wrapper to generate multi-horizon directional labels and SL/TP events.

    - horizons: list/iterable of integer horizons (bars)
    - threshold: directional threshold for labeling (absolute return)
    - sl/tp: relative stop-loss/take-profit levels
    """
    lg = LabelGenerator()
    df = features_df.copy()

    # Generate future returns and labels
    for h in horizons:
        df[f"future_ret_{h}"] = lg.future_returns(df["close"], h)
        df[f"label_{h}"] = lg.directional_labels(df["close"], h, threshold)

    # Generate SL/TP events for the largest horizon as a representative risk event
    max_h = max(horizons)
    try:
        df = lg.sl_tp_events(df, max_h, sl=sl, tp=tp)
    except ValueError:
        # If high/low missing, skip SL/TP
        pass

    # Final cleanup: drop rows where future_ret for largest horizon is NaN
    df = df.dropna(subset=[f"future_ret_{max_h}"]).reset_index(drop=True)
    return df
