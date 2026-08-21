from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy import stats


class FeatureEngine:
    """Feature engineering utilities for time-series market data.

    Methods:
    - add_technical_indicators: adds RSI(14), MACD, BBANDS, ATR using pandas_ta
    - compute_statistical_features: rolling volatility, skewness, kurtosis
    - fractional_differentiation: Marcos Lopez de Prado fractional differentiation
    - transform: full pipeline transformation with configurable frac_diff_order
    """

    def __init__(self, frac_diff_order: float = 0.5) -> None:
        self.frac_diff_order = frac_diff_order

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators using pandas_ta.

        Adds the following columns:
        - rsi_14
        - macd, macd_signal, macd_hist
        - bb_lower, bb_middle, bb_upper
        - atr_14

        This function returns a new DataFrame and drops rows with NaNs introduced
        by indicator lookbacks so that the downstream dataset contains no NaNs.
        """
        if df is None or df.empty:
            return df

        out = df.copy()

        # Ensure numeric
        for col in ["open", "high", "low", "close", "tick_volume", "real_volume", "volume"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        # RSI(14)
        try:
            out["rsi_14"] = ta.rsi(out["close"], length=14)
        except Exception:
            out["rsi_14"] = out["close"].pct_change().rolling(14).apply(lambda x: (x.mean() * 100), raw=False)

        # MACD (12,26,9)
        macd_df = ta.macd(out["close"], fast=12, slow=26, signal=9)
        if isinstance(macd_df, pd.DataFrame):
            # standard pandas_ta returns DataFrame with MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            # normalize to short names
            for col in macd_df.columns:
                lname = col.lower()
                if "macd" in lname and "s_" not in lname and "h" not in lname:
                    out["macd"] = macd_df[col]
                elif "macds" in lname or ("signal" in lname and "macd" not in lname):
                    out["macd_signal"] = macd_df[col]
                elif "macdh" in lname or "hist" in lname or "histo" in lname:
                    out["macd_hist"] = macd_df[col]
        elif isinstance(macd_df, pd.Series):
            out["macd"] = macd_df

        # Bollinger Bands (20,2)
        bb = ta.bbands(out["close"], length=20, std=2)
        if isinstance(bb, pd.DataFrame):
            for col in bb.columns:
                name = col.lower()
                # common names include 'BBL_20_2.0','BBM_20_2.0','BBU_20_2.0'
                if "bbl" in name:
                    out["bb_lower"] = bb[col]
                elif "bbm" in name:
                    out["bb_middle"] = bb[col]
                elif "bbu" in name:
                    out["bb_upper"] = bb[col]
                else:
                    out[name] = bb[col]

        # ATR(14)
        atr = ta.atr(high=out.get("high"), low=out.get("low"), close=out.get("close"), length=14)
        out["atr_14"] = atr

        # Drop any rows with NaNs introduced by the indicators' lookback windows
        out = out.dropna().reset_index(drop=True)
        return out

    def compute_statistical_features(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add rolling statistical features: volatility, skewness, kurtosis on returns.

        - vol_roll_{window}: rolling std of simple returns
        - skew_roll_{window}: rolling skewness
        - kurt_roll_{window}: rolling kurtosis
        """
        if df is None or df.empty:
            return df

        out = df.copy()
        out["ret"] = out["close"].pct_change()

        out[f"vol_roll_{window}"] = out["ret"].rolling(window).std()
        out[f"skew_roll_{window}"] = out["ret"].rolling(window).skew()
        out[f"kurt_roll_{window}"] = out["ret"].rolling(window).kurt()

        out = out.dropna().reset_index(drop=True)
        return out

    def fractional_differentiation(self, df: pd.DataFrame, column: str = "close", d: float = 0.4, window: int = 100) -> pd.Series:
        """Vectorized implementation of fixed-width fractional differentiation.

        Uses the finite-sum approximation with weights computed iteratively
        (Marcos Lopez de Prado style). The routine returns a pandas Series aligned
        to the input index where the initial (window-1) rows are trimmed to avoid
        partial-weight bias.
        """
        if df is None or df.empty:
            return pd.Series(dtype=float)

        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")

        series = df[column].astype(float).ffill().bfill()
        n = len(series)
        size = min(window, n)

        # compute weights and reverse for convolution alignment
        w = [1.0]
        for k in range(1, size):
            w.append(-w[-1] * (d - k + 1) / k)
        weights = np.array(w[::-1])

        conv = np.convolve(series.values, weights, mode="full")
        start = size - 1
        frac_values = conv[start : start + n]
        result = pd.Series(frac_values, index=series.index)

        # Trim initial rows that used fewer terms than `window`
        if size - 1 > 0:
            result = result.iloc[(size - 1) :].copy()
            result.index = series.index[(size - 1) :]

        # Ensure no NaNs remain
        result = result.ffill().bfill()
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full feature engineering pipeline.
        
        Applies technical indicators, statistical features, and fractional differentiation.
        
        Returns a DataFrame with columns:
        - Original OHLCV columns
        - Technical indicators (rsi_14, macd, macd_signal, macd_hist, bb_lower, bb_middle, bb_upper, atr_14)
        - Statistical features (ret, vol_roll_20, skew_roll_20, kurt_roll_20)
        - Fractional differentiated columns (close_fd_{d}, returns_fd_{d})
        """
        if df is None or df.empty:
            return df
        
        # Step 1: technical indicators
        df_ind = self.add_technical_indicators(df)
        
        # Step 2: rolling statistical features
        df_stats = self.compute_statistical_features(df_ind, window=20)
        
        # Step 3: fractional differentiation on close
        d = self.frac_diff_order
        fd_close = self.fractional_differentiation(df_stats, column="close", d=d, window=100)
        
        # Align dataframe to fd index
        df_final = df_stats.loc[fd_close.index].copy()
        fd_col_name = f"close_fd_{d}".replace(".", "_")
        df_final[fd_col_name] = fd_close.values
        
        # Add fractional differentiated returns
        ret_fd = fd_close.pct_change()
        ret_fd_col_name = f"returns_fd_{d}".replace(".", "_")
        df_final[ret_fd_col_name] = ret_fd.values
        
        # Add simple returns column for compatibility
        if "returns" not in df_final.columns and "ret" in df_final.columns:
            df_final["returns"] = df_final["ret"]
        
        # Add SMA_10 for compatibility
        if "sma_10" not in df_final.columns:
            df_final["sma_10"] = df_final["close"].rolling(10).mean()
        
        df_final = df_final.dropna().reset_index(drop=True)
        return df_final
