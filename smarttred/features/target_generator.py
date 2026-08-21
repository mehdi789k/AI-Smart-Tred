from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute ATR (simple moving average of True Range) as a numpy array.

    Returns an array of same length where the first values may be NaN until enough
    periods are available.
    """
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    # simple moving average of TR
    atr = np.full_like(tr, np.nan, dtype=float)
    # compute SMA efficiently
    cumsum = np.nancumsum(tr)
    # first period-1 are NaN
    if period <= 0:
        raise ValueError("ATR period must be > 0")
    if len(tr) >= period:
        atr[period - 1] = cumsum[period - 1] / period
        for i in range(period, len(tr)):
            atr[i] = (cumsum[i] - cumsum[i - period]) / period
    return atr


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    profit_taker: float = 0.01,
    stop_loss: float = 0.005,
    time_limit: int = 30,
    use_atr: bool = False,
    atr_period: int = 14,
    label_col: str = "target",
    barrier_col: str = "barrier_type",
    rt_col: str = "rt",
) -> pd.DataFrame:
    """Vectorized triple-barrier label generator.

    This implementation builds a (N x time_limit) shifted view of highs/lows and
    determines the first offset where the profit-taking (TP) or stop-loss (SL)
    barrier is breached.

    Parameters
    - df: DataFrame containing at least timestamp, high, low, close columns.
    - profit_taker / stop_loss: if use_atr==False these are fractional returns
      (e.g., 0.01 means +1%). If use_atr==True these are multiples of ATR.
    - time_limit: number of bars to look ahead for each entry.
    - use_atr: when True, compute ATR and set barriers relative to ATR magnitude.

    Returns a copy of df with added columns: label_col (1, -1, 0), barrier_col
    ('tp','sl','time'), and rt_col (realized return at barrier touch; NaN if
    could not be computed).
    """
    if not set(["timestamp", high_col, low_col, close_col]).issubset(df.columns):
        raise ValueError(f"DataFrame must contain timestamp, {high_col}, {low_col}, {close_col} columns")

    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    n = len(out)
    if n == 0:
        return out

    high = out[high_col].to_numpy(dtype=float)
    low = out[low_col].to_numpy(dtype=float)
    close = out[close_col].to_numpy(dtype=float)

    if use_atr:
        atr = _compute_atr(high, low, close, period=atr_period)
    else:
        atr = np.full(n, np.nan)

    entry_price = close  # entry at close by convention

    # Build shifted matrices for offsets 1..time_limit. Each column k contains
    # values of the bar at t+k for all t; padded with np.nan for positions beyond end.
    # Memory cost: O(n * time_limit) floats. For very large n consider chunking.
    shifts = []
    shifts_low = []
    shifts_high = []
    shifts_close = []
    for k in range(1, time_limit + 1):
        if k >= n:
            # all rows will be NaN after shifting beyond length
            pad = np.full(n, np.nan)
            shifts_high.append(pad)
            shifts_low.append(pad)
            shifts_close.append(pad)
            continue
        # shifted arrays: take values from k..n-1 and pad last k positions with NaN
        shifted_high = np.concatenate((high[k:], np.full(k, np.nan)))
        shifted_low = np.concatenate((low[k:], np.full(k, np.nan)))
        shifted_close = np.concatenate((close[k:], np.full(k, np.nan)))
        shifts_high.append(shifted_high)
        shifts_low.append(shifted_low)
        shifts_close.append(shifted_close)

    # Stack into (n, time_limit) arrays
    high_matrix = np.column_stack(shifts_high) if shifts_high else np.empty((n, 0))
    low_matrix = np.column_stack(shifts_low) if shifts_low else np.empty((n, 0))
    close_matrix = np.column_stack(shifts_close) if shifts_close else np.empty((n, 0))

    # Compute barrier thresholds per row
    if use_atr:
        # ATR may be NaN at the beginning; fall back to fractional returns when NaN
        tp_thresholds = np.where(np.isnan(atr), entry_price * (1.0 + profit_taker), entry_price + atr * profit_taker)
        sl_thresholds = np.where(np.isnan(atr), entry_price * (1.0 - stop_loss), entry_price - atr * stop_loss)
    else:
        tp_thresholds = entry_price * (1.0 + profit_taker)
        sl_thresholds = entry_price * (1.0 - stop_loss)

    # Broadcast thresholds to matrices and evaluate crossings
    if high_matrix.size == 0 or low_matrix.size == 0:
        # No forward-looking bars available (time_limit==0 or very small df)
        out[label_col] = 0
        out[barrier_col] = "time"
        out[rt_col] = np.nan
        return out

    # Comparison yields boolean matrices; True where barrier condition met at that offset
    tp_hits = high_matrix >= tp_thresholds[:, None]
    sl_hits = low_matrix <= sl_thresholds[:, None]

    # For each row detect if any hit occurred and the first offset (1-based)
    tp_any = tp_hits.any(axis=1)
    sl_any = sl_hits.any(axis=1)

    # np.argmax returns first occurrence index; but if no True it returns 0, so guard with any
    tp_first_offset = np.full(n, np.inf)
    sl_first_offset = np.full(n, np.inf)

    if tp_hits.shape[1] > 0:
        tp_first_offset[tp_any] = np.argmax(tp_hits[tp_any, :], axis=1) + 1
        sl_first_offset[sl_any] = np.argmax(sl_hits[sl_any, :], axis=1) + 1

    # Determine winners
    # Initialize outputs
    targets = np.zeros(n, dtype=int)
    barrier_type = np.array(["time"] * n, dtype=object)
    rt = np.full(n, np.nan, dtype=float)

    # Case where TP occurs strictly before SL
    tp_before_sl = (tp_first_offset < sl_first_offset)
    tp_idx = np.where(tp_before_sl)[0]
    if tp_idx.size > 0:
        offsets = tp_first_offset[tp_idx].astype(int)
        hit_prices = high_matrix[tp_idx, offsets - 1]
        targets[tp_idx] = 1
        barrier_type[tp_idx] = "tp"
        rt[tp_idx] = (hit_prices / entry_price[tp_idx]) - 1.0

    # Case where SL occurs strictly before TP
    sl_before_tp = (sl_first_offset < tp_first_offset)
    sl_idx = np.where(sl_before_tp)[0]
    if sl_idx.size > 0:
        offsets = sl_first_offset[sl_idx].astype(int)
        hit_prices = low_matrix[sl_idx, offsets - 1]
        targets[sl_idx] = -1
        barrier_type[sl_idx] = "sl"
        rt[sl_idx] = (hit_prices / entry_price[sl_idx]) - 1.0

    # Case where neither barrier touched within window -> time barrier
    neither = (~tp_any) & (~sl_any)
    if neither.any():
        idxs = np.where(neither)[0]
        # price at end of window: use close at t+time_limit if available else last close
        end_offsets = np.full(idxs.shape, time_limit, dtype=int)
        end_indices = idxs + end_offsets
        # clamp to last index
        end_indices = np.minimum(end_indices, n - 1)
        end_prices = close[end_indices]
        rt[idxs] = (end_prices / entry_price[idxs]) - 1.0
        targets[idxs] = 0
        barrier_type[idxs] = "time"

    # Case where both were hit within window at same offset (tie)
    tie = (tp_any & sl_any) & (tp_first_offset == sl_first_offset)
    if tie.any():
        tie_idx = np.where(tie)[0]
        offsets = tp_first_offset[tie_idx].astype(int)
        # For tie-break, fall back to close at the tie-bar: if close >= tp -> tp, elif close <= sl -> sl, else time
        # use close at the tie-bar for tie-breaking
        tie_close = close_matrix[tie_idx, offsets - 1]
        tp_thresh = tp_thresholds[tie_idx]
        sl_thresh = sl_thresholds[tie_idx]

        tp_mask = tie_close >= tp_thresh
        sl_mask = tie_close <= sl_thresh

        # assign where close indicates TP
        tp_assign = tie_idx[tp_mask]
        if tp_assign.size > 0:
            off = tp_first_offset[tp_assign].astype(int)
            hit_prices = high_matrix[tp_assign, off - 1]
            targets[tp_assign] = 1
            barrier_type[tp_assign] = "tp"
            rt[tp_assign] = (hit_prices / entry_price[tp_assign]) - 1.0

        # assign where close indicates SL
        sl_assign = tie_idx[sl_mask]
        if sl_assign.size > 0:
            off = sl_first_offset[sl_assign].astype(int)
            hit_prices = low_matrix[sl_assign, off - 1]
            targets[sl_assign] = -1
            barrier_type[sl_assign] = "sl"
            rt[sl_assign] = (hit_prices / entry_price[sl_assign]) - 1.0

        # remaining ambiguous ties -> treat as time
        remaining = np.setdiff1d(tie_idx, np.concatenate([tp_assign, sl_assign]))
        if remaining.size > 0:
            end_offsets = np.full(remaining.shape, time_limit, dtype=int)
            end_indices = remaining + end_offsets
            end_indices = np.minimum(end_indices, n - 1)
            end_prices = close[end_indices]
            rt[remaining] = (end_prices / entry_price[remaining]) - 1.0
            targets[remaining] = 0
            barrier_type[remaining] = "time"

    # Insert into DataFrame
    out[label_col] = targets
    out[barrier_col] = barrier_type
    out[rt_col] = rt
    return out
