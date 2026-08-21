from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

import joblib
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from smarttred.config.settings import Settings
from smarttred.features.feature_engine import FeatureEngine
from smarttred.trading.live_executor import LiveExecutor
from smarttred.trading.risk_manager import RiskManager

LOG_DIR = os.path.join(os.getcwd(), "data", "logs")
LOG_FILE = os.path.join(LOG_DIR, "live_trader.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("smarttred.live_execution")


def _symbol_base_quote(symbol: str) -> tuple[str, str]:
    s = symbol.upper().replace("_", "")
    if len(s) >= 6:
        base = s[:-3]
        quote = s[-3:]
        if base.isalpha() and quote.isalpha():
            return base, quote
        if len(s) >= 6 and s[:3].isalpha() and s[3:].isalpha():
            return s[:3], s[3:]
    return "", ""


def _resolve_pip_size(symbol: str) -> float:
    s = symbol.upper()
    if any(token in s for token in ("JPY", "XAU", "XAG", "XPT", "XPD", "BTC", "ETH")):
        return 0.01
    return 0.0001


def _fetch_symbol_mid(symbol: str) -> float | None:
    if mt5 is None:
        return None
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is not None:
            bid = getattr(tick, "bid", None)
            ask = getattr(tick, "ask", None)
            if bid is not None and ask is not None:
                return float((bid + ask) / 2.0)
            if bid is not None:
                return float(bid)
            if ask is not None:
                return float(ask)
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        bid = getattr(info, "bid", None)
        ask = getattr(info, "ask", None)
        last = getattr(info, "last", None)
        if bid is not None and ask is not None:
            return float((bid + ask) / 2.0)
        if bid is not None:
            return float(bid)
        if ask is not None:
            return float(ask)
        if last is not None:
            return float(last)
    except Exception:
        return None
    return None


def estimate_pip_value(symbol: str, account_currency: str | None = None) -> float:
    """Estimate pip value per lot using contract size and FX conversion.

    This is a conservative best-effort calculation for MT5 symbols. It is precise
    enough for most FX/commodity symbols when the account currency differs from the
    pair's quote/base currency.
    """
    if mt5 is None:
        return 1.0

    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning("Could not fetch symbol_info for %s; returning fallback pip value = 1.0", symbol)
            return 1.0

        contract_size = float(getattr(info, "trade_contract_size", 100000.0) or 100000.0)
        pip_size = _resolve_pip_size(symbol)
        base, quote = _symbol_base_quote(symbol)
        account_currency = (account_currency or (mt5.account_info().currency if mt5.account_info() else "USD")).upper()

        if not base or not quote:
            return contract_size * pip_size

        mid = _fetch_symbol_mid(symbol)
        mid = mid if mid is not None else 1.0

        if account_currency == quote:
            return contract_size * pip_size
        if account_currency == base:
            return contract_size * pip_size / max(mid, 1e-8)

        # Cross conversion attempts using direct FX pairs, if available
        cross_pairs = [
            f"{account_currency}{quote}",
            f"{quote}{account_currency}",
            f"{account_currency}{base}",
            f"{base}{account_currency}",
        ]
        for pair in cross_pairs:
            if pair == symbol or len(pair) < 6:
                continue
            pair_mid = _fetch_symbol_mid(pair)
            if pair_mid is None:
                continue
            if pair.startswith(account_currency) and pair.endswith(quote):
                return contract_size * pip_size * pair_mid
            if pair.startswith(quote) and pair.endswith(account_currency):
                return contract_size * pip_size / max(pair_mid, 1e-8)
            if pair.startswith(account_currency) and pair.endswith(base):
                return contract_size * pip_size * pair_mid / max(mid, 1e-8)
            if pair.startswith(base) and pair.endswith(account_currency):
                return contract_size * pip_size * max(mid, 1e-8) / max(pair_mid, 1e-8)

        return contract_size * pip_size
    except Exception as exc:
        logger.warning("Failed to estimate pip value for %s: %s; using default 1.0", symbol, exc)
        return 1.0


def load_model_and_features(model_path: str, features_path: str) -> tuple[Any, list[str]]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature list not found: {features_path}")

    model = joblib.load(model_path)
    with open(features_path, "r", encoding="utf-8") as fh:
        features = json.load(fh)
    return model, features


def fetch_m15_candles(symbol: str, n: int = 200) -> pd.DataFrame:
    if mt5 is None:
        raise ImportError("MetaTrader5 package is not installed")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, n)
    if rates is None:
        raise RuntimeError(f"Could not fetch rates for {symbol}")
    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
    return df


def prepare_features(df: pd.DataFrame, feature_engine: FeatureEngine) -> pd.DataFrame:
    out = feature_engine.add_technical_indicators(df)
    out = feature_engine.compute_statistical_features(out)
    return out


def infer_signal(model: Any, features: list[str], X_row: pd.DataFrame, threshold: float) -> int | None:
    missing = [f for f in features if f not in X_row.columns]
    if missing:
        logger.warning("Missing features at inference time: %s. Filling with 0s.", missing)
        for f in missing:
            X_row[f] = 0.0

    X_input = X_row[features].astype(float)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]
        pos_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
    else:
        pred = model.predict(X_input)[0]
        pos_prob = 1.0 if int(pred) == 1 else 0.0

    logger.info("Model probability for positive class: %.4f", pos_prob)

    if pos_prob >= threshold:
        return 1
    if pos_prob <= (1.0 - threshold):
        return -1
    return None


def compute_atr_from_df(df: pd.DataFrame, length: int = 14) -> float:
    if "atr_14" in df.columns:
        return float(df["atr_14"].iloc[-1])

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(length).mean().iloc[-1]
    return float(atr)


def main(
    symbol: str,
    model_path: str,
    features_path: str,
    threshold: float,
    risk_pct: float,
    pip_value_override: float | None,
    live: bool,
) -> None:
    settings = Settings.from_env()
    feature_engine = FeatureEngine()

    logger.info("Starting live execution engine for %s (live=%s)", symbol, live)

    model, expected_features = load_model_and_features(model_path, features_path)

    executor = LiveExecutor(settings=settings)
    if mt5 is None:
        logger.error("MetaTrader5 package is not installed; cannot proceed")
        return

    executor.connect()

    last_minute_executed: int | None = None

    try:
        while True:
            now = datetime.now()
            minute = now.minute
            if minute % 15 == 0 and minute != last_minute_executed:
                logger.info("Tick at %s - evaluating signal", now.isoformat())
                try:
                    df = fetch_m15_candles(symbol, n=300)
                    feats = prepare_features(df, feature_engine)
                    if feats.empty:
                        logger.warning("No features produced for %s, skipping this cycle", symbol)
                        last_minute_executed = minute
                        time.sleep(60)
                        continue

                    X_row = feats.iloc[[-1]].copy()

                    signal = infer_signal(model, expected_features, X_row, threshold)
                    if signal is None:
                        logger.info("No confident signal at this candle, skipping")
                        last_minute_executed = minute
                        time.sleep(60)
                        continue

                    open_positions = executor.get_open_positions(symbol)
                    if open_positions:
                        logger.info("Existing open positions found for %s: %s. Skipping new trade.", symbol, open_positions)
                        last_minute_executed = minute
                        time.sleep(60)
                        continue

                    atr_value = compute_atr_from_df(feats)
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        logger.error("Could not obtain tick for %s, skipping", symbol)
                        last_minute_executed = minute
                        time.sleep(60)
                        continue

                    sl_price, tp_price = RiskManager().calculate_dynamic_sl_tp(
                        symbol_info=tick,
                        atr_value=atr_value,
                    )

                    entry_price = float(tick.ask) if signal == 1 else float(tick.bid)
                    if signal == -1:
                        sl_price, tp_price = tp_price, sl_price

                    si = mt5.symbol_info(symbol)
                    point = float(getattr(si, "point", 0.0001))
                    stop_loss_pips = abs(entry_price - sl_price) / point

                    if pip_value_override is not None:
                        pip_value = pip_value_override
                    else:
                        account_info = mt5.account_info()
                        account_currency = getattr(account_info, "currency", "USD") if account_info is not None else "USD"
                        pip_value = estimate_pip_value(symbol, account_currency)
                        logger.info("Estimated pip value for %s in account currency %s: %.6f", symbol, account_currency, pip_value)

                    account_info = mt5.account_info()
                    if account_info is None:
                        logger.error("Could not fetch account info, aborting this trade cycle")
                        last_minute_executed = minute
                        time.sleep(60)
                        continue
                    balance = float(account_info.balance)

                    vol_min = getattr(si, "volume_min", None)
                    vol_max = getattr(si, "volume_max", None)
                    vol_step = getattr(si, "volume_step", None)

                    lot_size = RiskManager().calculate_lot_size(
                        balance=balance,
                        risk_per_trade_pct=risk_pct,
                        stop_loss_pips=stop_loss_pips,
                        pip_value=pip_value,
                        symbol_volume_min=vol_min,
                        symbol_volume_max=vol_max,
                        symbol_volume_step=vol_step,
                    )

                    logger.info(
                        "Placing trade: symbol=%s signal=%s entry=%.5f sl=%.5f tp=%.5f lots=%.2f",
                        symbol,
                        "BUY" if signal == 1 else "SELL",
                        entry_price,
                        sl_price,
                        tp_price,
                        lot_size,
                    )

                    if not live:
                        logger.info("--live flag not set; running in DEMO/SIMULATE mode. No order will be sent.")
                    else:
                        success = executor.execute_trade(symbol, signal, sl_price, tp_price, lot_size)
                        if not success:
                            logger.error("Trade execution reported failure for %s", symbol)

                except Exception as exc:
                    logger.exception("Error during live execution loop: %s", exc)
                finally:
                    last_minute_executed = minute

            time.sleep(30)
    finally:
        try:
            executor.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live execution engine (Demo by default). Runs on M15 candle close.")
    parser.add_argument("--symbol", required=True, help="Market symbol to trade, e.g. EURUSD")
    parser.add_argument(
        "--model-path",
        default=os.path.join(os.getcwd(), "data", "model_releases", "model_v1.pkl"),
        help="Path to trained model (joblib)",
    )
    parser.add_argument(
        "--features-path",
        default=os.path.join(os.getcwd(), "data", "model_releases", "model_v1_features.json"),
        help="Path to JSON file with ordered feature names",
    )
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold for entering trades")
    parser.add_argument("--risk", type=float, default=0.01, help="Risk per trade as fraction of balance (e.g. 0.01)")
    parser.add_argument("--pip-value", type=float, default=None, help="Override pip value per lot (optional)")
    parser.add_argument("--live", action="store_true", help="Enable live trading (must be explicitly set)")

    args = parser.parse_args()

    main(
        symbol=args.symbol,
        model_path=args.model_path,
        features_path=args.features_path,
        threshold=args.threshold,
        risk_pct=args.risk,
        pip_value_override=args.pip_value,
        live=args.live,
    )
