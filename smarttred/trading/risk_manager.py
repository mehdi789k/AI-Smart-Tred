from __future__ import annotations

import logging
from typing import Tuple


logger = logging.getLogger("smarttred.trading.risk_manager")


class RiskManager:
    """Risk manager for calculating lot sizes and dynamic SL/TP.

    This class supports both the current Phase-7 API and the project's legacy
    RiskManager API used by the existing tests and earlier executors.
    """

    def __init__(self, account_balance: float | None = None, risk_per_trade: float = 0.01) -> None:
        if account_balance is not None and account_balance <= 0:
            raise ValueError("account_balance must be greater than zero.")
        if not 0 < risk_per_trade < 1:
            raise ValueError("risk_per_trade must be between 0 and 1.")

        self.account_balance = float(account_balance) if account_balance is not None else None
        self.risk_per_trade = float(risk_per_trade)

    @property
    def risk_amount(self) -> float:
        if self.account_balance is None:
            raise ValueError("account_balance is not configured for this RiskManager instance.")
        return self.account_balance * self.risk_per_trade

    @staticmethod
    def _round_lot(lot: float, min_lot: float, max_lot: float, step: float) -> float:
        """Round a raw lot value to the broker-allowed step and clamp to min/max."""
        if lot <= 0:
            return 0.0
        try:
            steps = round((lot - min_lot) / step)
        except Exception:
            return max(min_lot, min(lot, max_lot))
        rounded = min_lot + steps * step
        rounded = max(min_lot, min(rounded, max_lot))
        return float(round(rounded, 8))

    def calculate_lot_size(self, *args, **kwargs) -> float:
        """Calculate lot size in a backward-compatible way.

        Supported call styles:
        1) Legacy: manager.calculate_lot_size(entry_price, stop_loss_price, pip_value=10.0, pip_size=0.0001)
        2) Phase-7: manager.calculate_lot_size(balance=..., risk_per_trade_pct=..., stop_loss_pips=..., pip_value=...)
        """
        # Phase-7 style keywords
        balance = kwargs.get("balance")
        risk_per_trade_pct = kwargs.get("risk_per_trade_pct")
        stop_loss_pips = kwargs.get("stop_loss_pips")
        pip_value = kwargs.get("pip_value")

        entry_price = kwargs.get("entry_price")
        stop_loss_price = kwargs.get("stop_loss_price")

        if (entry_price is not None or len(args) >= 1) and (stop_loss_price is not None or len(args) >= 2):
            if entry_price is None and len(args) >= 1:
                entry_price = float(args[0])
            if stop_loss_price is None and len(args) >= 2:
                stop_loss_price = float(args[1])
            if pip_value is None:
                pip_value = kwargs.get("pip_value", args[2] if len(args) > 2 else 10.0)
            pip_size = kwargs.get("pip_size", 0.0001)

            if self.account_balance is None:
                raise ValueError("account_balance must be configured before calling the legacy calculate_lot_size API.")
            stop_distance = abs(float(entry_price) - float(stop_loss_price))
            if stop_distance <= 0:
                raise ValueError("entry_price and stop_loss_price must differ.")
            pip_distance = stop_distance / float(pip_size)
            if pip_distance <= 0:
                raise ValueError("Calculated pip distance must be positive.")
            return self.risk_amount / (pip_distance * float(pip_value))

        if balance is None and len(args) >= 4:
            balance = float(args[0])
            risk_per_trade_pct = float(args[1])
            stop_loss_pips = float(args[2])
            pip_value = float(args[3])

        if balance is None:
            raise TypeError("calculate_lot_size requires either balance or entry_price/stop_loss_price arguments.")
        if risk_per_trade_pct is None:
            risk_per_trade_pct = self.risk_per_trade
        if stop_loss_pips is None:
            raise ValueError("stop_loss_pips must be provided for the Phase-7 API.")
        if pip_value is None:
            pip_value = 10.0

        balance = float(balance)
        risk_per_trade_pct = float(risk_per_trade_pct)
        stop_loss_pips = float(stop_loss_pips)
        pip_value = float(pip_value)

        if balance <= 0:
            raise ValueError("balance must be positive")
        if not 0 < risk_per_trade_pct < 1:
            raise ValueError("risk_per_trade_pct must be between 0 and 1")
        if stop_loss_pips <= 0:
            raise ValueError("stop_loss_pips must be positive")
        if pip_value <= 0:
            raise ValueError("pip_value must be positive")

        symbol_volume_min = float(kwargs.get("symbol_volume_min") or 0.01)
        symbol_volume_max = float(kwargs.get("symbol_volume_max") or 100.0)
        symbol_volume_step = float(kwargs.get("symbol_volume_step") or 0.01)

        risk_amount = balance * float(risk_per_trade_pct)
        loss_per_lot = float(stop_loss_pips) * float(pip_value)
        if loss_per_lot <= 0:
            raise ValueError("Computed loss_per_lot must be positive")

        raw_lots = risk_amount / loss_per_lot
        lot = self._round_lot(raw_lots, symbol_volume_min, symbol_volume_max, symbol_volume_step)

        if lot < symbol_volume_min:
            logger.warning(
                "Calculated lot (%.6f) is below broker minimum (%.6f); using min_lot",
                lot,
                symbol_volume_min,
            )
            return symbol_volume_min
        return lot

    def calculate_dynamic_sl_tp(
        self,
        symbol_info,
        atr_value: float,
        sl_atr_multiplier: float = 1.5,
        tp_atr_multiplier: float = 2.5,
    ) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit price levels based on ATR.

        symbol_info can be either a tick-like object with `bid`/`ask` attributes
        (mt5.Tick) or a symbol info that contains `point` and current tick can be
        retrieved externally. The function uses the mid price when bid/ask are present.

        Returns (sl_price, tp_price) using the mid price as reference for a BUY
        trade. For a SELL trade swap the two values.
        """
        if atr_value is None or atr_value <= 0:
            raise ValueError("atr_value must be a positive number")

        # Try to get a reasonable reference price
        price = None
        if hasattr(symbol_info, "bid") and hasattr(symbol_info, "ask"):
            try:
                price = (float(symbol_info.bid) + float(symbol_info.ask)) / 2.0
            except Exception:
                price = None

        # Fallback: try 'last' or 'close' attr
        if price is None and hasattr(symbol_info, "last"):
            price = float(getattr(symbol_info, "last"))
        if price is None and hasattr(symbol_info, "close"):
            price = float(getattr(symbol_info, "close"))

        if price is None:
            raise ValueError("symbol_info must provide a reference price (bid/ask/last/close)")

        sl_distance = float(atr_value) * float(sl_atr_multiplier)
        tp_distance = float(atr_value) * float(tp_atr_multiplier)

        sl_price = price - sl_distance
        tp_price = price + tp_distance

        # Round to symbol precision if available
        if hasattr(symbol_info, "digits") and getattr(symbol_info, "digits") is not None:
            digits = int(getattr(symbol_info, "digits"))
            sl_price = round(sl_price, digits)
            tp_price = round(tp_price, digits)

        return float(sl_price), float(tp_price)


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Return the Kelly fraction for a positive expectation strategy."""
    if not 0 <= win_rate <= 1:
        raise ValueError("win_rate must be between 0 and 1.")
    if avg_loss <= 0:
        raise ValueError("avg_loss must be greater than zero.")
    if avg_win <= 0:
        raise ValueError("avg_win must be greater than zero.")

    p = float(win_rate)
    q = 1.0 - p
    return (p * avg_win - q * avg_loss) / (avg_win * avg_loss)
