from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from smarttred.config.settings import Settings
from smarttred.trading.risk_manager import RiskManager

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - runtime dependency
    mt5 = None


@dataclass
class ExecutionRequest:
    """Simple representation of a live trading order request."""

    symbol: str
    side: int
    lot_size: float
    stop_loss: float | None = None
    take_profit: float | None = None
    comment: str = "smarttred"


class MT5OrderExecutor:
    """Execute risk-managed orders through the live MT5 terminal.

    This layer keeps the production path separated from unit-test code by using a
    thin wrapper around the MetaTrader5 API and an explicit risk calculation step.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        timeout: int = 10000,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.timeout = timeout
        self.logger = logger or logging.getLogger("smarttred.trading")
        self.connected = False

    def connect(self) -> bool:
        """Initialize the MT5 terminal connection."""
        if mt5 is None:
            raise ImportError("MetaTrader5 is not installed on this machine.")

        self.settings.validate_mt5()
        initialized = mt5.initialize(
            path=self.settings.mt5_terminal_path,
            login=self.settings.mt5_login,
            password=self.settings.mt5_password,
            server=self.settings.mt5_server,
            timeout=self.timeout,
        )
        if not initialized:
            error = mt5.last_error()
            self.logger.error("MT5 initialization failed: %s", error)
            self.connected = False
            raise ConnectionError(f"Unable to connect to MT5: {error}")

        self.connected = True
        self.logger.info("Connected to MT5 for login=%s", self.settings.mt5_login)
        return True

    def disconnect(self) -> None:
        """Close the MT5 terminal handle."""
        if self.connected and mt5 is not None:
            mt5.shutdown()
            self.connected = False
            self.logger.info("MT5 disconnected.")

    def ensure_connected(self) -> None:
        """Validate the MT5 session before sending orders."""
        if not self.connected:
            self.connect()

    def get_open_positions(self, symbol: str) -> list:
        """Return all open positions for a symbol.

        This is the compatibility wrapper for the Phase-7 LiveExecutor API.
        """
        self.ensure_connected()
        try:
            positions = mt5.positions_get(symbol=symbol)
        except Exception as exc:  # pragma: no cover - runtime
            self.logger.exception("Error fetching positions for %s: %s", symbol, exc)
            return []
        return list(positions) if positions else []

    def get_filling_mode(self, symbol: str) -> int:
        """Best-effort detection of the broker order filling mode."""
        if mt5 is None:
            raise ImportError("MetaTrader5 is not installed on this machine.")
        self.ensure_connected()
        info = mt5.symbol_info(symbol)
        if info is None:
            self.logger.warning("symbol_info unavailable for %s; defaulting to IOC", symbol)
            return getattr(mt5, "ORDER_FILLING_IOC", 0)

        for attr in ("filling_mode", "order_filling_mode"):
            val = getattr(info, attr, None)
            if val is None:
                continue
            try:
                iv = int(val)
                for candidate in (getattr(mt5, "ORDER_FILLING_FOK", None), getattr(mt5, "ORDER_FILLING_IOC", None), getattr(mt5, "ORDER_FILLING_RETURN", None)):
                    if candidate is not None and iv == candidate:
                        return iv
            except Exception:
                pass
            sval = str(val).upper()
            if "FOK" in sval:
                return getattr(mt5, "ORDER_FILLING_FOK", 0)
            if "IOC" in sval:
                return getattr(mt5, "ORDER_FILLING_IOC", 0)
            if "RETURN" in sval:
                return getattr(mt5, "ORDER_FILLING_RETURN", 0)

        return getattr(mt5, "ORDER_FILLING_IOC", 0)

    def execute_trade(self, symbol: str, signal: int, sl_price: float, tp_price: float, lot_size: float) -> bool:
        """Send a market order with optional SL/TP and return True on success."""
        if mt5 is None:
            raise ImportError("MetaTrader5 is not installed on this machine.")
        self.ensure_connected()

        if signal not in (-1, 1):
            raise ValueError("signal must be either 1 (buy) or -1 (sell).")

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self.logger.error("Could not fetch tick for %s before order submission.", symbol)
            return False

        bid, ask = float(tick.bid), float(tick.ask)
        order_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
        trade_price = ask if signal == 1 else bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": order_type,
            "price": trade_price,
            "sl": float(sl_price) if sl_price is not None else 0.0,
            "tp": float(tp_price) if tp_price is not None else 0.0,
            "deviation": 10,
            "magic": 2025001,
            "comment": "smarttred",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.get_filling_mode(symbol),
        }

        try:
            result = mt5.order_send(request)
        except Exception as exc:  # pragma: no cover - runtime
            self.logger.exception("Exception while sending order for %s: %s", symbol, exc)
            return False

        if result is None:
            self.logger.error("mt5.order_send returned None for %s", symbol)
            return False

        retcode = getattr(result, "retcode", None)
        comment = getattr(result, "comment", "")
        if retcode in {getattr(mt5, "TRADE_RETCODE_DONE", None), getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", None)}:
            self.logger.info(
                "Trade executed: symbol=%s direction=%s lots=%s sl=%s tp=%s retcode=%s",
                symbol,
                "BUY" if signal == 1 else "SELL",
                lot_size,
                sl_price,
                tp_price,
                retcode,
            )
            return True

        self.logger.error("Order rejected: symbol=%s retcode=%s comment=%s", symbol, retcode, comment)
        return False

    def get_bid_ask(self, symbol: str) -> tuple[float, float]:
        """Return current bid/ask for a market symbol."""
        self.ensure_connected()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Could not fetch market tick for {symbol!r}.")
        return float(tick.bid), float(tick.ask)

    def calculate_risk_lot_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        *,
        account_balance: float,
        risk_per_trade: float = 0.01,
        pip_value: float = 10.0,
        pip_size: float = 0.0001,
    ) -> float:
        """Compute a lot size aligned with account risk policy."""
        manager = RiskManager(account_balance=account_balance, risk_per_trade=risk_per_trade)
        return manager.calculate_lot_size(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            pip_value=pip_value,
            pip_size=pip_size,
        )

    def place_market_order(
        self,
        *,
        symbol: str,
        side: int,
        lot_size: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        comment: str = "smarttred",
        deviation: int = 10,
    ) -> Any:
        """Send a market order to MT5 with optional stop and target."""
        self.ensure_connected()

        if side not in (-1, 1):
            raise ValueError("side must be either 1 (buy) or -1 (sell).")
        if lot_size <= 0:
            raise ValueError("lot_size must be positive.")

        bid, ask = self.get_bid_ask(symbol)
        order_type = mt5.ORDER_TYPE_BUY if side == 1 else mt5.ORDER_TYPE_SELL
        trade_price = ask if side == 1 else bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": order_type,
            "price": trade_price,
            "sl": float(stop_loss) if stop_loss is not None else 0.0,
            "tp": float(take_profit) if take_profit is not None else 0.0,
            "deviation": deviation,
            "magic": 2025001,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"MT5 order_send returned None for {symbol}.")

        if result.retcode not in {mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL}:
            raise RuntimeError(f"Order rejected for {symbol}: {result.retcode} / {result.comment}")

        self.logger.info(
            "Order sent: symbol=%s side=%s lots=%s sl=%s tp=%s",
            symbol,
            side,
            lot_size,
            stop_loss,
            take_profit,
        )
        return result

    def place_risk_managed_order(
        self,
        *,
        symbol: str,
        signal: int,
        entry_price: float | None = None,
        stop_loss_pips: float = 30.0,
        take_profit_pips: float | None = None,
        account_balance: float = 10000.0,
        risk_per_trade: float = 0.01,
        pip_value: float = 10.0,
        pip_size: float = 0.0001,
        comment: str = "smarttred",
    ) -> Any:
        """Create a trade using risk-based position sizing and MT5 order routing."""
        if signal not in (-1, 1):
            raise ValueError("signal must be either 1 or -1.")

        if entry_price is None:
            bid, ask = self.get_bid_ask(symbol)
            entry_price = ask if signal == 1 else bid

        if stop_loss_pips <= 0:
            raise ValueError("stop_loss_pips must be positive.")

        direction = 1.0 if signal == 1 else -1.0
        stop_distance = pip_size * stop_loss_pips
        stop_loss_price = entry_price - (direction * stop_distance)
        if signal == -1:
            stop_loss_price = entry_price + stop_distance

        if take_profit_pips is None:
            take_profit_pips = stop_loss_pips * 2.0

        take_profit_price = entry_price + (direction * pip_size * take_profit_pips)

        lot_size = self.calculate_risk_lot_size(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            account_balance=account_balance,
            risk_per_trade=risk_per_trade,
            pip_value=pip_value,
            pip_size=pip_size,
        )

        return self.place_market_order(
            symbol=symbol,
            side=signal,
            lot_size=lot_size,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            comment=comment,
        )
