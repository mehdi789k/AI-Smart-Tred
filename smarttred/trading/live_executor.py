from __future__ import annotations

import logging
from typing import List, Optional

from smarttred.config.settings import Settings
from smarttred.trading.risk_manager import RiskManager

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - runtime dependency
    mt5 = None


logger = logging.getLogger("smarttred.trading.live_executor")


class LiveExecutor:
    """Live trade executor that talks to MetaTrader5.

    Methods provided:
    - __init__(settings): stores settings and prepares connection state
    - get_open_positions(symbol): returns list of open positions for symbol
    - get_filling_mode(symbol): best-effort detection of broker's filling mode
    - execute_trade(symbol, signal, sl_price, tp_price, lot_size): send market order

    This class uses demo trading by default; enabling live trading requires an
    explicit configuration/flag in the orchestration script.
    """

    def __init__(self, settings: Optional[Settings] = None, logger: Optional[logging.Logger] = None) -> None:
        self.settings = settings or Settings.from_env()
        self.logger = logger or logging.getLogger("smarttred.trading.live_executor")
        self.connected = False

    def connect(self) -> bool:
        if mt5 is None:
            raise ImportError("MetaTrader5 package is not installed")

        # Validate settings and initialize MT5
        self.settings.validate_mt5()
        self.logger.info("Initializing MT5 with login=%s", self.settings.mt5_login)
        initialized = mt5.initialize(
            path=self.settings.mt5_terminal_path,
            login=self.settings.mt5_login,
            password=self.settings.mt5_password,
            server=self.settings.mt5_server,
        )
        if not initialized:
            err = mt5.last_error()
            self.logger.error("MT5 initialize failed: %s", err)
            self.connected = False
            raise ConnectionError(f"Unable to initialize MT5: {err}")

        self.connected = True
        self.logger.info("MT5 initialized and connected")
        return True

    def disconnect(self) -> None:
        if self.connected and mt5 is not None:
            mt5.shutdown()
            self.connected = False
            self.logger.info("MT5 shutdown complete")

    def ensure_connected(self) -> None:
        if not self.connected:
            self.connect()

    def get_open_positions(self, symbol: str) -> List:
        """Return a list of open positions for the given symbol."""
        self.ensure_connected()
        try:
            positions = mt5.positions_get(symbol=symbol)
        except Exception as exc:  # pragma: no cover - runtime
            self.logger.exception("Error fetching positions for %s: %s", symbol, exc)
            return []
        if positions is None:
            return []
        return list(positions)

    def get_filling_mode(self, symbol: str) -> int:
        """Best-effort detection of the symbol's order filling mode.

        Attempts to read common fields from mt5.symbol_info and falls back
        to ORDER_FILLING_IOC which is commonly supported.
        """
        if mt5 is None:
            raise ImportError("MetaTrader5 package not available")
        self.ensure_connected()

        si = mt5.symbol_info(symbol)
        if si is None:
            self.logger.warning("symbol_info not available for %s; using IOC filling mode", symbol)
            return mt5.ORDER_FILLING_IOC

        # Common attribute names that may contain filling mode
        possible_attrs = ["filling_mode", "order_filling_mode", "filling_mode_name"]
        for attr in possible_attrs:
            val = getattr(si, attr, None)
            if val is None:
                continue
            # If the attribute is already an int constant used by mt5, return it
            try:
                if int(val) in {mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN}:
                    return int(val)
            except Exception:
                pass
            # If it's a string like 'FOK' map it
            sval = str(val).upper()
            if "FOK" in sval:
                return mt5.ORDER_FILLING_FOK
            if "IOC" in sval:
                return mt5.ORDER_FILLING_IOC
            if "RETURN" in sval or "RETURN" in sval:
                return mt5.ORDER_FILLING_RETURN

        # As a last attempt, check 'trade' attributes for common constants
        try:
            # Some platforms expose 'volume_step' etc.; no direct filling field.
            # Default to IOC which works for most brokers
            self.logger.debug("Could not determine filling mode for %s, defaulting to IOC", symbol)
        except Exception:
            pass
        return mt5.ORDER_FILLING_IOC

    def execute_trade(self, symbol: str, signal: int, sl_price: float, tp_price: float, lot_size: float) -> bool:
        """Construct and send a market order.

        signal: 1 for BUY, -1 for SELL
        Returns True on success, False otherwise.
        """
        if mt5 is None:
            raise ImportError("MetaTrader5 is not installed")
        self.ensure_connected()

        if signal not in (1, -1):
            raise ValueError("signal must be 1 (buy) or -1 (sell)")

        # Validate symbol and get current prices
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self.logger.error("Could not get tick for symbol %s", symbol)
            return False

        bid, ask = float(tick.bid), float(tick.ask)
        price = ask if signal == 1 else bid

        order_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL

        filling_mode = self.get_filling_mode(symbol)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": order_type,
            "price": float(price),
            "sl": float(sl_price) if sl_price is not None else 0.0,
            "tp": float(tp_price) if tp_price is not None else 0.0,
            "deviation": 20,
            "magic": 2025001,
            "comment": "smarttred-live",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        try:
            result = mt5.order_send(request)
        except Exception as exc:  # pragma: no cover - runtime
            self.logger.exception("Exception while sending order for %s: %s", symbol, exc)
            return False

        if result is None:
            self.logger.error("mt5.order_send returned None for %s", symbol)
            return False

        # result has attributes 'retcode' and 'comment' among others
        retcode = getattr(result, "retcode", None)
        comment = getattr(result, "comment", "")
        if retcode in (getattr(mt5, "TRADE_RETCODE_DONE", None), getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", None)):
            self.logger.info(
                "Trade executed: symbol=%s side=%s lots=%.2f sl=%s tp=%s retcode=%s",
                symbol,
                "BUY" if signal == 1 else "SELL",
                lot_size,
                sl_price,
                tp_price,
                retcode,
            )
            return True

        # Log detailed error
        self.logger.error(
            "Order failed: symbol=%s retcode=%s comment=%s result=%s",
            symbol,
            retcode,
            comment,
            result,
        )
        return False
