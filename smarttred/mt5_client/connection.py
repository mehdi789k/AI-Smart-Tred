from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - runtime dependency
    mt5 = None

from smarttred.config.settings import Settings


@dataclass
class MT5Connection:
    """Thin, production-oriented wrapper around the MT5 Python API."""

    settings: Settings
    timeout: int = 10000
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("smarttred.mt5"))
    connected: bool = False

    def __post_init__(self) -> None:
        self.settings.validate_mt5()

    def connect(self) -> bool:
        """Establish a connection to the live MT5 terminal."""
        if mt5 is None:
            raise ImportError("MetaTrader5 is not installed on this machine.")

        result = mt5.initialize(
            path=self.settings.mt5_terminal_path,
            login=self.settings.mt5_login,
            password=self.settings.mt5_password,
            server=self.settings.mt5_server,
            timeout=self.timeout,
        )
        if not result:
            error = mt5.last_error()
            self.logger.error("MT5 initialization failed: %s", error)
            self.connected = False
            raise ConnectionError(f"Unable to connect to MT5: {error}")

        self.connected = True
        self.logger.info("MT5 connection successful for login %s.", self.settings.mt5_login)
        return True

    def disconnect(self) -> None:
        """Close the MT5 connection."""
        if self.connected and mt5 is not None:
            mt5.shutdown()
            self.connected = False
            self.logger.info("MT5 disconnected.")

    def account_info(self) -> dict[str, Any] | None:
        """Return account information if connected."""
        if not self.connected:
            return None
        info = mt5.account_info()
        if info is None:
            return None
        return {
            "login": info.login,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "currency": info.currency,
        }

    def market_data(self, symbol: str, timeframe: int, count: int = 100) -> list[dict[str, Any]]:
        """Return a limited number of OHLCV bars for a symbol."""
        if not self.connected:
            raise ConnectionError("MT5 is not connected.")
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            return []

        rate_dtype = getattr(rates, "dtype", None)
        field_names = list(getattr(rate_dtype, "names", []) or [])
        out: list[dict[str, Any]] = []
        for rate in rates:
            if isinstance(rate, dict):
                out.append(dict(rate))
            elif hasattr(rate, "_asdict"):
                out.append(dict(rate._asdict()))
            elif isinstance(rate, np.void):
                values = rate.tolist()
                out.append({name: value for name, value in zip(field_names, values)})
            else:
                try:
                    out.append(dict(vars(rate)))
                except TypeError:
                    out.append({str(index): value for index, value in enumerate(rate)})
        return out
