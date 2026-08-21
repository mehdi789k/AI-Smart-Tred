from __future__ import annotations


def _check_runtime() -> None:
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:  # pragma: no cover - runtime only
        raise RuntimeError("MetaTrader5 package is not installed. Install it on a Windows MT5 machine.") from exc

    from smarttred.config.settings import Settings
    from smarttred.mt5_client import MT5Connection

    settings = Settings.from_env()
    settings.validate_mt5()

    connection = MT5Connection(settings)
    connection.connect()
    try:
        account = connection.account_info()
        print("ACCOUNT_INFO", account)
        symbols = ["EURUSD", "GBPUSD", "XAUUSD", "BTCUSD"]
        for symbol in symbols:
            try:
                bars = connection.market_data(symbol=symbol, timeframe=mt5.TIMEFRAME_M5, count=3)
                print(symbol, len(bars))
                if bars:
                    print(bars[0])
                break
            except Exception:
                continue
    finally:
        connection.disconnect()


if __name__ == "__main__":
    _check_runtime()
