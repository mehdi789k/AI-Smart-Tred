from __future__ import annotations

import argparse
from datetime import datetime, timedelta

from smarttred.config.settings import Settings
from smarttred.data_pipeline.mt5_extractor import MT5DataExtractor


ALL_TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1", "Q1", "Y1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MT5 OHLCV data for a symbol across all supported timeframes."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="XAUUSD_l",
        help="MT5 symbol to download, e.g. XAUUSD_l or EURUSD",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to download backwards from now",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/forex",
        help="Directory where Parquet files are saved",
    )
    parser.add_argument(
        "--timeframes",
        nargs="*",
        default=ALL_TIMEFRAMES,
        help="Timeframes to download. Defaults to all supported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    settings.validate_mt5()

    extractor = MT5DataExtractor(
        login=settings.mt5_login,
        password=settings.mt5_password,
        server=settings.mt5_server,
        path=settings.mt5_terminal_path,
    )

    if not extractor.connect():
        raise RuntimeError("Could not connect to MT5 using the supplied configuration.")

    try:
        end = datetime.now()
        start = end - timedelta(days=args.days)
        files = extractor.download_timeframes(
            symbol=args.symbol,
            start=start,
            end=end,
            timeframes=args.timeframes,
            output_dir=args.output_dir,
        )
        print(f"[INFO] Download completed for {args.symbol}")
        for timeframe, file_path in files.items():
            print(f"  - {timeframe}: {file_path}")
    finally:
        extractor.disconnect()


if __name__ == "__main__":
    main()
