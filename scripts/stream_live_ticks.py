from __future__ import annotations

import asyncio

from smarttred.config.settings import Settings
from smarttred.data_pipeline.mt5_extractor import MT5DataExtractor


async def main() -> None:
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
        await extractor.stream_ticks_to_parquet(
            symbol="EURUSD",
            output_dir="data/raw/forex",
            duration_seconds=10,
        )
    finally:
        extractor.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
