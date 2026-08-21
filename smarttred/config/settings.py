from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: str | Path = ".env") -> None:
    """Load environment variables from a local .env file if present."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass
class Settings:
    """Runtime settings for the project."""

    app_env: str = "development"
    mt5_terminal_path: str | None = None
    mt5_login: int | None = None
    mt5_password: str | None = None
    mt5_server: str | None = None
    storage_base: str = "data"
    db_name: str = "trades.db"

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        return cls(
            app_env=os.getenv("APP_ENV", "development"),
            mt5_terminal_path=os.getenv("MT5_TERMINAL_PATH"),
            mt5_login=int(os.getenv("MT5_LOGIN", "0")) or None,
            mt5_password=os.getenv("MT5_PASSWORD"),
            mt5_server=os.getenv("MT5_SERVER"),
            storage_base=os.getenv("STORAGE_BASE", "data"),
            db_name=os.getenv("DB_NAME", "trades.db"),
        )

    @property
    def is_local_dev(self) -> bool:
        return self.app_env.lower() in {"development", "dev"}

    def validate_mt5(self) -> None:
        missing = []
        if not self.mt5_terminal_path:
            missing.append("MT5_TERMINAL_PATH")
        if self.mt5_login in (None, 0):
            missing.append("MT5_LOGIN")
        if not self.mt5_password:
            missing.append("MT5_PASSWORD")
        if not self.mt5_server:
            missing.append("MT5_SERVER")
        if missing:
            raise ValueError(f"Missing required MT5 settings: {', '.join(missing)}")
