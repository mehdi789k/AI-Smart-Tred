import os

from smarttred.config.settings import Settings


def test_settings_from_env(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("MT5_TERMINAL_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
    monkeypatch.setenv("MT5_LOGIN", "123456")
    monkeypatch.setenv("MT5_PASSWORD", "secret")
    monkeypatch.setenv("MT5_SERVER", "Broker-Server")
    monkeypatch.setenv("STORAGE_BASE", "custom_data")
    monkeypatch.setenv("DB_NAME", "custom_db.db")

    settings = Settings.from_env()

    assert settings.app_env == "production"
    assert settings.mt5_login == 123456
    assert settings.mt5_password == "secret"
    assert settings.mt5_server == "Broker-Server"
    assert settings.storage_base == "custom_data"
    assert settings.db_name == "custom_db.db"


def test_settings_validate_mt5() -> None:
    settings = Settings(
        app_env="development",
        mt5_terminal_path=r"C:\Program Files\MetaTrader 5\terminal64.exe",
        mt5_login=123456,
        mt5_password="pass",
        mt5_server="Broker-Server",
    )

    settings.validate_mt5()
