# AI Smart Trader

A Python-based algorithmic trading project that connects MetaTrader 5 (MT5) with machine learning pipelines, feature engineering, and data storage in Parquet/Time-Series-friendly formats.

## Project goals

- Connect to MetaTrader 5 for historical and live market data
- Store OHLCV and tick data efficiently
- Build technical indicators and ML-ready features
- Generate labels with a triple-barrier style target logic
- Support backtesting and model retraining workflows

## Quick start

```bash
python -m venv .venv
. .venv\Scripts\activate
pip install -r requirements.txt
```

## Real MT5 configuration

This project reads the live MT5 settings from a local `.env` file. Keep the `.env` file out of source control. Example values are in `.env.example`.

```bash
copy .env.example .env
```

Then update the values for your account, especially:
- `MT5_TERMINAL_PATH`
- `MT5_LOGIN`
- `MT5_PASSWORD`
- `MT5_SERVER`

Validation command for a real account connection:

```bash
py scripts\validate_mt5_connection.py
```

## Structure

```text
smarttred/
├── config/
├── data_pipeline/
├── features/
├── utils/
└── __init__.py

data/
├── raw/
├── features/
├── trades/
```

## Safety notes

- Use a demo MT5 account for testing.
- Never deploy live trading without paper trading validation.
- Keep credentials in environment variables and do not commit them.

## License

This project is for educational and experimental trading research.
