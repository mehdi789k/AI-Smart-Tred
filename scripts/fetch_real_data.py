"""
Real Data Fetcher for AI Smart Trader
Supports:
1. Direct MetaTrader5 connection (Windows only)
2. CSV Import (Cross-platform)
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
TIMEFRAME = "M15"  # M1, M5, M15, H1, H4, D1
DAYS_TO_FETCH = 365  # Number of days of historical data

def get_timeframe_constant(tf_str):
    """Map string timeframe to MT5 constant or Pandas offset."""
    tf_map = {
        "M1": "1min", "M5": "5min", "M15": "15min", 
        "M30": "30min", "H1": "1h", "H4": "4h", "D1": "1D"
    }
    return tf_map.get(tf_str, "15min")

def fetch_from_mt5(symbol: str, days: int, tf_str: str):
    """Fetch data directly from MetaTrader5 (Windows Only)."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        logger.error("MetaTrader5 library not found. Please run on Windows with MT5 installed.")
        return None

    if not mt5.initialize():
        logger.error(f"MT5 initialization failed: {mt5.last_error()}")
        return None

    logger.info(f"Connected to MT5. Fetching {days} days of {symbol}...")
    
    rates = mt5.copy_rates_from_pos(
        symbol, 
        getattr(mt5, f"TIMEFRAME_{tf_str.replace('M', 'M').replace('H', 'H').replace('D', 'D')}"), 
        0, 
        days * 24 * 60 // int(''.join(filter(str.isdigit, tf_str))) if tf_str.startswith('M') else days * 24 // int(''.join(filter(str.isdigit, tf_str))) if tf_str.startswith('H') else days
    )
    
    # Fallback: fetch a large chunk simply
    rates = mt5.copy_rates_from(symbol, getattr(mt5, f"TIMEFRAME_{tf_str}"), datetime.now() - timedelta(days=days), datetime.now())

    if rates is None or len(rates) == 0:
        logger.warning(f"No data received for {symbol}")
        mt5.shutdown()
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'time': 'timestamp', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'tick_volume': 'volume'})
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    mt5.shutdown()
    logger.info(f"Fetched {len(df)} rows for {symbol}")
    return df

def fetch_from_csv(symbol: str, csv_path: str):
    """Fetch data from a CSV file (Cross-platform)."""
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return None
    
    logger.info(f"Loading data from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Standardize columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close']
    if 'time' in df.columns and 'timestamp' not in df.columns:
        df['timestamp'] = df['time']
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    # Select and order columns
    if 'volume' not in df.columns:
        df['volume'] = 0
        
    df = df[required_cols + ['volume']]
    df = df.dropna()
    logger.info(f"Loaded {len(df)} rows from CSV")
    return df

def main():
    mode = os.getenv("DATA_SOURCE", "mock")  # Options: mt5, csv, mock
    
    for symbol in SYMBOLS:
        logger.info(f"Processing {symbol}...")
        df = None
        
        if mode == "mt5":
            df = fetch_from_mt5(symbol, DAYS_TO_FETCH, TIMEFRAME)
        elif mode == "csv":
            csv_file = PROJECT_ROOT / "data" / "exports" / f"{symbol}_{TIMEFRAME}.csv"
            df = fetch_from_csv(symbol, str(csv_file))
        else:
            logger.warning("Running in MOCK mode (generating synthetic realistic data).")
            # Generate realistic mock data for testing pipeline without MT5
            dates = pd.date_range(end=datetime.now(), periods=10000, freq=get_timeframe_constant(TIMEFRAME))
            np.random.seed(42)
            walk = np.cumsum(np.random.randn(10000))
            price = 1.1000 + walk * 0.0001
            df = pd.DataFrame({
                'timestamp': dates,
                'open': price + np.random.randn(10000)*0.00005,
                'high': price + np.abs(np.random.randn(10000))*0.0001,
                'low': price - np.abs(np.random.randn(10000))*0.0001,
                'close': price + np.random.randn(10000)*0.00005,
                'volume': np.random.randint(100, 1000, 10000)
            })

        if df is not None:
            # Save to Parquet for efficiency
            output_file = RAW_DATA_DIR / f"{symbol}_{TIMEFRAME}.parquet"
            df.to_parquet(output_file, index=False)
            logger.info(f"Data saved to {output_file}")

if __name__ == "__main__":
    # Add success level to logger if missing
    if not hasattr(logging, 'success'):
        logging.success = lambda msg, *args, **kwargs: logging.info(f"SUCCESS: {msg}", *args, **kwargs)
    
    main()
