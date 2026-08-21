"""
Generate synthetic OHLCV data for backtesting and demo purposes.
Creates sample Parquet files in data/raw/ directory.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_ohlcv(
    symbol: str = "EURUSD",
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    timeframe: str = "M15",
    initial_price: float = 1.1000,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with realistic price movements."""
    np.random.seed(seed)
    
    # Create date range (15-minute bars)
    dates = pd.date_range(start=start_date, end=end_date, freq="15min")
    n_bars = len(dates)
    
    # Generate random walk for close prices
    # Start at a realistic EURUSD price (~1.1000)
    initial_price = 1.1000
    
    # Daily volatility ~0.5%, scaled to 15-min
    daily_vol = 0.005
    intraday_vol = daily_vol / np.sqrt(96)  # 96 15-min bars per day
    
    returns = np.random.normal(0, intraday_vol, n_bars)
    close_prices = initial_price * np.cumprod(1 + returns)
    
    # Generate OHLC from close prices
    # Add some intrabar noise
    bar_noise = np.random.uniform(-0.0002, 0.0002, size=(n_bars, 2))
    
    high_prices = np.maximum(close_prices, close_prices + bar_noise[:, 0])
    low_prices = np.minimum(close_prices, close_prices - bar_noise[:, 1])
    
    # Open is previous close (with small gap)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = initial_price
    
    # Add realistic volume (higher during trading hours)
    base_volume = 1000
    hour_factor = np.array([d.hour for d in dates]).reshape(-1)
    # Higher volume during London/NY sessions (8-17 UTC)
    volume_multiplier = np.where((hour_factor >= 8) & (hour_factor <= 17), 2.0, 0.5)
    volumes = (base_volume * volume_multiplier * np.random.uniform(0.5, 1.5, n_bars)).astype(int)
    
    df = pd.DataFrame({
        'time': dates.astype(np.int64) // 10**9,  # Unix timestamp
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'tick_volume': volumes,
    })
    
    return df

def main():
    """Generate and save synthetic data for multiple symbols."""
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    output_dir = "data/raw"
    
    os.makedirs(output_dir, exist_ok=True)
    
    for symbol in symbols:
        print(f"Generating synthetic data for {symbol}...")
        
        if symbol == "USDJPY":
            df = generate_synthetic_ohlcv(symbol, initial_price=145.00, seed=42)
        elif symbol == "XAUUSD":
            df = generate_synthetic_ohlcv(symbol, initial_price=1950.00, seed=43)
        else:
            df = generate_synthetic_ohlcv(symbol, seed=42)
        
        output_path = os.path.join(output_dir, f"{symbol}_synthetic.parquet")
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df)} bars to {output_path}")
    
    print("\nSynthetic data generation complete!")
    print("You can now run feature engineering scripts on this data.")

if __name__ == "__main__":
    main()
