"""
Live Data Collector from MetaTrader 5
Fetches real-time OHLCV data and saves to parquet/SQLite
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from pathlib import Path
import sqlite3
import time
import json

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 not available. Install with: pip install MetaTrader5")
    print("   Note: MT5 only works on Windows with MT5 terminal installed.")


class MT5DataCollector:
    """Professional data collector for MetaTrader 5."""
    
    def __init__(
        self,
        terminal_path: Optional[str] = None,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        timeout: int = 30000
    ):
        self.terminal_path = terminal_path
        self.login = login
        self.password = password
        self.server = server
        self.timeout = timeout
        self.connected = False
        
        if MT5_AVAILABLE:
            self.connect()
    
    def connect(self) -> bool:
        """Connect to MetaTrader 5."""
        if not MT5_AVAILABLE:
            print("❌ MetaTrader5 library not available")
            return False
        
        # Initialize MT5
        init_params = {}
        if self.terminal_path:
            init_params['terminal_path'] = self.terminal_path
        if self.login:
            init_params['login'] = self.login
        if self.password:
            init_params['password'] = self.password
        if self.server:
            init_params['server'] = self.server
        init_params['timeout'] = self.timeout
        
        try:
            if mt5.initialize(**init_params):
                self.connected = True
                account_info = mt5.account_info()
                if account_info:
                    print(f"✅ Connected to MT5 - Account: {account_info.login}, Server: {account_info.server}")
                    print(f"   Balance: ${account_info.balance:.2f}, Equity: ${account_info.equity:.2f}")
                return True
            else:
                print(f"❌ MT5 initialization failed: {mt5.last_error()}")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MetaTrader 5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("🔌 Disconnected from MT5")
    
    def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        if not self.connected:
            return []
        
        symbols = mt5.symbols_get()
        if symbols is None:
            return []
        
        return [s.name for s in symbols if s.visible]
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "M1",
        bars: int = 1000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from MT5."""
        if not self.connected:
            print("❌ Not connected to MT5")
            return None
        
        # Map timeframe strings to MT5 constants
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        
        tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_M1)
        
        # Fetch rates
        if start_date and end_date:
            rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        
        if rates is None or len(rates) == 0:
            print(f"⚠️ No data received for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'
        }, inplace=True)
        
        # Select only OHLCV columns
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        df.set_index('time', inplace=True)
        
        print(f"✅ Fetched {len(df)} bars for {symbol} ({timeframe})")
        return df
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = "M1",
        bars: int = 1000,
        output_dir: str = "data/raw"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols."""
        results = {}
        
        for symbol in symbols:
            print(f"\n📊 Fetching data for {symbol}...")
            df = self.fetch_ohlcv(symbol, timeframe, bars)
            if df is not None:
                results[symbol] = df
                
                # Save to parquet
                output_path = Path(output_dir) / f"{symbol}_{timeframe}.parquet"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(output_path)
                print(f"💾 Saved to {output_path}")
        
        return results
    
    def save_to_sqlite(
        self,
        df: pd.DataFrame,
        symbol: str,
        db_path: str = "data/ticks.db"
    ):
        """Save data to SQLite database."""
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        
        # Prepare data
        table_name = f"ticks_{symbol.replace('/', '_')}"
        df_reset = df.reset_index()
        df_reset.rename(columns={'time': 'timestamp'}, inplace=True)
        df_reset['symbol'] = symbol
        df_reset['created_at'] = datetime.now()
        
        # Save to database
        df_reset.to_sql(table_name, conn, if_exists='append', index=False)
        
        conn.commit()
        conn.close()
        
        print(f"💾 Saved {len(df)} records to SQLite table: {table_name}")
    
    def stream_live_data(
        self,
        symbol: str,
        callback=None,
        duration_seconds: int = 60,
        save_to_db: bool = True,
        db_path: str = "data/ticks.db"
    ):
        """Stream live tick data."""
        if not self.connected:
            print("❌ Not connected to MT5")
            return
        
        print(f"📡 Starting live stream for {symbol}...")
        print(f"   Duration: {duration_seconds} seconds")
        
        start_time = time.time()
        ticks_collected = []
        
        while time.time() - start_time < duration_seconds:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                tick_data = {
                    'timestamp': datetime.fromtimestamp(tick.time),
                    'symbol': symbol,
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last': tick.last,
                    'volume': tick.volume,
                    'time_msc': tick.time_msc
                }
                
                ticks_collected.append(tick_data)
                
                if callback:
                    callback(tick_data)
                
                print(f"📊 {tick_data['timestamp']} | Bid: {tick.bid:.5f} | Ask: {tick.ask:.5f} | Last: {tick.last:.5f}")
            
            time.sleep(0.5)  # 500ms delay
        
        # Save to database
        if save_to_db and ticks_collected:
            df_ticks = pd.DataFrame(ticks_collected)
            self.save_to_sqlite(df_ticks.set_index('timestamp'), symbol, db_path)
        
        print(f"✅ Stream completed. Collected {len(ticks_collected)} ticks.")
        return pd.DataFrame(ticks_collected)


def load_mt5_config(config_file: str = "config/mt5_config.json") -> Dict:
    """Load MT5 configuration from file."""
    config_path = Path(config_file)
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Create default config
        default_config = {
            "terminal_path": None,
            "login": None,
            "password": None,
            "server": None,
            "timeout": 30000,
            "symbols": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"],
            "timeframe": "M1",
            "bars": 10000
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"📝 Created default config at {config_path}")
        return default_config


def collect_historical_data(
    symbols: Optional[List[str]] = None,
    timeframe: str = "M1",
    bars: int = 10000,
    config_file: str = "config/mt5_config.json"
):
    """Collect historical data from MT5."""
    
    # Load config
    config = load_mt5_config(config_file)
    
    if symbols is None:
        symbols = config.get('symbols', ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'])
    
    # Initialize collector
    collector = MT5DataCollector(
        terminal_path=config.get('terminal_path'),
        login=config.get('login'),
        password=config.get('password'),
        server=config.get('server'),
        timeout=config.get('timeout', 30000)
    )
    
    if not collector.connected:
        print("❌ Failed to connect to MT5. Please check your configuration.")
        print("   Make sure MetaTrader 5 is installed and running on Windows.")
        return None
    
    try:
        # Fetch data
        data = collector.fetch_multiple_symbols(
            symbols=symbols,
            timeframe=timeframe,
            bars=bars,
            output_dir="data/raw"
        )
        
        print(f"\n✅ Successfully collected data for {len(data)} symbols")
        return data
    
    finally:
        collector.disconnect()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Custom symbols from command line
        symbols = sys.argv[1].split(',')
        collect_historical_data(symbols=symbols)
    else:
        # Use config file
        collect_historical_data()
