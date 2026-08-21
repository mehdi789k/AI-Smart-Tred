"""
Live Trading Engine for AI Smart Trader
Executes trades based on model signals with risk management
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from pathlib import Path
import json
import time

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


class LiveTradingEngine:
    """Professional live trading engine with risk management."""
    
    def __init__(
        self,
        model_path: str,
        symbol: str = "EURUSD",
        timeframe: str = "M1",
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.02,  # 2% risk per trade
        max_positions: int = 1,
        use_mt5: bool = True
    ):
        self.model_path = Path(model_path)
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.use_mt5 = use_mt5 and MT5_AVAILABLE
        
        self.model = None
        self.position = None
        self.trades_log = []
        self.connected = False
        
        # Load model
        self.load_model()
        
        # Connect to MT5 if enabled
        if self.use_mt5:
            self.connect_mt5()
    
    def load_model(self):
        """Load trained ML model."""
        try:
            import joblib
            self.model = joblib.load(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
            self.model = None
    
    def connect_mt5(self) -> bool:
        """Connect to MetaTrader 5."""
        if not MT5_AVAILABLE:
            print("❌ MetaTrader5 not available")
            return False
        
        try:
            if mt5.initialize():
                self.connected = True
                account_info = mt5.account_info()
                if account_info:
                    print(f"✅ Connected to MT5 - Account: {account_info.login}")
                return True
            else:
                print(f"❌ MT5 initialization failed: {mt5.last_error()}")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def get_current_price(self) -> Optional[Dict]:
        """Get current market price."""
        if not self.connected:
            return None
        
        tick = mt5.symbol_info_tick(self.symbol)
        if tick:
            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'time': datetime.fromtimestamp(tick.time),
                'volume': tick.volume
            }
        return None
    
    def get_historical_data(self, bars: int = 100) -> Optional[pd.DataFrame]:
        """Get historical data for feature calculation."""
        if not self.connected:
            return None
        
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        tf = timeframe_map.get(self.timeframe, mt5.TIMEFRAME_M1)
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, bars)
        
        if rates is None or len(rates) == 0:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'
        }, inplace=True)
        
        return df[['time', 'open', 'high', 'low', 'close', 'volume']]
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for model prediction."""
        # Import feature engine
        try:
            from src.feature_engine import FeatureEngine
            
            engine = FeatureEngine(
                window_sizes=[5, 10, 20],
                add_statistical_features=True,
                frac_diff_order=0.5
            )
            
            features_df = engine.transform(df)
            return features_df.iloc[-1:]  # Last row for prediction
            
        except Exception as e:
            print(f"⚠️ Feature calculation failed: {e}")
            # Fallback: use raw OHLCV
            return df.iloc[-1:].copy()
    
    def predict_signal(self, features: pd.DataFrame) -> int:
        """Generate trading signal from model."""
        if self.model is None:
            return 0  # No signal
        
        try:
            # Select numeric features
            feature_cols = [c for c in features.columns 
                          if c not in ['time', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
            X = features[feature_cols].select_dtypes(include=[np.number])
            
            if X.empty:
                return 0
            
            # Predict
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0] if hasattr(self.model, 'predict_proba') else None
            
            # Convert to signal: 1=Buy, -1=Sell, 0=Hold
            if isinstance(prediction, (int, np.integer)):
                signal = 1 if prediction > 0 else (-1 if prediction < 0 else 0)
            else:
                signal = 1 if prediction > 0.5 else (-1 if prediction < 0.5 else 0)
            
            print(f"📊 Prediction: {prediction:.4f}, Signal: {signal}, Prob: {probability}")
            return signal
            
        except Exception as e:
            print(f"⚠️ Prediction failed: {e}")
            return 0
    
    def calculate_position_size(self, price: float, stop_loss_pips: float = 20) -> int:
        """Calculate position size based on risk management."""
        # Risk amount in account currency
        risk_amount = self.initial_capital * self.risk_per_trade
        
        # Pip value (approximate for EURUSD)
        pip_value = 0.0001 * 100000  # Standard lot
        
        # Position size in lots
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Round to nearest 0.01 lot
        position_size = round(position_size * 100) / 100
        
        # Convert to units
        units = int(position_size * 100000)
        
        return max(units, 1000)  # Minimum 1000 units
    
    def execute_trade(
        self,
        signal: int,
        price: float,
        stop_loss_pips: float = 20,
        take_profit_pips: float = 40
    ) -> Optional[Dict]:
        """Execute trade based on signal."""
        if not self.connected:
            print("❌ Not connected to MT5")
            return None
        
        if signal == 0:
            print("ℹ️ No trade signal (Hold)")
            return None
        
        # Check existing positions
        positions = mt5.positions_get(symbol=self.symbol)
        if positions and len(positions) >= self.max_positions:
            print("⚠️ Maximum positions reached")
            return None
        
        # Calculate position size
        position_size = self.calculate_position_size(price, stop_loss_pips)
        
        # Calculate SL/TP levels
        sl_distance = stop_loss_pips * 0.0001
        tp_distance = take_profit_pips * 0.0001
        
        if signal == 1:  # Buy
            order_type = mt5.ORDER_TYPE_BUY
            sl_price = price - sl_distance
            tp_price = price + tp_distance
        else:  # Sell
            order_type = mt5.ORDER_TYPE_SELL
            sl_price = price + sl_distance
            tp_price = price - tp_distance
        
        # Prepare order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position_size / 100000,  # Convert to lots
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 10,
            "magic": 234000,
            "comment": "AI Smart Trader",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Order failed: {result.comment}")
            return None
        
        trade_record = {
            'time': datetime.now(),
            'symbol': self.symbol,
            'type': 'BUY' if signal == 1 else 'SELL',
            'price': price,
            'volume': position_size,
            'sl': sl_price,
            'tp': tp_price,
            'order_id': result.order,
            'signal': signal
        }
        
        self.trades_log.append(trade_record)
        print(f"✅ Trade executed: {trade_record['type']} {position_size} units @ {price:.5f}")
        print(f"   SL: {sl_price:.5f}, TP: {tp_price:.5f}")
        
        return trade_record
    
    def run_trading_loop(
        self,
        duration_minutes: int = 60,
        check_interval_seconds: int = 60
    ):
        """Run live trading loop."""
        if not self.connected:
            print("❌ Cannot start trading: Not connected to MT5")
            return
        
        print(f"🚀 Starting live trading for {self.symbol}")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Check interval: {check_interval_seconds} seconds")
        
        start_time = time.time()
        duration_seconds = duration_minutes * 60
        
        while time.time() - start_time < duration_seconds:
            try:
                # Get current price
                price_data = self.get_current_price()
                if not price_data:
                    print("⚠️ Unable to get price, waiting...")
                    time.sleep(check_interval_seconds)
                    continue
                
                current_price = price_data['ask']  # Use ask for entry
                
                # Get historical data
                hist_data = self.get_historical_data(bars=100)
                if hist_data is None or len(hist_data) < 20:
                    print("⚠️ Insufficient historical data")
                    time.sleep(check_interval_seconds)
                    continue
                
                # Calculate features
                features = self.calculate_features(hist_data)
                
                # Generate signal
                signal = self.predict_signal(features)
                
                # Execute trade if signal
                if signal != 0:
                    self.execute_trade(signal, current_price)
                
                # Wait for next check
                time.sleep(check_interval_seconds)
                
            except KeyboardInterrupt:
                print("\n⛔ Trading stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in trading loop: {e}")
                time.sleep(check_interval_seconds)
        
        print("✅ Trading session completed")
        self.save_trades_log()
    
    def save_trades_log(self, output_path: str = "data/live_trades.json"):
        """Save trades log to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert datetime objects to strings
        serializable_log = []
        for trade in self.trades_log:
            trade_copy = trade.copy()
            if 'time' in trade_copy and isinstance(trade_copy['time'], datetime):
                trade_copy['time'] = trade_copy['time'].isoformat()
            serializable_log.append(trade_copy)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_log, f, indent=2)
        
        print(f"💾 Trades log saved to {output_file}")
    
    def close_all_positions(self):
        """Close all open positions."""
        if not self.connected:
            return
        
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            print("ℹ️ No open positions")
            return
        
        for pos in positions:
            # Prepare close order
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": pos.volume,
                "type": order_type,
                "position": pos.ticket,
                "price": mt5.symbol_info_tick(self.symbol).ask if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(self.symbol).bid,
                "deviation": 10,
                "magic": 234000,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Position closed: {pos.ticket}")
            else:
                print(f"❌ Failed to close position {pos.ticket}: {result.comment}")


def run_live_trading(
    model_path: str,
    symbol: str = "EURUSD",
    timeframe: str = "M1",
    duration_minutes: int = 60,
    check_interval: int = 60
):
    """Run live trading session."""
    
    engine = LiveTradingEngine(
        model_path=model_path,
        symbol=symbol,
        timeframe=timeframe
    )
    
    if not engine.connected:
        print("❌ Failed to connect to MT5. Exiting.")
        return
    
    try:
        engine.run_trading_loop(
            duration_minutes=duration_minutes,
            check_interval_seconds=check_interval
        )
    finally:
        engine.close_all_positions()
        if engine.connected:
            mt5.shutdown()


if __name__ == "__main__":
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "data/model_releases/best_model.joblib"
    symbol = sys.argv[2] if len(sys.argv) > 2 else "EURUSD"
    
    run_live_trading(model_path=model_path, symbol=symbol)
