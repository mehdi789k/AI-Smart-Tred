"""
Live Trading Executor for AI Smart Trader
Supports Paper Trading and Live Execution via MetaTrader5
"""
import sys
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.trading_config import get_config
from src.data_collector.mt5_connector import MT5Connector
from src.feature_engineering.feature_engine import FeatureEngine
from src.ml_trainer.model_loader import load_model
from src.risk_manager.position_sizer import calculate_position_size

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveTrader:
    def __init__(self, symbol: str, config):
        self.symbol = symbol
        self.config = config
        self.mode = config.live.mode
        self.connector = None
        self.feature_engine = FeatureEngine()
        self.model = None
        self.is_running = False
        
        # Risk management state
        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        self.positions_open = 0
        
        logger.info(f"Initialized LiveTrader for {symbol} in {self.mode.upper()} mode")
    
    def initialize(self):
        """Initialize connections and load model."""
        # Load trained model
        model_path = self.config.data.model_dir / f"{self.symbol}_model_v1.pkl"
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}. Run training first.")
            return False
        
        self.model = load_model(str(model_path))
        logger.info(f"✓ Model loaded from {model_path}")
        
        # Connect to MT5 if in live/paper mode
        if self.mode in ["live", "paper"]:
            self.connector = MT5Connector()
            if not self.connector.connect():
                logger.error("Failed to connect to MT5")
                return False
            logger.info("✓ Connected to MT5")
        
        return True
    
    def get_latest_data(self, n_bars: int = 100) -> Optional[Dict]:
        """Fetch latest market data."""
        if self.mode == "paper" and self.connector is None:
            # Mock data for paper trading without MT5
            import pandas as pd
            import numpy as np
            dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='15min')
            np.random.seed(int(time.time()))
            walk = np.cumsum(np.random.randn(n_bars))
            price = 1.1000 + walk * 0.0001
            return {
                'timestamp': dates,
                'open': price + np.random.randn(n_bars)*0.00005,
                'high': price + np.abs(np.random.randn(n_bars))*0.0001,
                'low': price - np.abs(np.random.randn(n_bars))*0.00005,
                'close': price + np.random.randn(n_bars)*0.00005,
                'volume': np.random.randint(100, 1000, n_bars)
            }
        elif self.connector:
            return self.connector.fetch_rates(self.symbol, n_bars)
        else:
            logger.warning("No data source available")
            return None
    
    def predict_signal(self, data: Dict) -> Optional[int]:
        """Generate trading signal from model."""
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            
            # Generate features
            df_features = self.feature_engine.transform(df)
            
            # Get latest feature vector
            latest_features = df_features.iloc[-1:].dropna(axis=1)
            
            if latest_features.empty:
                logger.warning("No valid features for prediction")
                return None
            
            # Predict
            prediction = self.model.predict(latest_features)[0]
            logger.info(f"Signal generated: {prediction} (0=Sell, 1=Hold, 2=Buy)")
            return prediction
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def execute_trade(self, signal: int):
        """Execute trade based on signal."""
        if signal == 1:  # Hold
            return
        
        # Check risk limits
        if self.daily_pnl < -self.config.live.max_daily_loss_pct:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl}%")
            return
        
        if self.current_drawdown > self.config.live.max_drawdown_pct:
            logger.warning(f"Max drawdown reached: {self.current_drawdown}%")
            return
        
        if self.positions_open >= self.config.backtest.max_positions:
            logger.warning("Max positions limit reached")
            return
        
        # Calculate position size
        lot_size = calculate_position_size(
            account_balance=10000,  # Mock balance
            risk_pct=self.config.backtest.risk_per_trade,
            stop_loss_points=25,  # Example SL
            symbol=self.symbol
        )
        
        order_type = "BUY" if signal == 2 else "SELL"
        
        if self.mode == "live":
            # Real order execution
            if self.connector:
                success = self.connector.place_order(
                    symbol=self.symbol,
                    order_type=order_type,
                    lots=lot_size
                )
                if success:
                    self.positions_open += 1
                    logger.info(f"✓ LIVE ORDER EXECUTED: {order_type} {lot_size} lots on {self.symbol}")
                else:
                    logger.error("Order execution failed")
        else:
            # Paper trading simulation
            logger.info(f"📄 PAPER TRADE: {order_type} {lot_size:.2f} lots on {self.symbol} @ {datetime.now()}")
            self.positions_open += 1
    
    def run(self, interval_seconds: int = 60):
        """Main trading loop."""
        logger.info(f"Starting trading loop (interval: {interval_seconds}s)...")
        self.is_running = True
        
        while self.is_running:
            try:
                # Fetch data
                data = self.get_latest_data()
                if data is None:
                    time.sleep(interval_seconds)
                    continue
                
                # Generate signal
                signal = self.predict_signal(data)
                
                # Execute trade
                if signal is not None:
                    self.execute_trade(signal)
                
                # Update risk metrics (mock implementation)
                # In real implementation, fetch open positions and calculate PnL
                
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Trading stopped by user")
                self.stop()
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(interval_seconds)
    
    def stop(self):
        """Stop the trading loop."""
        self.is_running = False
        logger.info("Trading loop stopped")
        if self.connector:
            self.connector.disconnect()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live Trading Executor")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Trading symbol")
    parser.add_argument("--mode", type=str, default="paper", choices=["paper", "live"], help="Trading mode")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    args = parser.parse_args()
    
    config = get_config()
    config.live.mode = args.mode
    
    trader = LiveTrader(args.symbol, config)
    
    if trader.initialize():
        trader.run(interval_seconds=args.interval)
    else:
        logger.error("Failed to initialize trader")

if __name__ == "__main__":
    main()
