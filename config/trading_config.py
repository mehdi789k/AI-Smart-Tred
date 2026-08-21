"""
Configuration Manager for AI Smart Trader
Centralized configuration for Feature Engineering, Labeling, and Trading parameters.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent

@dataclass
class DataConfig:
    """Data ingestion and storage settings."""
    symbols: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"])
    timeframe: str = "M15"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    features_dir: Path = PROJECT_ROOT / "data" / "features"
    model_dir: Path = PROJECT_ROOT / "data" / "model_releases"
    
    def __post_init__(self):
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class FeatureConfig:
    """Feature engineering parameters."""
    lookback_windows: List[int] = field(default_factory=lambda: [14, 20, 50])
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    frac_diff_order: float = 0.5
    use_volatility_features: bool = True
    use_volume_features: bool = True

@dataclass
class LabelConfig:
    """Triple Barrier Method and Target Labeling settings."""
    # Triple Barrier Parameters (in bars)
    profit_target_bars: int = 50      # Take Profit horizon
    stop_loss_bars: int = 50          # Stop Loss horizon
    time_limit_bars: int = 100        # Maximum holding period
    
    # Thresholds (in ATR multiples or fixed points)
    tp_multiplier: float = 1.5        # TP = Entry + (ATR * 1.5)
    sl_multiplier: float = 1.0        # SL = Entry - (ATR * 1.0)
    
    # For fixed point targets (if not using ATR)
    fixed_tp_points: float = 0.0050   # 50 pips for EURUSD
    fixed_sl_points: float = 0.0025   # 25 pips for EURUSD
    
    use_dynamic_thresholds: bool = True  # If True, uses ATR multipliers; else uses fixed points
    min_sample_weight: float = 0.1    # Minimum weight for sample weighting

@dataclass
class ModelConfig:
    """Machine Learning model training settings."""
    test_size: float = 0.2
    purged_kfold_splits: int = 5
    embargo_pct: float = 0.05  # 5% embargo to prevent leakage
    class_weight_strategy: str = "balanced"  # Options: balanced, scale, None
    
    # LightGBM Parameters
    lgbm_params: Dict = field(default_factory=lambda: {
        "objective": "multiclass",
        "num_class": 3,  # Buy, Hold, Sell
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1
    })

@dataclass
class BacktestConfig:
    """Backtesting engine settings."""
    initial_capital: float = 10000.0
    commission_pct: float = 0.0001  # 0.01% per trade
    slippage_points: float = 0.5    # Average slippage in points
    risk_per_trade: float = 0.01    # 1% risk per trade
    max_positions: int = 1          # Max concurrent positions
    
    # Position Sizing
    use_kelly_criterion: bool = False
    kelly_fraction: float = 0.25    # Fractional Kelly (25%)

@dataclass
class LiveTradingConfig:
    """Live/Paper Trading settings."""
    mode: str = "paper"  # Options: paper, live
    mt5_login: int = int(os.getenv("MT5_LOGIN", "0"))
    mt5_password: str = os.getenv("MT5_PASSWORD", "")
    mt5_server: str = os.getenv("MT5_SERVER", "")
    mt5_path: str = os.getenv("MT5_TERMINAL_PATH", "")
    
    # Risk Controls
    max_daily_loss_pct: float = 3.0   # Stop trading if daily loss > 3%
    max_drawdown_pct: float = 10.0    # Stop trading if drawdown > 10%
    news_filter_enabled: bool = True  # Skip trades during high-impact news
    
    # Execution
    order_retry_attempts: int = 3
    order_timeout_seconds: int = 30

# Global Configuration Instance
config = type('GlobalConfig', (), {
    'data': DataConfig(),
    'features': FeatureConfig(),
    'labeling': LabelConfig(),
    'model': ModelConfig(),
    'backtest': BacktestConfig(),
    'live': LiveTradingConfig()
})

def get_config():
    return config
