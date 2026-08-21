"""
Advanced Backtesting Engine with Walk-Forward Analysis
Supports VectorBT and Pandas fallback, handles commission/slippage dynamically.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    print("⚠️ VectorBT not available, using pandas fallback")


class AdvancedBacktester:
    """Professional backtesting engine with walk-forward analysis."""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.0002,  # 0.02%
        slippage: float = 0.0001,    # 0.01%
        risk_per_trade: float = 0.02,  # 2% risk per trade
        use_vectorbt: bool = True
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_per_trade = risk_per_trade
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
        
        self.results: Optional[Dict] = None
        self.equity_curve: Optional[pd.Series] = None
        self.trades: Optional[pd.DataFrame] = None
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for backtesting."""
        data = df.copy()
        
        # Ensure required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close']
        if 'time' in data.columns and 'timestamp' not in data.columns:
            data['timestamp'] = data['time']
            
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Set timestamp as index
        if 'timestamp' in data.columns:
            data.set_index('timestamp', inplace=True)
        
        # Convert index to datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Sort by time
        data.sort_index(inplace=True)
        
        # Drop NaN values
        data.dropna(inplace=True)
        
        return data
    
    def generate_signals(self, df: pd.DataFrame, model=None, threshold: float = 0.5) -> pd.Series:
        """Generate trading signals from model predictions or existing signal column."""
        if 'signal' in df.columns:
            signals = df['signal']
        elif 'prediction' in df.columns:
            signals = (df['prediction'] > threshold).astype(int)
            signals = signals.replace(0, -1)  # 0 -> Hold/Sell, 1 -> Buy
        elif model is not None:
            # Generate predictions using model
            feature_cols = [c for c in df.columns if c not in 
                          ['timestamp', 'time', 'open', 'high', 'low', 'close', 'volume', 
                           'target', 'label', 'signal', 'prediction']]
            X = df[feature_cols].select_dtypes(include=[np.number])
            predictions = model.predict(X)
            signals = pd.Series(predictions, index=df.index)
            signals = signals.replace(0, -1)
        else:
            # Default: random signals for testing
            signals = pd.Series(np.random.choice([-1, 0, 1], size=len(df)), index=df.index)
        
        return signals
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        signals: Optional[pd.Series] = None,
        model=None,
        threshold: float = 0.5,
        title: str = "Backtest Results"
    ) -> Dict:
        """Run backtest and return comprehensive metrics."""
        
        # Prepare data
        data = self.prepare_data(df)
        
        # Generate signals if not provided
        if signals is None:
            signals = self.generate_signals(data, model, threshold)
        
        # Align signals with data
        signals = signals.reindex(data.index).fillna(0)
        
        if self.use_vectorbt:
            results = self._run_vectorbt_backtest(data, signals, title)
        else:
            results = self._run_pandas_backtest(data, signals, title)
        
        self.results = results
        self.equity_curve = results.get('equity_curve')
        
        return results
    
    def _run_vectorbt_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        title: str
    ) -> Dict:
        """Run backtest using VectorBT."""
        
        try:
            # Create portfolio using vbt Portfolio.from_signals
            price = data['close']
            
            # Convert signals to entries and exits
            entries = signals == 1
            exits = signals == -1
            
            # Handle commission and slippage
            portfolio = vbt.Portfolio.from_signals(
                close=price,
                entries=entries,
                exits=exits,
                init_cash=self.initial_capital,
                fees=self.commission,
                slippage=self.slippage,
                freq=data.index.freq or '1min'
            )
            
            # Calculate metrics
            total_return = portfolio.total_return()
            sharpe_ratio = portfolio.sharpe_ratio()
            max_drawdown = portfolio.max_drawdown()
            win_rate = portfolio.win_rate()
            profit_factor = portfolio.profit_factor()
            
            # Get equity curve
            equity_curve = portfolio.value()
            
            # Get trades
            trades_df = portfolio.trades.records_readable if hasattr(portfolio, 'trades') else None
            
            return {
                'total_return': float(total_return) if hasattr(total_return, '__float__') else float(total_return.iloc[0]) if hasattr(total_return, 'iloc') else float(total_return),
                'sharpe_ratio': float(sharpe_ratio) if hasattr(sharpe_ratio, '__float__') else float(sharpe_ratio.iloc[0]) if hasattr(sharpe_ratio, 'iloc') else float(sharpe_ratio),
                'max_drawdown': float(max_drawdown) if hasattr(max_drawdown, '__float__') else float(max_drawdown.iloc[0]) if hasattr(max_drawdown, 'iloc') else float(max_drawdown),
                'win_rate': float(win_rate) if hasattr(win_rate, '__float__') else float(win_rate.iloc[0]) if hasattr(win_rate, 'iloc') else float(win_rate),
                'profit_factor': float(profit_factor) if hasattr(profit_factor, '__float__') else float(profit_factor.iloc[0]) if hasattr(profit_factor, 'iloc') else float(profit_factor),
                'equity_curve': equity_curve,
                'trades': trades_df,
                'initial_capital': self.initial_capital,
                'final_value': float(portfolio.value().iloc[-1]),
                'title': title,
                'method': 'vectorbt'
            }
            
        except Exception as e:
            print(f"⚠️ VectorBT error: {e}, falling back to pandas")
            self.use_vectorbt = False
            return self._run_pandas_backtest(data, signals, title)
    
    def _run_pandas_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        title: str
    ) -> Dict:
        """Run backtest using pure pandas (fallback)."""
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity = []
        
        for i, (idx, row) in enumerate(data.iterrows()):
            signal = signals.loc[idx] if idx in signals.index else 0
            price = row['close']
            
            # Entry logic
            if signal == 1 and position == 0:
                # Buy
                shares = int(capital * self.risk_per_trade / price)
                if shares > 0:
                    cost = shares * price * (1 + self.commission + self.slippage)
                    if cost <= capital:
                        capital -= cost
                        position = shares
                        entry_price = price
            
            # Exit logic
            elif signal == -1 and position > 0:
                # Sell
                revenue = position * price * (1 - self.commission - self.slippage)
                profit = revenue - (position * entry_price * (1 + self.commission))
                capital += revenue
                
                trades.append({
                    'entry_time': data.index[i-1] if i > 0 else idx,
                    'exit_time': idx,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'shares': position,
                    'profit': profit,
                    'return_pct': profit / (position * entry_price) if position * entry_price > 0 else 0
                })
                
                position = 0
                entry_price = 0
            
            # Track equity
            total_value = capital + (position * price)
            equity.append(total_value)
        
        # Calculate final metrics
        equity_series = pd.Series(equity, index=data.index)
        returns = equity_series.pct_change().dropna()
        
        if len(returns) == 0 or returns.sum() == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'equity_curve': equity_series,
                'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
                'initial_capital': self.initial_capital,
                'final_value': equity_series.iloc[-1] if len(equity_series) > 0 else self.initial_capital,
                'title': title,
                'method': 'pandas_fallback'
            }
        
        # Calculate metrics
        total_return = (equity_series.iloc[-1] - self.initial_capital) / self.initial_capital
        
        if returns.std() != 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60)  # Annualized for minute data
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # Win rate and profit factor
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df['profit'] > 0]
            losing_trades = trades_df[trades_df['profit'] < 0]
            
            win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
            
            gross_profit = winning_trades['profit'].sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades['profit'].sum()) if len(losing_trades) > 0 else 0
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = 0.0
            profit_factor = 0.0
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'equity_curve': equity_series,
            'trades': trades_df,
            'initial_capital': self.initial_capital,
            'final_value': float(equity_series.iloc[-1]),
            'title': title,
            'method': 'pandas_fallback'
        }
    
    def walk_forward_analysis(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        train_ratio: float = 0.7,
        model_class=None,
        model_params: Optional[Dict] = None
    ) -> Dict:
        """Perform walk-forward analysis."""
        
        data = self.prepare_data(df)
        results = []
        
        # Split data into periods
        n_samples = len(data)
        period_size = n_samples // n_splits
        
        for i in range(n_splits - 1):
            # Define train/test split
            train_start = i * period_size
            train_end = int(train_start + period_size * train_ratio)
            test_start = train_end
            test_end = (i + 1) * period_size
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            if len(train_data) < 100 or len(test_data) < 10:
                continue
            
            # Train model if provided
            model = None
            if model_class is not None:
                try:
                    feature_cols = [c for c in train_data.columns if c not in 
                                  ['timestamp', 'time', 'open', 'high', 'low', 'close', 'volume', 
                                   'target', 'label', 'signal', 'prediction']]
                    X_train = train_data[feature_cols].select_dtypes(include=[np.number]).dropna()
                    y_train = train_data['target'].loc[X_train.index] if 'target' in train_data.columns else np.random.randint(0, 2, size=len(X_train))
                    
                    model = model_class(**(model_params or {}))
                    model.fit(X_train, y_train)
                except Exception as e:
                    print(f"⚠️ Model training failed in fold {i}: {e}")
            
            # Run backtest on test period
            backtest_result = self.run_backtest(test_data, model=model)
            backtest_result['fold'] = i
            backtest_result['train_period'] = f"{train_data.index[0]} to {train_data.index[-1]}"
            backtest_result['test_period'] = f"{test_data.index[0]} to {test_data.index[-1]}"
            
            results.append(backtest_result)
        
        # Aggregate results
        if not results:
            return {'error': 'No valid folds for walk-forward analysis'}
        
        avg_return = np.mean([r['total_return'] for r in results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in results])
        avg_win_rate = np.mean([r['win_rate'] for r in results])
        
        return {
            'folds': results,
            'average_total_return': float(avg_return),
            'average_sharpe_ratio': float(avg_sharpe),
            'average_max_drawdown': float(avg_drawdown),
            'average_win_rate': float(avg_win_rate),
            'n_folds': len(results),
            'robustness_score': float(avg_sharpe / (avg_drawdown + 0.01)) if avg_drawdown >= 0 else 0
        }
    
    def save_results(self, results: Dict, output_path: str):
        """Save backtest results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert non-serializable objects
        serializable_results = {}
        for key, value in results.items():
            if key in ['equity_curve', 'trades', 'folds']:
                if isinstance(value, pd.Series):
                    serializable_results[key] = value.to_dict()
                elif isinstance(value, pd.DataFrame):
                    serializable_results[key] = value.to_dict()
                elif isinstance(value, list):
                    serializable_results[key] = [
                        {k: (v.to_dict() if isinstance(v, (pd.Series, pd.DataFrame)) else v) 
                         for k, v in item.items()}
                        for item in value
                    ]
                else:
                    serializable_results[key] = value
            else:
                serializable_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"✅ Results saved to {output_path}")
    
    def print_summary(self, results: Optional[Dict] = None):
        """Print formatted backtest summary."""
        if results is None:
            results = self.results
        
        if results is None:
            print("❌ No backtest results available")
            return
        
        print("\n" + "="*60)
        print(f"📊 {results.get('title', 'BACKTEST SUMMARY')}")
        print("="*60)
        print(f"💰 Initial Capital:    ${results.get('initial_capital', 0):,.2f}")
        print(f"📈 Final Value:        ${results.get('final_value', 0):,.2f}")
        print(f"📊 Total Return:       {results.get('total_return', 0)*100:.2f}%")
        print(f"📉 Max Drawdown:       {results.get('max_drawdown', 0)*100:.2f}%")
        print(f"⚡ Sharpe Ratio:       {results.get('sharpe_ratio', 0):.3f}")
        print(f"🎯 Win Rate:           {results.get('win_rate', 0)*100:.2f}%")
        print(f"💹 Profit Factor:      {results.get('profit_factor', 0):.2f}")
        print(f"🔧 Method:             {results.get('method', 'unknown')}")
        
        if 'robustness_score' in results:
            print(f"🛡️ Robustness Score:   {results['robustness_score']:.3f}")
        
        print("="*60 + "\n")


def run_backtest_from_file(
    feature_file: str,
    output_file: str = "data/backtest/results.json",
    initial_capital: float = 10000.0,
    commission: float = 0.0002,
    slippage: float = 0.0001,
    risk_per_trade: float = 0.02,
    do_walk_forward: bool = False
):
    """Run backtest from feature file."""
    
    # Load data
    feature_path = Path(feature_file)
    if not feature_path.exists():
        print(f"❌ Feature file not found: {feature_path}")
        return None
    
    print(f"📂 Loading data from {feature_path}...")
    df = pd.read_parquet(feature_path) if feature_path.suffix == '.parquet' else pd.read_csv(feature_path)
    
    # Initialize backtester
    backtester = AdvancedBacktester(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
        risk_per_trade=risk_per_trade
    )
    
    # Run simple backtest
    print("🚀 Running backtest...")
    results = backtester.run_backtest(df)
    backtester.print_summary(results)
    
    # Run walk-forward analysis if requested
    if do_walk_forward:
        print("🔄 Running walk-forward analysis...")
        try:
            from lightgbm import LGBMClassifier
            wf_results = backtester.walk_forward_analysis(
                df,
                n_splits=5,
                train_ratio=0.7,
                model_class=LGBMClassifier,
                model_params={'n_estimators': 100, 'random_state': 42}
            )
            backtester.print_summary(wf_results)
            results['walk_forward'] = wf_results
        except Exception as e:
            print(f"⚠️ Walk-forward analysis failed: {e}")
    
    # Save results
    backtester.save_results(results, output_file)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        feature_file = sys.argv[1]
    else:
        # Default: look for latest feature file
        feature_dir = Path("data/features")
        feature_files = list(feature_dir.glob("*.parquet")) + list(feature_dir.glob("*.csv"))
        if feature_files:
            feature_file = str(feature_files[0])
        else:
            print("❌ No feature files found. Please specify a file.")
            sys.exit(1)
    
    results = run_backtest_from_file(
        feature_file,
        do_walk_forward=True
    )
