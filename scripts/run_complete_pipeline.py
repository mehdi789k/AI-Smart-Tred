"""Complete ML Pipeline Script for Production Use.
Generates features, labels, trains model, and runs backtest in one flow."""
import os
import sys
import json
import joblib
import argparse
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from smarttred.features.feature_engine import FeatureEngine
from smarttred.models.trainer import MLTrainer
from smarttred.backtest.ml_backtester import MLBacktester
from smarttred.backtest.vectorbt_backtest import VectorBTBacktester
from smarttred.features.target_generator import triple_barrier_labels


def run_pipeline(
    input_path: str,
    output_dir: str = "data/model_releases",
    target_horizon: int = 10,
    n_splits: int = 5,
    embargo_pct: float = 0.05,
    run_backtest: bool = True,
):
    """Run complete ML pipeline from raw data to trained model."""
    
    print("=" * 60)
    print("AI SMART TRADER - COMPLETE ML PIPELINE")
    print("=" * 60)
    
    # Step 1: Load raw data
    print(f"\n[1/5] Loading raw data from: {input_path}")
    df_raw = pd.read_parquet(input_path)
    print(f"Loaded {len(df_raw)} bars with columns: {df_raw.columns.tolist()[:8]}...")
    
    # Step 2: Feature Engineering
    print(f"\n[2/5] Generating technical indicators and statistical features...")
    feature_engine = FeatureEngine()
    df_features = feature_engine.add_technical_indicators(df_raw)
    df_features = feature_engine.compute_statistical_features(df_features)
    print(f"Generated {len(df_features.columns)} features")
    
    # Step 3: Label Generation
    print(f"\n[3/5] Generating triple-barrier labels (horizon={target_horizon})...")
    
    # Generate labels using the function-based API
    df_labeled = triple_barrier_labels(
        df_features,
        profit_taker=0.01,
        stop_loss=0.005,
        time_limit=target_horizon,
        use_atr=True,
    )
    
    # Create unified target column for training
    if "target" not in df_labeled.columns:
        raise ValueError(f"Target column 'target' not found. Available: {df_labeled.columns.tolist()}")
    
    print(f"Label distribution:")
    print(df_labeled['target'].value_counts().to_dict())
    
    # Step 4: Model Training
    print(f"\n[4/5] Training LightGBM model with Purged K-Fold (splits={n_splits}, embargo={embargo_pct})...")
    trainer = MLTrainer(n_splits=n_splits, embargo_pct=embargo_pct)
    
    X, y, feature_names = trainer.prepare_data(df_labeled, target_col='target')
    print(f"Training with {X.shape[0]} samples and {X.shape[1]} features")
    
    model = trainer.train_and_evaluate(X, y, feature_names)
    
    # Save model and features
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model_v1.pkl")
    features_path = os.path.join(output_dir, "model_v1_features.json")
    
    joblib.dump(model, model_path)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Feature names saved to: {features_path}")
    
    # Step 5: Backtesting
    if run_backtest:
        print(f"\n[5/5] Running vectorized backtest...")
        
        # Generate signals from model predictions
        pred_probs = model.predict_proba(X)[:, 1]
        df_signals = df_labeled.iloc[X.index].copy()
        df_signals['signal_prob'] = pred_probs
        
        # Convert probabilities to trading signals (-1, 0, 1)
        threshold = 0.6
        df_signals['signal'] = 0
        df_signals.loc[pred_probs >= threshold, 'signal'] = 1
        df_signals.loc[pred_probs <= (1 - threshold), 'signal'] = -1
        
        print(f"Signal distribution: {df_signals['signal'].value_counts().to_dict()}")
        
        # Run backtest
        backtester = VectorBTBacktester(
            initial_balance=10000.0,
            commission_per_trade=0.0005,
            slippage_bps=5.0,
        )
        
        backtest_results = backtester.run(df_signals, signal_col='signal')
        
        # Calculate performance metrics
        final_balance = backtest_results['balance'].iloc[-1]
        total_return = (final_balance - 10000.0) / 10000.0 * 100
        
        print(f"\n{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Initial Balance: $10,000.00")
        print(f"Final Balance: ${final_balance:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Number of Trades: {(df_signals['signal'].diff() != 0).sum()}")
        print(f"{'='*60}")
        
        # Save backtest results
        backtest_path = os.path.join(output_dir, "backtest_results.csv")
        backtest_results.to_csv(backtest_path, index=False)
        print(f"Backtest results saved to: {backtest_path}")
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Model ready for live trading at: {model_path}")
    print(f"To run live execution:")
    print(f"  python scripts/live_execution_engine.py --symbol EURUSD --live")
    print(f"{'='*60}\n")
    
    return {
        'model_path': model_path,
        'features_path': features_path,
        'final_balance': final_balance if run_backtest else None,
        'total_return': total_return if run_backtest else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete ML Pipeline for AI Smart Trader")
    parser.add_argument("--input", required=True, help="Path to raw OHLCV parquet file")
    parser.add_argument("--output-dir", default="data/model_releases", help="Output directory for model artifacts")
    parser.add_argument("--target-horizon", type=int, default=10, help="Target horizon for labeling (1, 5, 10, 20)")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV splits")
    parser.add_argument("--embargo-pct", type=float, default=0.05, help="Embargo percentage for purged CV")
    parser.add_argument("--no-backtest", action="store_true", help="Skip backtesting step")
    
    args = parser.parse_args()
    
    results = run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        target_horizon=args.target_horizon,
        n_splits=args.n_splits,
        embargo_pct=args.embargo_pct,
        run_backtest=not args.no_backtest,
    )
