"""
Full Pipeline Runner for AI Smart Trader
Orchestrates: Data Fetching -> Feature Engineering -> Labeling -> Training -> Backtesting
"""
import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use correct module paths for this project structure
from config.trading_config import get_config
from scripts.fetch_real_data import main as fetch_data
from smarttred.features.feature_engine import FeatureEngine
from smarttred.models.trainer import MLTrainer as ModelTrainer
from smarttred.backtest.vectorbt_backtest import VectorBTBacktester

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_full_pipeline(symbol: str = "EURUSD"):
    """Execute the complete ML trading pipeline."""
    config = get_config()
    
    logger.info("="*60)
    logger.info(f"STARTING FULL PIPELINE FOR {symbol}")
    logger.info("="*60)
    
    # Step 1: Fetch/Prepare Data
    logger.info("\n[STEP 1] Fetching/Preparing Data...")
    try:
        fetch_data()  # Runs in mock mode by default
        logger.info("✓ Data preparation complete")
    except Exception as e:
        logger.error(f"✗ Data fetching failed: {e}")
        return
    
    # Step 2: Feature Engineering & Labeling
    logger.info("\n[STEP 2] Generating Features and Labels...")
    try:
        input_file = config.data.raw_data_dir / f"{symbol}_{config.data.timeframe}.parquet"
        output_file = config.data.features_dir / f"{symbol}_{config.data.timeframe}_features.parquet"
        
        if not input_file.exists():
            logger.warning(f"Input file not found: {input_file}. Using synthetic data.")
            from scripts.generate_synthetic_data import generate_and_save
            generate_and_save(symbol, n_samples=10000)
            input_file = config.data.raw_data_dir / f"{symbol}_{config.data.timeframe}.parquet"
        
        # Load data and ensure correct column names for labeling
        import pandas as pd
        df = pd.read_parquet(input_file)
        
        # Rename timestamp to time for compatibility with target_generator
        if 'timestamp' in df.columns and 'time' not in df.columns:
            df['time'] = df['timestamp']
        
        # Generate features
        feature_engine = FeatureEngine()
        df_features = feature_engine.transform(df)
        
        # Apply Triple Barrier labeling
        from smarttred.features.target_generator import triple_barrier_labels
        df_labeled = triple_barrier_labels(
            df_features,
            profit_taker=config.labeling.tp_multiplier / 100.0,  # Convert to fractional return
            stop_loss=config.labeling.sl_multiplier / 100.0,
            time_limit=config.labeling.time_limit_bars,
            use_atr=config.labeling.use_dynamic_thresholds
        )
        
        # Save features with labels
        df_labeled.to_parquet(output_file, index=False)
        logger.info(f"✓ Features and labels saved to {output_file}")
    except Exception as e:
        logger.error(f"✗ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Model Training
    logger.info("\n[STEP 3] Training ML Model...")
    try:
        model_path = config.data.model_dir / f"{symbol}_model_v1.pkl"
        report_path = config.data.model_dir / f"{symbol}_report_v1.txt"
        
        trainer = MLTrainer(
            n_splits=config.model.purged_kfold_splits,
            embargo_pct=config.model.embargo_pct
        )
        
        trained_model, report = trainer.train(
            data=df_labeled,
            target_col='target',  # Using correct parameter name
            model_type='hist_gradient_boosting'  # Fallback that works well
        )
        
        # Save model and report
        import joblib
        joblib.dump(trained_model, model_path)
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"✓ Model saved to {model_path}")
        logger.info(f"✓ Report saved to {report_path}")
    except Exception as e:
        logger.error(f"✗ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Backtesting
    logger.info("\n[STEP 4] Running Backtest...")
    try:
        backtester = VectorBTBacktester(
            data_path=str(output_file),
            model_path=str(model_path),
            symbol=symbol
        )
        results = backtester.run()
        
        logger.info("✓ Backtest complete")
        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        print(results)
        print("="*60)
    except Exception as e:
        logger.error(f"✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        # Continue anyway as backtest is optional for deployment
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    logger.info(f"\nNext Steps:")
    logger.info(f"1. Review model report: {report_path}")
    logger.info(f"2. Run live trading: python scripts/live_trader.py --symbol {symbol}")
    logger.info(f"3. Monitor performance in dashboard")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run full AI trading pipeline")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Trading symbol")
    args = parser.parse_args()
    
    run_full_pipeline(args.symbol)
