"""CLI برای اجرای backtest برداری با پیش‌بینی‌های مدل ML

نحوهٔ اجرا (نمونه):
  python scripts\train_ml_model.py --input data/labeled/test_set.parquet --model data/model_releases/model_v1.pkl --features data/model_releases/model_v1_features.json --threshold 0.6

نیازمندی‌ها:
  pip install pandas numpy joblib vectorbt[full] plotly

خروجی‌ها:
  - چاپ خلاصهٔ معیارها
  - ذخیرهٔ گزارش HTML در data/trades/backtest_report.html
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from smarttred.backtest.ml_backtester import MLBacktester


def format_metrics(metrics: dict) -> str:
    lines = []
    lines.append("Backtest Summary:\n")
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:20s}: {v:0.4f}")
        else:
            lines.append(f"  {k:20s}: {v}")
    return "\n".join(lines)


def main(input_path: str, model_path: str, feature_path: str, threshold: float = 0.6, position_sizing: dict | None = None) -> None:
    print(f"Loading data from: {input_path}")
    df = pd.read_parquet(input_path)

    if "close" not in df.columns:
        raise ValueError("Input data must contain 'close' column for price series")

    backtester = MLBacktester()
    model, feature_names = backtester.load_model_and_features(model_path, feature_path)

    print("Generating signals from model predictions...")
    signals = backtester.generate_signals(df, model, feature_names, prob_threshold=threshold)

    print("Running vectorized backtest (this may take a few seconds)...")
    pf, metrics = backtester.run_backtest(
        df,
        signals,
        initial_capital=10000,
        commission=0.0005,
        slippage=0.0002,
        position_sizing=position_sizing,
    )

    print('\n' + format_metrics(metrics) + '\n')

    out_dir = os.path.join(os.getcwd(), "data", "trades")
    os.makedirs(out_dir, exist_ok=True)

    trade_log_path = os.path.join(out_dir, "trade_log.csv")
    trade_df = backtester.save_trade_log(pf, trade_log_path)
    print(f"Trade log saved to: {trade_log_path} ({len(trade_df)} trades)")

    report_path = os.path.join(out_dir, "backtest_report.html")
    try:
        fig = pf.plot()
        fig.write_html(report_path, include_plotlyjs='cdn')
        print(f"Interactive backtest report saved to: {report_path}")
    except Exception as e:
        print("Warning: failed to create interactive HTML report:", e)

    try:
        extra = backtester.save_additional_reports(pf, out_dir)
        for name, path in extra.items():
            print(f"Saved {name}: {path}")
    except Exception as e:
        print("Warning: failed to create additional plots:", e)

    print("Backtest finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run vectorized backtest using a trained ML model')
    parser.add_argument('--input', required=True, help='Path to test/unseen parquet file (must contain close and feature columns)')
    parser.add_argument('--model', required=True, help='Path to saved model (joblib .pkl)')
    parser.add_argument('--features', required=True, help='Path to saved feature names JSON file')
    parser.add_argument('--threshold', type=float, default=0.6, help='Probability threshold for long/short signals (default 0.6)')

    # Position sizing options
    parser.add_argument('--sizing-method', choices=['fractional', 'fixed_risk', 'atr'], default=None, help='Position sizing method to apply')
    parser.add_argument('--fraction', type=float, default=0.02, help='For fractional sizing: fraction of initial capital per trade (e.g., 0.02)')
    parser.add_argument('--risk-amount', type=float, default=100.0, help='For fixed_risk/atr: risk amount in account currency per trade')
    parser.add_argument('--stop-pct', type=float, default=0.01, help='For fixed_risk: stop distance as percentage of price (e.g., 0.01)')
    parser.add_argument('--atr-multiplier', type=float, default=3.0, help='For atr sizing: multiplier for ATR to set stop distance')
    parser.add_argument('--atr-window', type=int, default=14, help='For atr sizing: ATR lookback window')

    args = parser.parse_args()

    position_sizing = None
    if args.sizing_method is not None:
        position_sizing = {'method': args.sizing_method}
        if args.sizing_method == 'fractional':
            position_sizing['fraction'] = args.fraction
        elif args.sizing_method == 'fixed_risk':
            position_sizing['risk_amount'] = args.risk_amount
            position_sizing['stop_pct'] = args.stop_pct
        elif args.sizing_method == 'atr':
            position_sizing['risk_amount'] = args.risk_amount
            position_sizing['atr_multiplier'] = args.atr_multiplier
            position_sizing['atr_window'] = args.atr_window

    main(args.input, args.model, args.features, args.threshold, position_sizing)
