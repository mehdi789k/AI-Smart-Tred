"""smarttred.backtest.ml_backtester

این ماژول کلاس MLBacktester را پیاده‌سازی می‌کند که مسئول بارگذاری مدل آموزش‌دیده، تولید سیگنال
بر اساس احتمال پیش‌بینی‌شدهٔ کلاس مثبت و اجرای backtest برداری با استفاده از vectorbt است.

توضیحات (فارسی):
- بارگذاری مدل و feature_names از فایل‌های ذخیره‌شده
- تولید سیگنال‌های Long/Short/Hold بر اساس آستانهٔ احتمال
- اجرای backtest با اعمال کمیسیون و slippage و محاسبهٔ معیارهای کلیدی

نکته: برای اجرای run_backtest نیاز است که کتابخانهٔ vectorbt و وابستگی‌هایش نصب شده باشند
(پیشنهاد: pip install "vectorbt[full]").
"""
from __future__ import annotations

from typing import List, Tuple
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import joblib

try:
    import vectorbt as vbt
except Exception:  # pragma: no cover - graceful fallback if vectorbt not installed
    vbt = None  # type: ignore


class MLBacktester:
    """کلاس برای اجرای backtest برداری با پیش‌بینی‌های مدل ML.

    متدها:
    - load_model_and_features(model_path, feature_path)
    - generate_signals(df, model, feature_names, prob_threshold=0.60)
    - run_backtest(df, signals, initial_capital=10000, commission=0.0005, slippage=0.0002)
    """

    def __init__(self) -> None:
        self.model = None
        self.feature_names: List[str] = []

    def load_model_and_features(self, model_path: str, feature_path: str) -> Tuple[object, List[str]]:
        """بارگذاری مدل ذخیره‌شده و لیست فیچرها.

        ورودی:
        - model_path: مسیر فایل مدل (joblib .pkl یا lightgbm native)
        - feature_path: مسیر فایل JSON که لیست feature_names را دارد

        بازگشتی: (model, feature_names)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        # تلاش برای بارگذاری مدل: ابتدا joblib (معمول برای sklearn/lightgbm wrapper)
        model = None
        load_error = None
        try:
            model = joblib.load(model_path)
        except Exception as e:
            load_error = e

        # اگر joblib موفق نبود یا مدل نهایی از نوع native ذخیره شده باشد، سعی می‌کنیم فرمت‌های native را بارگذاری کنیم
        if model is None:
            # تلاش برای بارگذاری native LightGBM
            try:
                import lightgbm as lgb

                # lgb.Booster از model_file پشتیبانی می‌کند برای txt/json
                booster = lgb.Booster(model_file=model_path)

                class _LGBMNativeWrapper:
                    def __init__(self, booster):
                        self.booster = booster
                        # فرض اینکه مسئله دودویی است مگر اینکه خروجی نشان دهد
                        self.classes_ = np.array([0, 1])

                    def predict_proba(self, X):
                        data = X.values if isinstance(X, pd.DataFrame) else X
                        proba = self.booster.predict(data)
                        proba = np.array(proba)
                        if proba.ndim == 1:
                            return np.vstack([1 - proba, proba]).T
                        return proba

                    def predict(self, X):
                        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

                model = _LGBMNativeWrapper(booster)
            except Exception:
                # تلاش برای بارگذاری XGBoost native
                try:
                    import xgboost as xgb
                    bst = xgb.Booster()
                    bst.load_model(model_path)

                    class _XGBWrapper:
                        def __init__(self, booster):
                            self.booster = booster
                            self.classes_ = np.array([0, 1])

                        def predict_proba(self, X):
                            import xgboost as xgb
                            dmat = xgb.DMatrix(X.values if isinstance(X, pd.DataFrame) else X)
                            proba = self.booster.predict(dmat)
                            proba = np.array(proba)
                            if proba.ndim == 1:
                                return np.vstack([1 - proba, proba]).T
                            return proba

                        def predict(self, X):
                            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

                    model = _XGBWrapper(bst)
                except Exception as e2:
                    # اگر هیچکدام موفق نبودند، گزارش خطای اولیه را بالا می‌بریم
                    raise RuntimeError(f"Failed to load model. joblib error: {load_error}; lgb/xgb errors: {e2}")

        # بارگذاری لیست فیچرها
        with open(feature_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        if not isinstance(feature_names, list):
            raise ValueError("feature file must contain a JSON list of feature names")

        self.model = model
        self.feature_names = feature_names
        return model, feature_names

    def _align_features(self, df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """اطمینان از تطابق دقیق فیچرها با مدل.

        - ستون‌های اضافی حذف می‌شوند
        - اگر فیچری کم باشد، آن را با 0 پر می‌کنیم و هشدار می‌دهیم (راه حل دیگر: خطا دادن)
        """
        df_copy = df.copy()
        missing = [f for f in feature_names if f not in df_copy.columns]
        if missing:
            warnings.warn(f"Missing features in input data: {missing}. Filling missing features with 0.")
            for m in missing:
                df_copy[m] = 0.0

        # فقط فیچرهای مورد نیاز را بر می‌گردانیم (ترتیب مطابق feature_names)
        aligned = df_copy.loc[:, feature_names]
        return aligned

    def generate_signals(self, df: pd.DataFrame, model: object, feature_names: List[str], prob_threshold: float = 0.60) -> pd.Series:
        """تولید سیگنال‌ها بر اساس احتمال پیش‌بینی‌شده برای کلاس مثبت.

        قواعد:
        - اگر prob >= prob_threshold => signal = 1 (Long)
        - اگر prob <= (1 - prob_threshold) => signal = -1 (Short)
        - در غیر این صورت => signal = 0 (Hold)

        خروجی: pd.Series از سیگنال‌ها با اندیس زمانی مطابق df
        """
        if self.model is None:
            self.model = model
        if not feature_names:
            raise ValueError("feature_names must be provided")

        # هم‌ترازی فیچرها
        X = self._align_features(df, feature_names)

        # تلاش برای گرفتن predict_proba
        prob_pos = None
        try:
            proba = model.predict_proba(X)
            # تعیین ستون مربوط به کلاس مثبت (1)
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 1 in classes:
                    idx_pos = classes.index(1)
                else:
                    # اگر کلاس 1 موجود نیست، فرض می‌کنیم آخرین ستون کلاس مثبت است
                    idx_pos = -1
            else:
                idx_pos = 1 if proba.shape[1] > 1 else 0
            prob_pos = proba[:, idx_pos]
        except Exception:
            # fallback: مدل احتمالا فقط predict دارد یا خطا داده است
            try:
                preds = model.predict(X)
                # تبدیل پیش‌بینی دوتایی به احتمال تقریبی (1->0.9, 0->0.1)
                prob_pos = np.where(preds == 1, 0.9, 0.1)
                warnings.warn("Model does not support predict_proba; using deterministic mapping of predict -> pseudo-proba.")
            except Exception as e:
                raise RuntimeError(f"Model cannot predict probabilities nor labels: {e}")

        prob_series = pd.Series(prob_pos, index=df.index)

        upper = prob_threshold
        lower = 1.0 - prob_threshold

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[prob_series >= upper] = 1
        signals[prob_series <= lower] = -1

        return signals

    def extract_trade_log(self, pf) -> pd.DataFrame:
        """استخراج گزارش معاملات به شکل DataFrame."""
        try:
            if hasattr(pf, "trades"):
                trades = pf.trades
                if hasattr(trades, "records_read"):
                    trade_df = trades.records_read
                elif hasattr(trades, "records"):
                    trade_df = trades.records
                elif hasattr(trades, "to_dict"):
                    trade_df = trades.to_dict()
                else:
                    trade_df = trades
            else:
                trade_df = None
        except Exception:
            trade_df = None

        if trade_df is None:
            return pd.DataFrame(columns=[
                "entry_time", "exit_time", "direction", "entry_price", "exit_price",
                "pnl", "return_pct", "duration", "fees", "slippage"
            ])

        if isinstance(trade_df, dict):
            trade_df = pd.DataFrame(trade_df)

        if not isinstance(trade_df, pd.DataFrame):
            try:
                trade_df = pd.DataFrame(trade_df)
            except Exception:
                return pd.DataFrame(columns=[
                    "entry_time", "exit_time", "direction", "entry_price", "exit_price",
                    "pnl", "return_pct", "duration", "fees", "slippage"
                ])

        trade_df = trade_df.copy()
        rename_map = {
            "open_time": "entry_time",
            "close_time": "exit_time",
            "entry": "entry_price",
            "exit": "exit_price",
            "pnl": "pnl",
            "profit": "pnl",
            "return": "return_pct",
            "direction": "direction",
            "duration": "duration",
            "fee": "fees",
            "fees": "fees",
            "slippage": "slippage",
        }
        for old_name, new_name in rename_map.items():
            if old_name in trade_df.columns and new_name not in trade_df.columns:
                trade_df.rename(columns={old_name: new_name}, inplace=True)

        required = [
            "entry_time", "exit_time", "direction", "entry_price", "exit_price",
            "pnl", "return_pct", "duration", "fees", "slippage"
        ]
        for col in required:
            if col not in trade_df.columns:
                trade_df[col] = np.nan

        # تبدیل به DataFrame قابل ذخیره‌سازی و مستقیم برای CSV
        trade_df = trade_df[required].copy()
        trade_df["entry_time"] = pd.to_datetime(trade_df["entry_time"], errors="coerce")
        trade_df["exit_time"] = pd.to_datetime(trade_df["exit_time"], errors="coerce")
        return trade_df

    def save_trade_log(self, pf, output_path: str) -> pd.DataFrame:
        """ذخیرهٔ فایل CSV trade log و بازگشت DataFrame."""
        trade_df = self.extract_trade_log(pf)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        trade_df.to_csv(output_path, index=False)
        return trade_df

    def save_additional_reports(self, pf, out_dir: str) -> dict:
        """ذخیرهٔ نمودارهای اضافی: drawdown و monthly returns heatmap."""
        eq = None
        try:
            eq = pf.total_equity()
        except Exception:
            eq = None

        outputs: dict[str, str] = {}

        if eq is not None:
            # Equity Curve
            fig, ax = plt.subplots(figsize=(12, 6))
            eq.plot(ax=ax)
            ax.set_title("Equity Curve")
            ax.set_xlabel("Time")
            ax.set_ylabel("Portfolio Value")
            eq_path = os.path.join(out_dir, "equity_curve.png")
            fig.tight_layout()
            fig.savefig(eq_path)
            plt.close(fig)
            outputs["equity_curve"] = eq_path

            # Drawdown
            try:
                dd = pf.drawdown()
            except Exception:
                dd = eq / eq.cummax() - 1.0
            fig, ax = plt.subplots(figsize=(12, 5))
            dd.plot(ax=ax, color="red")
            ax.set_title("Drawdown (%)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Drawdown")
            dd_path = os.path.join(out_dir, "drawdown_curve.png")
            fig.tight_layout()
            fig.savefig(dd_path)
            plt.close(fig)
            outputs["drawdown_curve"] = dd_path

            # Monthly Returns Heatmap
            try:
                ret = eq.pct_change().dropna()
                monthly = ret.resample("M").sum().to_frame(name="return")
                monthly["Year"] = monthly.index.year
                monthly["Month"] = monthly.index.month
                pivot = monthly.pivot_table(values="return", index="Year", columns="Month", aggfunc="mean")
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                pivot = pivot.reindex(columns=list(range(1, 13)), fill_value=np.nan)
                pivot.columns = [month_names[i - 1] for i in range(1, 13)]
                fig, ax = plt.subplots(figsize=(12, 6))
                im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", aspect="auto")
                ax.set_title("Monthly Returns Heatmap")
                ax.set_xticks(np.arange(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns)
                ax.set_yticks(np.arange(len(pivot.index)))
                ax.set_yticklabels([str(y) for y in pivot.index])
                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        val = pivot.iloc[i, j]
                        if pd.notna(val):
                            ax.text(j, i, f"{val:.2%}", ha="center", va="center", color="black")
                fig.colorbar(im, ax=ax, label="Return")
                fig.tight_layout()
                heatmap_path = os.path.join(out_dir, "monthly_returns_heatmap.png")
                fig.savefig(heatmap_path)
                plt.close(fig)
                outputs["monthly_returns_heatmap"] = heatmap_path
            except Exception as e:
                warnings.warn(f"Failed to generate monthly returns heatmap: {e}")

        return outputs

    def _compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """محاسبه ATR ساده (Moving Average of True Range)."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window, min_periods=1).mean()
        return atr

    def run_backtest(self, df: pd.DataFrame, signals: pd.Series, initial_capital: float = 10000, commission: float = 0.0005, slippage: float = 0.0002, position_sizing: dict | None = None):
        """اجرای backtest با vectorbt.

        - df باید شامل ستون 'close' باشد که قیمت برای backtest استفاده می‌شود.
        - signals باید اندیس مشابه df داشته باشد.

        موقعیت‌های sizing پشتیبانی‌شده (position_sizing dict):
          method: 'fractional' | 'fixed_risk' | 'atr'
          برای 'fractional': fraction (مثال 0.02 برای 2% از سرمایه اولیه)
          برای 'fixed_risk': risk_amount (دلار یا واحد پولی)، stop_pct (اختیاری، پیش‌فرض 0.01)
          برای 'atr': risk_amount (دلار)، atr_multiplier (مثال 3), atr_window (مثال 14)

        بازگشتی: (portfolio, metrics_dict)
        metrics_dict شامل: Total Return (%), Sharpe Ratio, Max Drawdown (%), Win Rate (%), Total Number of Trades
        """
        if vbt is None:
            raise RuntimeError("vectorbt is not installed. Please install with: pip install 'vectorbt[full]' and ensure plotly is available.")

        if "close" not in df.columns:
            raise ValueError("Input dataframe must contain 'close' column for price series")

        close = df["close"]

        # تبدیل سیگنال‌ها به نقاط ورود/خروج برای long/short
        sig = signals.reindex(close.index).fillna(0).astype(int)

        # entry زمانی است که سیگنال به 1 تغییر می‌کند از غیر 1 به 1
        entries = (sig == 1) & (sig.shift(1).fillna(0) != 1)
        exits = (sig != 1) & (sig.shift(1).fillna(0) == 1)

        short_entries = (sig == -1) & (sig.shift(1).fillna(0) != -1)
        short_exits = (sig != -1) & (sig.shift(1).fillna(0) == -1)

        # پیش‌فرض: بدون sizing خاص (هر معامله یک واحد)
        size = None
        short_size = None

        if position_sizing is not None:
            method = position_sizing.get("method", "fractional")
            if method == "fractional":
                fraction = float(position_sizing.get("fraction", 0.02))
                # تعداد واحد = floor((fraction * initial_capital) / price)
                size_vals = np.floor((fraction * initial_capital) / close.replace(0, np.nan)).fillna(0).astype(int)
                short_size_vals = size_vals.copy()
                # فقط در نقاط ورود اعمال شود
                size = pd.Series(0, index=close.index, dtype=float)
                short_size = pd.Series(0, index=close.index, dtype=float)
                size.loc[entries] = size_vals.loc[entries]
                short_size.loc[short_entries] = short_size_vals.loc[short_entries]

            elif method == "fixed_risk":
                risk_amount = float(position_sizing.get("risk_amount", 100.0))
                stop_pct = float(position_sizing.get("stop_pct", 0.01))
                # stop_distance per share = price * stop_pct
                stop_dist = close * stop_pct
                # shares = floor(risk_amount / stop_dist)
                shares = np.floor(risk_amount / stop_dist.replace(0, np.nan)).fillna(0).astype(int)
                size = pd.Series(0, index=close.index, dtype=float)
                short_size = pd.Series(0, index=close.index, dtype=float)
                size.loc[entries] = shares.loc[entries]
                short_size.loc[short_entries] = shares.loc[short_entries]

            elif method == "atr":
                risk_amount = float(position_sizing.get("risk_amount", 100.0))
                atr_multiplier = float(position_sizing.get("atr_multiplier", 3.0))
                atr_window = int(position_sizing.get("atr_window", 14))
                if not all(col in df.columns for col in ["high", "low", "close"]):
                    raise ValueError("ATR sizing requires 'high', 'low', and 'close' columns in df")
                atr = self._compute_atr(df["high"], df["low"], df["close"], window=atr_window)
                stop_dist = atr * atr_multiplier
                shares = np.floor(risk_amount / stop_dist.replace(0, np.nan)).fillna(0).astype(int)
                size = pd.Series(0, index=close.index, dtype=float)
                short_size = pd.Series(0, index=close.index, dtype=float)
                size.loc[entries] = shares.loc[entries]
                short_size.loc[short_entries] = shares.loc[short_entries]

            else:
                raise ValueError(f"Unknown position_sizing method: {method}")

        # استفاده از vectorbt Portfolio.from_signals
        # fees در vectorbt نشان‌دهندهٔ نسبت ثابت کارمزد به حجم معامله است
        pf = vbt.Portfolio.from_signals(close, entries, exits,
                                        short_entries=short_entries, short_exits=short_exits,
                                        size=size, short_size=short_size,
                                        init_cash=initial_capital, fees=commission, slippage=slippage,
                                        freq=close.index.freq or None)

        # استخراج معیارها
        # total_return به صورت درصد تبدیل می‌شود
        try:
            total_return = float(pf.total_return() * 100)
        except Exception:
            total_return = float((pf.total_return() if hasattr(pf, 'total_return') else pf.stats()['Total Return']) * 100)

        try:
            sharpe = float(pf.sharpe_ratio())
        except Exception:
            sharpe = float(pf.stats().get('Sharpe Ratio', np.nan))

        try:
            max_dd = float(pf.max_drawdown() * 100)
        except Exception:
            max_dd = float(pf.stats().get('Max Drawdown', np.nan))

        try:
            win_rate = float(pf.win_rate() * 100)
        except Exception:
            win_rate = float(pf.stats().get('Win Rate', np.nan))

        try:
            total_trades = int(pf.total_trades())
        except Exception:
            total_trades = int(pf.stats().get('Total Trades', 0))

        metrics = {
            "Total Return (%)": total_return,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_dd,
            "Win Rate (%)": win_rate,
            "Total Trades": total_trades,
        }

        return pf, metrics
