"""smarttred.models.trainer

این ماژول کلاس MLTrainer را پیاده‌سازی می‌کند که مسئول آماده‌سازی داده، انجام Purged K-Fold Cross-Validation با Embargo
و آموزش یک مدل درختی (LightGBM) برای پیش‌بینی ستون target است.

توضیحات (به فارسی):
- از متد purged_kfold_split برای جلوگیری از Data Leakage در سری‌های زمانی مالی استفاده می‌شود.
- متد prepare_data فیچرها و هدف را جدا می‌کند و ستون‌های غیر فیچر را حذف می‌کند.
- متد train_and_evaluate مدل را با استفاده از تقسیم‌بندی purged-kfold آموزش داده و گزارش معیارها و ماتریس آشفتگی را برای هر فولد چاپ می‌کند.

نکته: انتظار می‌رود dataframe ورودی شامل ستون‌های زمان مشاهده (timestamp یا ایندکس DatetimeIndex) و ستون پایان برچسب (t1)
باشد که توسط روش Triple-Barrier تعیین شده است.
"""
from __future__ import annotations

from typing import Generator, List, Optional, Tuple
import os
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, classification_report)
import matplotlib.pyplot as plt
import lightgbm as lgb


class MLTrainer:
    """کلاس آموزش مدل‌های درختی با Purged K-Fold و Embargo.

    پارامترها:
    - model_params: دیکشنری پارامترها برای LGBM
    - n_splits: تعداد فولدها
    - embargo_pct: درصد داده برای embargo (مثلاً 0.05)
    - random_state: بذر تصادفی (برای reproducibility)
    """

    def __init__(self, model_params: Optional[dict] = None, n_splits: int = 5,
                 embargo_pct: float = 0.05, random_state: int = 42) -> None:
        self.model_params = model_params or {
            "objective": "binary",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbosity": -1,
            # is_unbalance handled later depending on labels
        }
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.random_state = random_state
        self.df: Optional[pd.DataFrame] = None

    def prepare_data(self, df: pd.DataFrame, target_col: str = "target") -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """جدا کردن X و y و لیست نام فیچرها.

        این متد نسخه‌ای از dataframe را در self.df نگه می‌دارد (برای استفادهٔ purged_kfold_split)
        و ستون‌های غیر فیچر را از X حذف می‌کند. ستون‌های غیر فیچر پیشنهادی:
        ['timestamp', 't1', 'barrier_type', 'open', 'high', 'low', 'close', 'volume']

        بازگشتی: X, y, feature_names
        """
        self.df = df.copy()

        # بررسی وجود ستون timestamp یا time یا داشتن ایندکس Datetime
        if "timestamp" in self.df.columns:
            # اطمینان از نوع datetime
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])  # type: ignore
        elif "time" in self.df.columns:
            # تبدیل ستون time به timestamp برای سازگاری
            self.df["timestamp"] = pd.to_datetime(self.df["time"], unit='s')  # type: ignore
        else:
            # اگر ایندکس از نوع DatetimeIndex است، آن را به ستون timestamp منتقل نمی‌کنیم ولی purged_kfold از ایندکس استفاده خواهد کرد
            if not isinstance(self.df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame must contain a 'timestamp' or 'time' column or have a DatetimeIndex")

        # ستون‌های غیر فیچر که باید حذف شوند (در صورت وجود)
        non_feature_cols = ["timestamp", "t1", "barrier_type", "open", "high", "low", "close", "volume"]
        non_feature_present = [c for c in non_feature_cols if c in self.df.columns]

        if target_col not in self.df.columns:
            raise ValueError(f"target column '{target_col}' not found in dataframe")

        y = self.df[target_col].copy()
        X = self.df.drop(columns=[c for c in non_feature_present if c in self.df.columns] + [target_col], errors="ignore")

        feature_names = list(X.columns)
        return X, y, feature_names

    def purged_kfold_split(self, df: pd.DataFrame, n_splits: Optional[int] = None, embargo_pct: Optional[float] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """نسخهٔ ساده ولی مؤثر از Purged K-Fold with Embargo.

        ورودی:
        - df: دادهٔ اصلی که حداقل باید شامل ستون‌های زمان مشاهده (timestamp یا DatetimeIndex) و ستون 't1' (زمان پایان برچسب) باشد.
        - n_splits: تعداد فولدها (در صورت None از self.n_splits استفاده شود)
        - embargo_pct: نسبت داده که باید پس از انتهای هر تست به عنوان embargo حذف شود

        خروجی: تولیدگر زوج ایندکس‌های آموزشی و تست (به صورت integer positions)

        پیاده‌سازی:
        - داده را به n_splits بلوک پیوسته بر اساس ترتیب زمانی تقسیم می‌کند تا تست‌ها بلوک‌های زمانی باشند.
        - برای هر بلوک تست، نمونه‌های آموزشی را پاک می‌کند (purge) اگر پنجره برچسب آن‌ها (از زمان مشاهده تا t1)
          با بازهٔ زمانی نمونه‌های تست همپوشانی داشته باشد.
        - embargo: پس از بازهٔ زمانی نمونهٔ تست، یک بازهٔ زمانی به اندازهٔ embargo_pct از داده (بر حسب تعداد سطرها) از آموزش حذف می‌شود.
        """
        n_splits = n_splits or self.n_splits
        embargo_pct = embargo_pct if embargo_pct is not None else self.embargo_pct

        N = len(df)
        if N < n_splits:
            raise ValueError("Number of splits cannot be larger than number of samples")

        # استخراج زمان مشاهده (start times) و زمان پایان برچسب (t1)
        if "timestamp" in df.columns:
            obs_times = pd.to_datetime(df["timestamp"])  # type: ignore
        else:
            if isinstance(df.index, pd.DatetimeIndex):
                obs_times = pd.Series(df.index, index=df.index)
                obs_times = pd.to_datetime(obs_times).reset_index(drop=True)
            else:
                raise ValueError("DataFrame must contain 'timestamp' column or DatetimeIndex for purged_kfold_split")

        if "t1" not in df.columns:
            raise ValueError("DataFrame must contain 't1' column with label end timestamps for purged_kfold_split")

        t1_times = pd.to_datetime(df["t1"]).reset_index(drop=True)
        obs_times = pd.to_datetime(obs_times).reset_index(drop=True)

        # تقسیم اندیس‌ها به بلوک‌های متوالی برای تست
        indices = np.arange(N)
        fold_sizes = np.full(n_splits, N // n_splits, dtype=int)
        fold_sizes[: N % n_splits] += 1
        current = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            folds.append(indices[start:stop])
            current = stop

        embargo_size = int(N * embargo_pct)

        for i in range(n_splits):
            test_idx = folds[i]

            test_start_time = obs_times.iloc[test_idx].min()
            test_end_time = obs_times.iloc[test_idx].max()

            # کاندیداهای آموزش: تمام اندیس‌ها غیر از تست
            train_candidates = np.setdiff1d(indices, test_idx)

            # Purge: حذف نمونه‌هایی که پنجرهٔ برچسب‌شان با بازهٔ تست همپوشانی دارد
            # پنجرهٔ برچسب برای سطر j از obs_time[j] تا t1_times[j]
            purge_mask = []
            for j in train_candidates:
                start_j = obs_times.iloc[j]
                end_j = t1_times.iloc[j]
                # اگر end_j نال باشد (label ندارد)، امن در نظر می‌گیریم و purge نمی‌کنیم
                if pd.isna(end_j):
                    purge_mask.append(False)
                    continue
                # شرط همپوشانی
                overlap = (start_j <= test_end_time) and (end_j >= test_start_time)
                purge_mask.append(overlap)

            to_purge = train_candidates[np.array(purge_mask, dtype=bool)]
            train_idx = np.setdiff1d(train_candidates, to_purge)

            # Embargo: حذف اندیس‌های بلافاصله بعد از محدودهٔ تست به اندازهٔ embargo_size
            test_max_idx = test_idx.max()
            embargo_start = test_max_idx + 1
            embargo_end = min(test_max_idx + embargo_size, N - 1)
            if embargo_start <= embargo_end:
                embargo_indices = np.arange(embargo_start, embargo_end + 1)
                train_idx = np.setdiff1d(train_idx, embargo_indices)

            yield train_idx.astype(int), test_idx.astype(int)

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> lgb.LGBMClassifier:
        """آموزش مدل با استفاده از Purged K-Fold و چاپ معیارها.

        بازگشتی: مدل آموزش دیدهٔ نهایی (که روی تمامی داده‌ها دوباره آموزش داده شده تا برای استقرار استفاده شود).
        """
        if self.df is None:
            raise RuntimeError("self.df is not set. Call prepare_data(df) before train_and_evaluate")

        metrics = {
            "f1": [],
            "precision": [],
            "recall": []
        }
        fold_models = []

        # تصمیم‌گیری در مورد نوع مسئله (دوتایی یا چندکلاسه)
        unique_labels = np.unique(y.dropna())
        is_binary = len(unique_labels) == 2

        # برای هر فولد
        for fold_idx, (train_idx, test_idx) in enumerate(self.purged_kfold_split(self.df, n_splits=self.n_splits, embargo_pct=self.embargo_pct)):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            # مدیریت عدم توازن کلاس
            lgb_params = dict(self.model_params)  # copy
            if is_binary:
                # scale_pos_weight: نسبت منفی/مثبت
                counts = y_train.value_counts()
                if 1 in counts.index and 0 in counts.index:
                    neg = counts.get(0, 0)
                    pos = counts.get(1, 0)
                    if pos == 0:
                        lgb_params["scale_pos_weight"] = 1.0
                    else:
                        lgb_params["scale_pos_weight"] = float(neg) / float(pos)
                else:
                    lgb_params["is_unbalance"] = True
            else:
                # multiclass
                lgb_params["objective"] = "multiclass"
                lgb_params["num_class"] = int(len(unique_labels))
                lgb_params["class_weight"] = "balanced"

            model = lgb.LGBMClassifier(**lgb_params)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # محاسبه معیارها
            if is_binary:
                f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
                prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
                rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
            else:
                f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
                prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
                rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

            metrics["f1"].append(f1)
            metrics["precision"].append(prec)
            metrics["recall"].append(rec)
            fold_models.append((model, f1))

            # چاپ گزارش
            print(f"\nFold {fold_idx + 1}/{self.n_splits}")
            print(f"Samples train={len(train_idx)}, test={len(test_idx)}")
            print(f"F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
            print("Classification report:")
            try:
                print(classification_report(y_test, y_pred, zero_division=0))
            except Exception:
                pass

            # ماتریس اشتباه
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix - Fold {fold_idx + 1}")
            plt.colorbar()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            for (m, n), val in np.ndenumerate(cm):
                plt.text(n, m, int(val), ha='center', va='center', color='red')
            out_dir = os.path.join(os.getcwd(), "data", "model_releases")
            os.makedirs(out_dir, exist_ok=True)
            cm_path = os.path.join(out_dir, f"confusion_matrix_fold_{fold_idx + 1}.png")
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close()
            print(f"Confusion matrix saved to: {cm_path}")

        # گزارش میانگین معیارها
        mean_f1 = float(np.mean(metrics["f1"]))
        mean_prec = float(np.mean(metrics["precision"]))
        mean_rec = float(np.mean(metrics["recall"]))
        print("\nCross-validation results:")
        print(f"Mean F1: {mean_f1:.4f}")
        print(f"Mean Precision: {mean_prec:.4f}")
        print(f"Mean Recall: {mean_rec:.4f}")

        # انتخاب بهترین فلد بر اساس f1
        best_model, best_f1 = max(fold_models, key=lambda x: x[1])
        print(f"Best fold F1: {best_f1:.4f}")

        # آموزش نهایی مدل روی کل داده‌ها (برای استقرار)
        final_params = dict(self.model_params)
        if is_binary:
            counts_full = y.value_counts()
            if 1 in counts_full.index and 0 in counts_full.index:
                neg = counts_full.get(0, 0)
                pos = counts_full.get(1, 0)
                if pos == 0:
                    final_params["scale_pos_weight"] = 1.0
                else:
                    final_params["scale_pos_weight"] = float(neg) / float(pos)
            else:
                final_params["is_unbalance"] = True
        else:
            final_params["objective"] = "multiclass"
            final_params["num_class"] = int(len(unique_labels))
            final_params["class_weight"] = "balanced"

        final_model = lgb.LGBMClassifier(**final_params)
        final_model.fit(X, y)

        # ترسیم اهمیت فیچرها (Top 10)
        try:
            importances = final_model.feature_importances_
            feat_imp = pd.Series(importances, index=feature_names)
            feat_imp_sorted = feat_imp.sort_values(ascending=False)
            topk = min(10, len(feat_imp_sorted))
            top_feats = feat_imp_sorted.iloc[:topk]

            plt.figure(figsize=(8, 6))
            top_feats.plot(kind='bar')
            plt.title('Top feature importances')
            plt.tight_layout()
            fi_path = os.path.join(out_dir, "feature_importance_top10.png")
            plt.savefig(fi_path)
            plt.close()
            print(f"Feature importance (top {topk}) saved to: {fi_path}")
            print("Top features:")
            print(top_feats.to_string())
        except Exception as e:
            print("Warning: failed to plot feature importances:", e)

        return final_model
