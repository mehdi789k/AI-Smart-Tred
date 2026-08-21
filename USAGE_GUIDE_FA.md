# AI Smart Trader - راهنمای استفاده واقعی

## 📋 پیش‌نیازها

### نصب وابستگی‌ها
```bash
pip install -r requirements.txt
```

### تنظیمات MetaTrader5 (فقط برای ویندوز)
1. نصب MetaTrader5 از وبسایت رسمی
2. ایجاد حساب دمو یا واقعی
3. دریافت اطلاعات ورود (Login, Password, Server)

### متغیرهای محیطی
یک فایل `.env` در ریشه پروژه بسازید:
```bash
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=MetaQuotes-Demo
MT5_TERMINAL_PATH=C:\Program Files\MetaTrader 5
DATA_SOURCE=mock  # گزینه‌ها: mt5, csv, mock
```

---

## 🚀 راه‌اندازی سریع

### مرحله ۱: دریافت داده‌ها
```bash
# حالت Mock (برای تست بدون MT5)
python scripts/fetch_real_data.py

# حالت واقعی (نیاز به MT5 دارد)
export DATA_SOURCE=mt5
python scripts/fetch_real_data.py

# استفاده از فایل CSV
export DATA_SOURCE=csv
python scripts/fetch_real_data.py
```

### مرحله ۲: اجرای کامل پایپ‌لاین
```bash
# آموزش مدل و بکتست برای EURUSD
python scripts/run_full_pipeline.py --symbol EURUSD

# برای جفت‌ارزهای دیگر
python scripts/run_full_pipeline.py --symbol GBPUSD
python scripts/run_full_pipeline.py --symbol XAUUSD
```

### مرحله ۳: معامله کاغذی (Paper Trading)
```bash
# شروع Paper Trading
python scripts/live_trader.py --symbol EURUSD --mode paper --interval 60
```

### مرحله ۴: معامله واقعی (Live Trading) ⚠️
```bash
# فقط پس از اطمینان از عملکرد مدل در بکتست و Paper Trading
python scripts/live_trader.py --symbol EURUSD --mode live --interval 60
```

---

## ⚙️ تنظیمات پیشرفته

### تنظیم پارامترهای Triple-Barrier
فایل `config/trading_config.py` را ویرایش کنید:

```python
@dataclass
class LabelConfig:
    profit_target_bars: int = 50      # افق Take Profit (تعداد کندل)
    stop_loss_bars: int = 50          # افق Stop Loss
    time_limit_bars: int = 100        # حداکثر دوره نگهداری
    
    tp_multiplier: float = 1.5        # TP = Entry + (ATR × 1.5)
    sl_multiplier: float = 1.0        # SL = Entry - (ATR × 1.0)
    
    use_dynamic_thresholds: bool = True  # استفاده از ATR داینامیک
```

**پیشنهادها برای بازارهای مختلف:**
- **EURUSD**: TP=1.5, SL=1.0, Time Limit=100
- **XAUUSD (Gold)**: TP=2.0, SL=1.5, Time Limit=80 (نوسان بیشتر)
- **GBPJPY**: TP=2.5, SL=2.0, Time Limit=60 (نوسان خیلی زیاد)

### تنظیم مدیریت ریسک
```python
@dataclass
class BacktestConfig:
    initial_capital: float = 10000.0   # سرمایه اولیه
    commission_pct: float = 0.0001     # کارمزد (0.01%)
    risk_per_trade: float = 0.01       # ریسک هر معامله (1%)
    max_positions: int = 1             # حداکثر پوزیشن همزمان

@dataclass
class LiveTradingConfig:
    max_daily_loss_pct: float = 3.0    # حد ضرر روزانه
    max_drawdown_pct: float = 10.0     # حداکثر افت سرمایه
    news_filter_enabled: bool = True   # فیلتر اخبار
```

---

## 📊 تفسیر خروجی‌ها

### گزارش مدل
پس از آموزش، فایل `data/model_releases/{SYMBOL}_report_v1.txt` شامل:
- ماتریس آشفتگی (Confusion Matrix)
- دقت (Accuracy) برای هر کلاس
- F1-Score加权平均

### نتایج بکتست
خروجی بکتست شامل:
- **Sharpe Ratio**: نسبت سود به نوسان (بالای 1 خوب است)
- **Max Drawdown**: حداکثر افت سرمایه (زیر 20% مطلوب)
- **Total Return**: بازده کل
- **Win Rate**: نرخ برد معاملات

---

## 🔧 عیب‌یابی

### خطای "Model not found"
```bash
# ابتدا پایپ‌لاین کامل را اجرا کنید
python scripts/run_full_pipeline.py --symbol EURUSD
```

### خطای MT5 Connection
- مطمئن شوید MT5 نصب و اجرا شده است
- بررسی کنید `MT5_TERMINAL_PATH` صحیح باشد
- در لینوکس/macOS از حالت `mock` یا `csv` استفاده کنید

### هشدارهای Pandas
- این هشدارها خطرناک نیستند
- برای رفع: `freq='T'` → `freq='min'`, `freq='H'` → `freq='h'`

---

## 📈 بهبود عملکرد

### ۱. بهینه‌سازی هایپرپارامترها
```bash
python scripts/hyperparameter_tuning.py --symbol EURUSD
```

### ۲. افزودن ویژگی‌های جدید
- ویژگی‌های سفارشی به `src/feature_engineering/feature_engine.py` اضافه کنید
- اندیکاتورهای تکنیکال پیشرفته (Ichimoku, ADX, etc.)

### ۳. Walk-Forward Analysis
برای اعتبارسنجی قوی‌تر:
```python
# در src/ml_trainer/train_model.py
from sklearn.model_selection import TimeSeriesSplit
```

### ۴. Ensemble Models
ترکیب چند مدل برای کاهش واریانس:
```python
from sklearn.ensemble import VotingClassifier
```

---

## ⚠️ هشدارهای مهم

1. **ریسک مالی**: این سیستم برای اهداف آموزشی است. قبل از استفاده واقعی، حتماً:
   - بکتست گسترده انجام دهید
   - حداقل ۳ ماه Paper Trading کنید
   - با سرمایه بسیار کم شروع کنید

2. **Overfitting**: مدل‌های ML ممکن است روی داده‌های گذشته بیش‌ازحد برازش شوند
   - از Purged K-Fold استفاده شده است
   - همیشه Out-of-Sample تست کنید

3. **Market Regime Changes**: الگوهای بازار تغییر می‌کنند
   - مدل را هر ۳-۶ ماه دوباره آموزش دهید
   - نظارت مستمر بر عملکرد داشته باشید

---

## 📞 پشتیبانی

- مستندات کامل: `docs/README.md`
- مثال‌های بیشتر: `examples/`
- گزارش باگ: GitHub Issues

**موفق باشید! 🚀**
