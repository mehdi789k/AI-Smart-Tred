# 📊 AI Smart Trader - Professional Dashboard

## داشبورد حرفه‌ای معاملات الگوریتمی در سطح جهانی

یک داشبورد تعاملی و حرفه‌ای برای سیستم معاملات الگوریتمی مبتنی بر هوش مصنوعی که با **Streamlit** و **Plotly** ساخته شده است.

---

## 🚀 راه‌اندازی سریع

### نصب وابستگی‌ها
```bash
pip install streamlit plotly pandas lightgbm scikit-learn
```

### اجرای داشبورد
```bash
streamlit run dashboard.py --server.headless true --server.port 8501
```

### دسترسی به داشبورد
- **Local URL:** http://localhost:8501
- **Network URL:** http://YOUR_IP:8501

---

## ✨ ویژگی‌های کلیدی

### 1. 📈 Market Overview (نمای کلی بازار)
- نمودار شمعی ژاپنی (Candlestick) تعاملی
- نمایش حجم معاملات
- شاخص‌های کلیدی بازار (ATR, RSI, Bollinger Bands, ADX)
- توزیع سیگنال‌های خرید/فروش/نگهداری

### 2. 🤖 Model Analytics (تحلیل مدل)
- معیارهای عملکرد مدل (Accuracy, Precision, Recall, F1 Score)
- اهمیت ویژگی‌ها (Feature Importance)
- توزیع اطمینان پیش‌بینی‌ها
- ماتریس آشفتگی (Confusion Matrix)

### 3. 💰 Backtest Results (نتایج بک‌تست)
- منحنی سرمایه (Equity Curve)
- تحلیل Drawdown
- Heatmap عملکرد بر اساس ساعت و روز هفته
- جدول معاملات اخیر با جزئیات کامل
- معیارهای کلیدی:
  - Total P&L
  - Win Rate
  - Profit Factor
  - Sharpe Ratio
  - Max Drawdown

### 4. ⚡ Live Signals (سیگنال‌های زنده)
- نمایش سیگنال‌های فعال با فیلتر اطمینان
- جزئیات هر سیگنال (نوع، قدرت، قیمت ورود، بازده پیش‌بینی‌شده)
- آمار سیگنال‌ها (تعداد خرید/فروش، میانگین اطمینان)
- ماشین‌حساب اندازه پوزیشن (Position Size Calculator)

### 5. ⚙️ Settings (تنظیمات)
- پیکربندی اتصال به داده‌ها (MT5, Database)
- آپلود مدل‌های آموزش‌دیده
- تنظیمات اطلاع‌رسانی (Email, Telegram, Webhook)
- خروجی گرفتن از داده‌ها (CSV, JSON)

---

## 🎨 طراحی رابط کاربری

### تم تاریک حرفه‌ای
- استفاده از `plotly_dark` template
- رنگ‌بندی حرفه‌ای برای کندل‌ها (سبز/قرمز)
- کارت‌های متریک با فونت بزرگ و خوانا

### چیدمان واکنش‌گرا
- Sidebar برای کنترل‌های اصلی
- Tabs برای دسته‌بندی بخش‌ها
- Columns برای نمایش همزمان چندین نمودار

### المان‌های تعاملی
- SelectBox برای انتخاب نماد و تایم‌فریم
- Slider برای تنظیم آستانه اطمینان و ریسک
- Expander برای نمایش جزئیات سیگنال‌ها
- Download Button برای خروجی داده‌ها

---

## 🔧 شخصی‌سازی

### تغییر نمادهای معاملاتی
در تابع `main()`، لیست `symbols` را ویرایش کنید:
```python
symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'BTCUSD', 'ETHUSD']
```

### اتصال به داده‌های واقعی
تابع `load_sample_data()` را با کد زیر جایگزین کنید:
```python
def load_real_data(symbol='EURUSD', timeframe='1H'):
    # اتصال به MT5 یا دیتابیس
    from MetaTrader5 import get_rates
    data = get_rates(symbol, timeframe, 1000)
    return pd.DataFrame(data)
```

### تنظیم پارامترهای ریسک
در Sidebar، اسلایدرهای ریسک را تنظیم کنید:
- Risk per Trade: 0.5% تا 5.0%
- Max Drawdown Limit: 5% تا 20%

---

## 📊 نمونه تصاویر داشبورد

### Tab 1: Market Overview
- نمودار Candlestick با Volume
- Pie Chart توزیع سیگنال‌ها
- 4 کارت متریک (قیمت، حجم، سیگنال‌ها، نوسان)

### Tab 2: Model Analytics
- Bar Chart اهمیت ویژگی‌ها
- Histogram توزیع اطمینان
- Confusion Matrix حرارتی

### Tab 3: Backtest Results
- Equity Curve با fill area
- Drawdown Chart
- Performance Heatmap
- جدول معاملات با رنگ‌بندی WIN/LOSS

### Tab 4: Live Signals
- لیست سیگنال‌های فعال در Expander
- ماشین‌حساب اندازه پوزیشن

---

## 🛠️ تکنولوژی‌های استفاده‌شده

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Charts | Plotly |
| Data Processing | Pandas, NumPy |
| ML Backend | LightGBM, Scikit-learn |
| Data Source | MetaTrader 5, SQLite |

---

## 📝 نکات مهم

1. **داده‌های نمونه**: در حال حاضر از داده‌های تصادفی استفاده می‌شود. برای استفاده واقعی، توابع `load_sample_data()` و `load_backtest_results()` را به منابع داده واقعی متصل کنید.

2. **Auto Refresh**: گزینه Auto Refresh در Sidebar هر 30 ثانیه صفحه را تازه می‌کند.

3. **Export Data**: امکان دانلود نتایج بک‌تست (CSV) و سیگنال‌ها (JSON) فراهم است.

4. **Risk Calculator**: ماشین‌حساب اندازه پوزیشن بر اساس فرمول استاندارد محاسبه می‌کند:
   ```
   Position Size = (Account Balance × Risk%) / (Stop Loss × Pip Value)
   ```

---

## 🚀 گام‌های بعدی

- [ ] اتصال به API واقعی MT5
- [ ] افزودن نمودارهای TradingView
- [ ] یکپارچه‌سازی با Telegram Bot
- [ ] تولید گزارش PDF
- [ ] پشتیبانی از چندین حساب معاملاتی
- [ ] افزودن حالت موبایل (Responsive Design)

---

## 📄 مجوز

این پروژه تحت مجوز MIT منتشر شده است.

---

## 🤝 مشارکت

برای گزارش باگ یا پیشنهاد ویژگی جدید، لطفاً Issue ایجاد کنید.

---

**ساخته شده با ❤️ توسط AI Smart Trader Team**
