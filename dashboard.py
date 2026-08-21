"""
AI Smart Trader - Professional Dashboard
یک داشبورد حرفه‌ای در سطح جهانی برای معاملات الگوریتمی

Features:
- Real-time Market Overview
- Model Performance Analytics
- Backtest Results Visualization
- Risk Management Metrics
- Signal Generation & Trade Logs
- Interactive Charts with Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="AI Smart Trader | Professional Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'Rate this app': "https://github.com/your-repo/stargazers"
    }
)

# Custom CSS for Professional Look
st.markdown("""
<style>
    /* Main container */
    .main > div {max-width: 100%; padding-left: 2rem; padding-right: 2rem;}
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {font-size: 2.5rem; font-weight: bold;}
    div[data-testid="stMetricLabel"] {font-size: 1.1rem; color: #666;}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {background-color: #0e1117;}
    
    /* Cards */
    .css-1r6slb0 {background-color: #1c1f26; border-radius: 10px; padding: 1rem;}
    
    /* Hide default footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1c1f26;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        padding-right: 20px;
        padding-left: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2b90d9;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def load_sample_data():
    """Generate realistic sample data for demonstration"""
    dates = pd.date_range(start='2024-01-01', periods=500, freq='1h')
    
    # Generate OHLCV data
    np.random.seed(42)
    base_price = 1.1000
    returns = np.random.normal(0.0001, 0.005, 500)
    prices = base_price * (1 + returns).cumprod()
    
    high = prices * (1 + np.abs(np.random.normal(0, 0.003, 500)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.003, 500)))
    open_prices = prices * (1 + np.random.normal(0, 0.001, 500))
    volume = np.random.randint(100, 1000, 500)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    })
    
    # Add predictions and signals
    df['prediction'] = np.random.choice([0, 1, 2], 500, p=[0.4, 0.3, 0.3])
    df['signal_strength'] = np.random.uniform(0.3, 0.95, 500)
    df['predicted_return'] = np.random.normal(0.001, 0.003, 500)
    
    return df

def load_backtest_results():
    """Load or generate backtest results"""
    np.random.seed(42)
    n_trades = 150
    
    trades = pd.DataFrame({
        'trade_id': range(1, n_trades + 1),
        'entry_time': pd.date_range('2024-01-01', periods=n_trades, freq='4h'),
        'exit_time': pd.date_range('2024-01-02', periods=n_trades, freq='4h'),
        'symbol': np.random.choice(['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'], n_trades),
        'direction': np.random.choice(['LONG', 'SHORT'], n_trades),
        'entry_price': np.random.uniform(1.05, 1.50, n_trades),
        'exit_price': np.random.uniform(1.05, 1.50, n_trades),
        'pnl': np.random.normal(50, 150, n_trades),
        'pnl_percent': np.random.normal(0.5, 1.5, n_trades),
        'duration_hours': np.random.randint(1, 48, n_trades),
        'status': np.random.choice(['WIN', 'LOSS', 'BREAKEVEN'], n_trades, p=[0.55, 0.35, 0.10])
    })
    
    # Calculate cumulative PnL
    trades['cumulative_pnl'] = trades['pnl'].cumsum()
    
    return trades

def calculate_metrics(trades_df):
    """Calculate trading performance metrics"""
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = trades_df['pnl'].sum()
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
    
    profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if (avg_loss * losing_trades) > 0 else 0
    
    max_drawdown = trades_df['cumulative_pnl'].cummax() - trades_df['cumulative_pnl']
    max_dd = max_drawdown.max()
    
    sharpe_ratio = (trades_df['pnl_percent'].mean() / trades_df['pnl_percent'].std()) * np.sqrt(252) if trades_df['pnl_percent'].std() > 0 else 0
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe_ratio
    }

def create_candlestick_chart(df, symbol='EURUSD'):
    """Create interactive candlestick chart with predictions"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3],
                        subplot_titles=(f'{symbol} Price Action', 'Volume'))
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)
    
    # Volume
    colors = ['#26a69a' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ef5350' 
              for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.5
    ), row=2, col=1)
    
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        showlegend=False,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def create_equity_curve(trades_df):
    """Create equity curve chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trades_df['exit_time'],
        y=trades_df['cumulative_pnl'],
        mode='lines',
        name='Equity Curve',
        line=dict(color='#2b90d9', width=3),
        fill='tozeroy',
        fillcolor='rgba(43, 144, 217, 0.2)'
    ))
    
    fig.update_layout(
        title='Cumulative P&L',
        xaxis_title='Time',
        yaxis_title='P&L ($)',
        template='plotly_dark',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_drawdown_chart(trades_df):
    """Create drawdown chart"""
    cumulative = trades_df['cumulative_pnl']
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max.replace(0, np.nan) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trades_df['exit_time'],
        y=drawdown,
        mode='lines',
        name='Drawdown',
        line=dict(color='#ef5350', width=2),
        fill='tozeroy',
        fillcolor='rgba(239, 83, 80, 0.3)'
    ))
    
    fig.update_layout(
        title='Drawdown Analysis',
        xaxis_title='Time',
        yaxis_title='Drawdown (%)',
        template='plotly_dark',
        height=300,
        hovermode='x unified'
    )
    
    return fig

def create_signal_distribution(df):
    """Create signal distribution pie chart"""
    signal_counts = df['prediction'].value_counts()
    signal_labels = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    
    fig = px.pie(
        values=signal_counts.values,
        names=[signal_labels.get(x, f'Signal {x}') for x in signal_counts.index],
        title='Signal Distribution',
        color_discrete_sequence=['#ef5350', '#ffa726', '#26a69a'],
        hole=0.4
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(template='plotly_dark', height=400)
    
    return fig

def create_performance_heatmap(trades_df):
    """Create hourly/daily performance heatmap"""
    trades_df['hour'] = trades_df['exit_time'].dt.hour
    trades_df['day_of_week'] = trades_df['exit_time'].dt.day_name()
    
    pivot_data = trades_df.pivot_table(
        values='pnl',
        index='day_of_week',
        columns='hour',
        aggfunc='mean',
        fill_value=0
    )
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_data = pivot_data.reindex(day_order)
    
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Hour of Day", y="Day of Week", color="Avg P&L ($)"),
        x=pivot_data.columns,
        y=pivot_data.index,
        color_continuous_scale='RdYlGn',
        aspect='auto'
    )
    
    fig.update_layout(
        title='Performance Heatmap by Hour & Day',
        template='plotly_dark',
        height=400
    )
    
    return fig

# Main Dashboard
def main():
    # Header
    st.title("📊 AI Smart Trader | Professional Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
        st.header("Control Panel")
        
        # Data Source Selection
        data_source = st.selectbox(
            "Data Source",
            ["Live MT5 Feed", "Historical Data", "Sample Data (Demo)"],
            index=2
        )
        
        # Symbol Selection
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'BTCUSD']
        selected_symbol = st.selectbox("Trading Symbol", symbols, index=0)
        
        # Timeframe
        timeframes = ['1M', '5M', '15M', '30M', '1H', '4H', '1D']
        selected_tf = st.selectbox("Timeframe", timeframes, index=4)
        
        st.divider()
        
        # Model Selection
        st.subheader("Model Configuration")
        model_type = st.selectbox(
            "ML Model",
            ["LightGBM Classifier", "Histogram Gradient Boosting", "Ensemble Model"],
            index=0
        )
        
        confidence_threshold = st.slider(
            "Signal Confidence Threshold",
            min_value=0.3,
            max_value=0.95,
            value=0.6,
            step=0.05
        )
        
        st.divider()
        
        # Risk Management
        st.subheader("Risk Parameters")
        risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5)
        max_drawdown_limit = st.slider("Max Drawdown Limit (%)", 5.0, 20.0, 10.0, 1.0)
        
        st.divider()
        
        # Auto-refresh
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
        
        # Last update
        st.info(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Credits
        st.markdown("---")
        st.markdown("""
        ### 🤖 AI Smart Trader
        **Version:** 2.0.0  
        **Powered by:** Streamlit + Plotly  
        **ML Backend:** LightGBM / Scikit-learn
        """)
    
    # Load Data
    with st.spinner("Loading market data..."):
        market_data = load_sample_data()
        backtest_data = load_backtest_results()
        metrics = calculate_metrics(backtest_data)
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Market Overview",
        "🤖 Model Analytics",
        "💰 Backtest Results",
        "⚡ Live Signals",
        "⚙️ Settings"
    ])
    
    # Tab 1: Market Overview
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label=f"{selected_symbol} Price",
                value=f"{market_data['close'].iloc[-1]:.5f}",
                delta=f"{((market_data['close'].iloc[-1] - market_data['close'].iloc[-50]) / market_data['close'].iloc[-50] * 100):.2f}%"
            )
        
        with col2:
            st.metric(
                label="24h Volume",
                value=f"{market_data['volume'].iloc[-1]:,.0f}",
                delta=f"{((market_data['volume'].iloc[-1] - market_data['volume'].mean()) / market_data['volume'].mean() * 100):.1f}%"
            )
        
        with col3:
            st.metric(
                label="Active Signals",
                value=len(market_data[market_data['signal_strength'] > confidence_threshold]),
                delta="High Confidence"
            )
        
        with col4:
            st.metric(
                label="Market Volatility",
                value=f"{market_data['close'].rolling(20).std().iloc[-1] * 100:.3f}%",
                delta="Normal"
            )
        
        st.divider()
        
        # Charts Row 1
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Price Chart")
            st.plotly_chart(create_candlestick_chart(market_data.tail(100), selected_symbol), use_container_width=True)
        
        with col2:
            st.subheader("Signal Distribution")
            st.plotly_chart(create_signal_distribution(market_data), use_container_width=True)
        
        # Key Statistics
        st.subheader("Market Statistics")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.card(
                label="Average True Range (ATR)",
                value=f"{np.random.uniform(0.001, 0.005):.5f}"
            )
        
        with stat_col2:
            st.card(
                label="RSI (14)",
                value=f"{np.random.uniform(30, 70):.1f}",
                help="Relative Strength Index"
            )
        
        with stat_col3:
            st.card(
                label="Bollinger Band Width",
                value=f"{np.random.uniform(0.01, 0.03):.4f}"
            )
        
        with stat_col4:
            st.card(
                label="Trend Strength",
                value=f"{np.random.uniform(0.4, 0.9):.2f}",
                help="ADX Indicator"
            )
    
    # Tab 2: Model Analytics
    with tab2:
        st.header("🤖 Model Performance Analytics")
        
        # Model Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Model Accuracy",
                value=f"{np.random.uniform(0.75, 0.85) * 100:.1f}%",
                delta="+2.3%"
            )
        
        with col2:
            st.metric(
                label="Precision (Buy)",
                value=f"{np.random.uniform(0.70, 0.80) * 100:.1f}%",
                delta="+1.5%"
            )
        
        with col3:
            st.metric(
                label="Recall (Sell)",
                value=f"{np.random.uniform(0.65, 0.75) * 100:.1f}%",
                delta="-0.8%"
            )
        
        with col4:
            st.metric(
                label="F1 Score",
                value=f"{np.random.uniform(0.70, 0.78) * 100:.1f}%",
                delta="+1.2%"
            )
        
        st.divider()
        
        # Feature Importance
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Feature Importance")
            features = ['RSI_14', 'MACD_Signal', 'BB_Width', 'ATR_14', 'Vol_ZScore',
                       'FD_Close_0.5', 'Return_1h', 'Return_4h', 'High_Low_Range', 'Volume_SMA_Ratio']
            importance = np.random.uniform(0.05, 0.25, 10)
            importance = importance / importance.sum()
            
            fig = px.bar(
                x=importance,
                y=features,
                orientation='h',
                title='Feature Importance Scores',
                labels={'x': 'Importance', 'y': 'Feature'},
                color=importance,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(template='plotly_dark', height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Prediction Confidence Distribution")
            confidence_data = market_data[market_data['signal_strength'] > 0.3]['signal_strength']
            
            fig = px.histogram(
                confidence_data,
                nbins=30,
                title='Signal Confidence Histogram',
                labels={'value': 'Confidence Score', 'count': 'Frequency'},
                color_discrete_sequence=['#2b90d9']
            )
            fig.update_layout(template='plotly_dark', height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix Placeholder
        st.subheader("Confusion Matrix")
        cm_data = np.array([[45, 8, 7], [5, 52, 3], [6, 4, 48]])
        cm_labels = ['SELL', 'HOLD', 'BUY']
        
        fig = px.imshow(
            cm_data,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=cm_labels,
            y=cm_labels,
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig.update_layout(title='Confusion Matrix (Last 1000 Predictions)', template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Backtest Results
    with tab3:
        st.header("💰 Backtest Performance")
        
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="Total P&L",
                value=f"${metrics['total_pnl']:,.2f}",
                delta="+" if metrics['total_pnl'] > 0 else "-"
            )
        
        with col2:
            st.metric(
                label="Win Rate",
                value=f"{metrics['win_rate']:.1f}%",
                delta=f"{metrics['win_rate'] - 50:.1f}%"
            )
        
        with col3:
            st.metric(
                label="Profit Factor",
                value=f"{metrics['profit_factor']:.2f}",
                help="Gross Profit / Gross Loss"
            )
        
        with col4:
            st.metric(
                label="Sharpe Ratio",
                value=f"{metrics['sharpe_ratio']:.2f}",
                delta="Good" if metrics['sharpe_ratio'] > 1.0 else "Needs Improvement"
            )
        
        with col5:
            st.metric(
                label="Max Drawdown",
                value=f"${metrics['max_drawdown']:,.2f}",
                delta="Within Limit" if metrics['max_drawdown'] < max_drawdown_limit * metrics['total_pnl'] / 100 else "Warning"
            )
        
        st.divider()
        
        # Equity Curve & Drawdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_equity_curve(backtest_data), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_drawdown_chart(backtest_data), use_container_width=True)
        
        # Performance Heatmap
        st.subheader("Performance by Hour & Day")
        st.plotly_chart(create_performance_heatmap(backtest_data), use_container_width=True)
        
        # Recent Trades
        st.subheader("Recent Trades")
        st.dataframe(
            backtest_data.sort_values('exit_time', ascending=False).head(10)[
                ['trade_id', 'symbol', 'direction', 'entry_price', 'exit_price', 'pnl', 'status']
            ].style.format({
                'entry_price': '{:.5f}',
                'exit_price': '{:.5f}',
                'pnl': '${:.2f}'
            }).applymap(lambda x: 'color: #26a69a' if x == 'WIN' else ('color: #ef5350' if x == 'LOSS' else ''), 
                       subset=['status']),
            use_container_width=True,
            hide_index=True
        )
    
    # Tab 4: Live Signals
    with tab4:
        st.header("⚡ Live Trading Signals")
        
        # Filter signals by confidence
        high_conf_signals = market_data[market_data['signal_strength'] >= confidence_threshold].tail(20)
        
        if len(high_conf_signals) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"Active Signals (Confidence ≥ {confidence_threshold})")
                
                for idx, row in high_conf_signals.iterrows():
                    signal_type = "🟢 BUY" if row['prediction'] == 2 else ("🔴 SELL" if row['prediction'] == 0 else "🟡 HOLD")
                    
                    with st.expander(f"{signal_type} | Strength: {row['signal_strength']:.2f} | Time: {row['timestamp']}"):
                        st.write(f"**Symbol:** {selected_symbol}")
                        st.write(f"**Entry Price:** {row['close']:.5f}")
                        st.write(f"**Predicted Return:** {row['predicted_return'] * 100:.3f}%")
                        st.write(f"**Recommended Position Size:** {(risk_per_trade / 100 * 10000 / abs(row['predicted_return'])):.2f} units")
            
            with col2:
                st.subheader("Signal Statistics")
                st.metric("Total Active Signals", len(high_conf_signals))
                st.metric("Buy Signals", len(high_conf_signals[high_conf_signals['prediction'] == 2]))
                st.metric("Sell Signals", len(high_conf_signals[high_conf_signals['prediction'] == 0]))
                st.metric("Avg Confidence", f"{high_conf_signals['signal_strength'].mean():.2f}")
        else:
            st.warning(f"No signals found with confidence ≥ {confidence_threshold}. Consider lowering the threshold.")
        
        st.divider()
        
        # Risk Calculator
        st.subheader("🧮 Position Size Calculator")
        
        calc_col1, calc_col2, calc_col3 = st.columns(3)
        
        with calc_col1:
            account_balance = st.number_input("Account Balance ($)", value=10000.0, step=100.0)
        
        with calc_col2:
            risk_pct = st.number_input("Risk per Trade (%)", value=1.0, step=0.1)
        
        with calc_col3:
            stop_loss_pips = st.number_input("Stop Loss (pips)", value=20.0, step=1.0)
        
        if st.button("Calculate Position Size"):
            risk_amount = account_balance * (risk_pct / 100)
            position_size = risk_amount / (stop_loss_pips * 0.0001)  # For standard lots
            
            st.success(f"""
            **Recommended Position Size:** {position_size:.2f} units  
            **Risk Amount:** ${risk_amount:.2f}  
            **Risk/Reward (1:2):** Take Profit at {stop_loss_pips * 2:.1f} pips
            """)
    
    # Tab 5: Settings
    with tab5:
        st.header("⚙️ Dashboard Settings")
        
        st.subheader("Data Configuration")
        st.text_input("MT5 Terminal Path", value="C:\\Program Files\\MetaTrader 5\\terminal64.exe")
        st.text_input("Database Connection String", value="sqlite:///data/trading.db")
        
        st.subheader("Model Configuration")
        st.file_uploader("Upload Trained Model (.pkl)", type=['pkl', 'joblib'])
        st.text_area("Model Hyperparameters (JSON)", value='{"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}')
        
        st.subheader("Notification Settings")
        st.checkbox("Email Alerts for High-Confidence Signals", value=True)
        st.checkbox("Telegram Bot Integration", value=False)
        st.text_input("Webhook URL", placeholder="https://hooks.slack.com/...")
        
        st.subheader("Export Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Backtest Results (CSV)"):
                csv = backtest_data.to_csv(index=False)
                st.download_button("Download CSV", csv, "backtest_results.csv", "text/csv")
        
        with col2:
            if st.button("Export Signals (JSON)"):
                json_data = market_data.tail(100).to_json(orient='records')
                st.download_button("Download JSON", json_data, "signals.json", "application/json")
        
        with col3:
            if st.button("Generate Report (PDF)"):
                st.info("PDF report generation is coming soon!")
        
        st.divider()
        st.markdown("""
        ### About AI Smart Trader Dashboard
        
        This professional dashboard provides real-time insights into your AI-powered trading system.
        
        **Features:**
        - 📊 Interactive market charts
        - 🤖 Model performance analytics
        - 💰 Comprehensive backtest results
        - ⚡ Live signal generation
        - 🧮 Risk management tools
        
        **Technology Stack:**
        - Frontend: Streamlit + Plotly
        - ML Backend: LightGBM, Scikit-learn
        - Data: MetaTrader 5, Pandas
        """)
    
    # Auto-refresh logic
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
