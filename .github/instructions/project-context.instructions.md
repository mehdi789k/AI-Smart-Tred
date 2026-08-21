# AI-Smart-Tred Project Context

## Overview

AI-Smart-Tred is an intelligent trading system based on machine learning and MetaTrader 5 integration. The project enables automated trading decisions using ML models trained on historical market data.

## Technology Stack

### Core Technologies
- **Language**: Python 3.11+
- **Trading Platform**: MetaTrader 5 (MT5)
- **ML Libraries**: scikit-learn, XGBoost, LightGBM
- **Data Processing**: pandas, numpy
- **Storage**: Parquet (via pyarrow), SQLite/PostgreSQL
- **Technical Analysis**: ta-lib, pandas-ta

### Project Structure

```
smarttred/
├── mt5_client/          # MT5 connectivity layer
│   ├── connection.py    # MT5 initialization & API wrapper
│   └── __init__.py
├── data_pipeline/       # Data ingestion & storage
│   ├── data_ingestion.py
│   └── storage.py
├── features/            # Feature engineering
│   ├── indicators.py    # Technical indicators
│   └── feature_engineering.py
├── models/              # ML models
│   ├── ml_models.py     # Model definitions
│   └── model_training.py
├── trading/             # Trading execution
│   ├── signal_generator.py
│   └── order_execution.py
└── notebooks/           # Jupyter experiments

data/
├── raw/                 # Raw market data
│   ├── crypto/
│   └── forex/
├── features/            # Processed features
└── trades/              # Trade logs
```

## Development Guidelines

### Language & Documentation
- Code comments: English or Persian (Farsi)
- Docstrings: Persian preferred for team understanding
- Commit messages: Conventional Commits in English
- README and docs: Persian (Farsi)

### Code Quality Standards
- Follow PEP 8 style guide
- Use type hints for all functions
- Write unit tests for critical components
- Log all trading operations
- Never commit credentials or API keys

### Trading System Safety Rules
1. **Always validate** MT5 connection before operations
2. **Implement risk management** in every trade
3. **Use demo accounts** for testing
4. **Log all decisions** for audit trail
5. **Never hardcode** trading parameters
6. **Add circuit breakers** for loss limits

### Git Workflow
```bash
# Feature development
git checkout -b feature/description
git commit -m "feat(scope): description"
git push origin feature/description

# Bug fixes
git checkout -b fix/description
git commit -m "fix(scope): description"
```

## Key Components

### 1. MT5 Client
- Handles connection to MetaTrader 5 terminal
- Provides unified API for market data access
- Manages order execution and position tracking

### 2. Data Pipeline
- Ingests historical and real-time data from MT5
- Cleans and validates financial time-series
- Stores data efficiently in Parquet format

### 3. Feature Engineering
- Calculates technical indicators (RSI, MACD, EMA, etc.)
- Creates custom features for ML models
- Handles look-ahead bias prevention

### 4. ML Models
- Binary classification (up/down movement)
- Regression (price prediction)
- Ensemble methods for robustness
- Time-series cross-validation

### 5. Trading Engine
- Generates trading signals from model predictions
- Executes orders through MT5
- Manages position sizing and risk
- Records all trades for analysis

## Testing Strategy

- Unit tests for individual functions
- Integration tests for MT5 connection (mocked in CI)
- Backtesting on historical data
- Paper trading validation before live deployment

## Deployment Considerations

- Windows required for live MT5 trading
- Linux/Mac suitable for development and backtesting
- Docker containers for ML training workloads
- Separate environments: dev, staging, production

## Security Notes

- Store API keys in environment variables
- Encrypt sensitive configuration files
- Use separate accounts for demo/live trading
- Implement rate limiting for API calls
