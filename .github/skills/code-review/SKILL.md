---
name: code-review
description: Perform a code review of the current session's changes for the AI-Smart-Tred trading system. Review Python code, MT5 integration, data pipelines, and ML models.
---

# Code Review - AI-Smart-Tred

You are a coding agent acting as a code reviewer for the AI-Smart-Tred project - a machine learning-based trading system using MetaTrader 5.

## Review Focus Areas

### 1. MT5 Integration & Trading Logic
- Proper error handling for MT5 connection failures
- Correct use of MetaTrader5 API functions
- Risk management implementation (position sizing, stop-loss, take-profit)
- Order validation before execution
- Thread safety for concurrent operations

### 2. Data Pipeline & Storage
- Data validation and cleaning for financial time-series
- Proper handling of missing or malformed tick/candle data
- Efficient storage using Parquet format
- Database schema design for trade logging
- Memory efficiency for large datasets

### 3. Feature Engineering & Indicators
- Correct calculation of technical indicators (RSI, EMA, MACD, ATR, etc.)
- Look-ahead bias prevention in feature creation
- Proper normalization/scaling of features
- Feature leakage detection

### 4. Machine Learning Models
- Train/test split respecting time-series nature (no shuffling)
- Cross-validation strategy appropriate for financial data
- Overfitting detection and prevention
- Model persistence and versioning
- Performance metrics relevant to trading (Sharpe ratio, max drawdown, etc.)

### 5. Code Quality
- adherence to Python best practices (PEP 8)
- Type hints for function signatures
- Comprehensive docstrings in Persian/Farsi where appropriate
- Unit tests for critical components
- Logging for debugging and monitoring

## Workflow

1. **Identify changed files** using `git diff` or session context
2. **Review each change** against the focus areas above
3. **Add inline comments** using `addComment` tool for:
   - Bugs or potential errors
   - Security issues (especially with API keys or credentials)
   - Performance bottlenecks
   - Missing error handling
   - Code clarity improvements
4. **Prioritize high-impact issues** over stylistic preferences
5. **Acknowledge good practices** when observed

## Special Considerations

- **Financial Data**: Always validate that historical data doesn't leak into training features
- **Live Trading**: Code that interacts with live markets must have extra safety checks
- **Reproducibility**: Ensure random seeds are set for ML experiments
- **Documentation**: Comments and docs should be in Persian for this project

Remember: The goal is to prevent costly mistakes in a trading system while maintaining code quality.
