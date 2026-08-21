---
name: generate-run-commands
description: Generate run commands for AI-Smart-Tred Python trading system. Setup virtual environment, install dependencies, and run trading modules.
---

# Generate Run Commands - AI-Smart-Tred

Help the user set up run commands for the AI-Smart-Tred project - a Python-based machine learning trading system using MetaTrader 5.

## Project Stack Detection

This is a **Python project** with the following characteristics:
- Uses `requirements.txt` for dependencies
- MetaTrader5 integration (Windows-only runtime)
- Machine learning libraries (scikit-learn, pandas, numpy)
- Data storage with Parquet format
- Modular architecture (mt5_client, data_pipeline, features, models, trading)

## Recommended Run Commands

### 1. Setup Environment (Auto-run on worktree creation)
```json
{
  "label": "Setup Python Environment",
  "type": "shell",
  "command": "python -m venv smarttred-env && .\\smarttred-env\\Scripts\\activate && pip install -r requirements.txt",
  "inAgents": true,
  "runOptions": { "runOn": "worktreeCreated" }
}
```

### 2. Activate Virtual Environment
```json
{
  "label": "Activate Venv",
  "type": "shell",
  "command": ".\\smarttred-env\\Scripts\\activate",
  "inAgents": true
}
```

### 3. Run MT5 Connection Test
```json
{
  "label": "Test MT5 Connection",
  "type": "shell",
  "command": ".\\smarttred-env\\Scripts\\python -c \"import MetaTrader5 as mt5; print('MT5 initialized:', mt5.initialize()); mt5.shutdown()\"",
  "inAgents": true
}
```

### 4. Run Data Ingestion
```json
{
  "label": "Run Data Pipeline",
  "type": "shell",
  "command": ".\\smarttred-env\\Scripts\\python -m smarttred.data_pipeline.data_ingestion",
  "inAgents": true
}
```

### 5. Run Feature Engineering
```json
{
  "label": "Calculate Indicators",
  "type": "shell",
  "command": ".\\smarttred-env\\Scripts\\python -m smarttred.features.indicators",
  "inAgents": true
}
```

### 6. Train ML Model
```json
{
  "label": "Train Model",
  "type": "shell",
  "command": ".\\smarttred-env\\Scripts\\python -m smarttred.models.ml_models",
  "inAgents": true
}
```

### 7. Run Trading System (Live/Demo)
```json
{
  "label": "Start Trading Bot",
  "type": "shell",
  "command": ".\\smarttred-env\\Scripts\\python -m smarttred.trading.order_execution",
  "inAgents": true
}
```

### 8. Run All Tests
```json
{
  "label": "Run Tests",
  "type": "shell",
  "command": ".\\smarttred-env\\Scripts\\python -m pytest tests/ -v",
  "inAgents": true
}
```

### 9. Generate Documentation
```json
{
  "label": "Build Docs",
  "type": "shell",
  "command": ".\\smarttred-env\\Scripts\\python -m pdoc smarttred -o docs/",
  "inAgents": true
}
```

## Decision Logic

1. **Check existing `.vscode/tasks.json`** for run commands
2. **If commands exist**: Ask user what to modify
3. **If no commands**: Propose the essential ones:
   - Setup Environment (auto-run)
   - Test MT5 Connection
   - Run Data Pipeline
   - Start Trading Bot

## Platform Considerations

- **Windows**: Primary platform (MT5 requirement)
  - Use `.\smarttred-env\Scripts\activate` and `.\smarttred-env\Scripts\python`
- **Linux/Mac**: Development only (no live MT5)
  - Use `source smarttred-env/bin/activate` and `python`
  - Warn about MT5 unavailability

## Writing tasks.json

Always write to `.vscode/tasks.json` in workspace root. Merge with existing tasks.

After writing, confirm:
- Which commands were added
- How to access them via Run button
- Any platform-specific notes (Windows required for live trading)
