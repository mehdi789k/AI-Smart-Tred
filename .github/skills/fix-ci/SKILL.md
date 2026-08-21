---
name: fix-ci
description: Diagnose and fix CI/CD issues for AI-Smart-Tred Python trading system. Handle test failures, linting errors, and build problems.
---

# Fix CI - AI-Smart-Tred

Help diagnose and resolve CI/CD pipeline failures for the AI-Smart-Tred project - a Python-based machine learning trading system.

## Common CI Issues & Solutions

### 1. Python Environment Setup Failures

**Symptoms:**
- `ModuleNotFoundError` for MetaTrader5, pandas, numpy, etc.
- Virtual environment creation errors
- pip install failures

**Solutions:**
```bash
# Ensure requirements.txt is complete
pip install --upgrade pip
pip install -r requirements.txt

# For MT5 on Linux (development only)
pip install MetaTrader5 || echo "MT5 requires Windows for live trading"
```

### 2. Test Failures

**Types of failures:**
- Unit test assertion errors
- MT5 connection timeouts (expected in CI without MT5)
- Data file not found errors

**Approach:**
1. Run tests locally to reproduce
2. Check if tests are properly mocked (MT5 should be mocked in CI)
3. Verify test data files exist in repository
4. Add `@pytest.mark.skipif` for platform-specific tests

### 3. Linting Errors (flake8, pylint, black)

**Common issues:**
- PEP 8 violations
- Missing type hints
- Line length > 79 characters
- Import order issues

**Fix commands:**
```bash
# Auto-format with black
black smarttred/

# Sort imports with isort
isort smarttred/

# Check linting
flake8 smarttred/
pylint smarttred/
```

### 4. Type Checking Errors (mypy)

**Common issues:**
- Missing type annotations
- Type mismatches in function signatures
- Union type handling

**Fix approach:**
```bash
# Run mypy to see errors
mypy smarttred/

# Add proper type hints
from typing import Optional, List, Dict, Union
import pandas as pd
import numpy as np
```

### 5. Data Pipeline Issues

**Common issues:**
- Parquet file schema changes
- Database migration failures
- Missing data directories

**Solutions:**
- Ensure data directories are created programmatically
- Use relative paths from project root
- Add schema validation for Parquet files

### 6. ML Model Training Failures

**Common issues:**
- Memory errors with large datasets
- Random seed inconsistencies
- Model serialization errors

**Solutions:**
```python
# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Use joblib for model persistence
import joblib
joblib.dump(model, 'model.pkl')
```

## Debugging Workflow

### 1. Analyze CI Logs
- Identify the failing step (install, lint, test, build)
- Extract specific error messages
- Check if it's a new failure or recurring

### 2. Reproduce Locally
```bash
# Create clean environment
python -m venv test-env
source test-env/bin/activate  # Windows: .\test-env\Scripts\activate
pip install -r requirements.txt
pip install pytest flake8 black mypy

# Run the failing step
pytest tests/ -v
flake8 smarttred/
```

### 3. Fix and Validate
- Make minimal changes to fix the issue
- Run all tests locally
- Ensure no new warnings introduced

### 4. Commit and Push
```bash
git add -A
git commit -m "fix(ci): resolve [specific issue]"
git push
```

## Platform-Specific Considerations

### Windows (Primary)
- MT5 works natively
- Path separators: `\`
- Virtual env: `.\venv\Scripts\activate`

### Linux/Mac (CI/Development)
- MT5 unavailable (mock required)
- Path separators: `/`
- Virtual env: `source venv/bin/activate`
- Use `pytest-mock` for MT5 simulation

## CI Configuration Best Practices

```yaml
# Example GitHub Actions structure
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-mock
      - name: Lint
        run: |
          flake8 smarttred/
          black --check smarttred/
      - name: Test
        run: |
          pytest tests/ -v --cov=smarttred
        env:
          MT5_MOCKED: "true"  # Signal to use mocks
```

## When to Ask for Help

- CI infrastructure changes required
- Third-party service integration issues
- Performance regression investigations
- Security vulnerability fixes
