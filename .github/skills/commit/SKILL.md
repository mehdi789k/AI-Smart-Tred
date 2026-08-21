---
name: commit
description: Commit changes for AI-Smart-Tred project with conventional commits following Persian/Farsi-friendly messages. Handles Python trading system code, MT5 integration, and ML models.
---

# Commit Changes - AI-Smart-Tred

Help the user commit code changes for the AI-Smart-Tred trading system with well-crafted commit messages in English (with optional Persian translations for complex concepts).

## Project-Specific Guidelines

### Commit Message Convention

This project uses **Conventional Commits** format:
```
<type>(<scope>): <subject>

<body> (optional)

<footer> (optional)
```

**Types:**
- `feat`: New feature (e.g., new indicator, ML model, trading strategy)
- `fix`: Bug fix (e.g., MT5 connection issue, data processing error)
- `docs`: Documentation changes (README, code comments in Persian)
- `style`: Code style changes (formatting, missing semicolons)
- `refactor`: Code refactoring without behavior change
- `test`: Adding or updating tests
- `chore`: Build process, dependencies, tool updates
- `perf`: Performance improvements

**Scopes:**
- `mt5`: MetaTrader5 integration
- `data`: Data pipeline and storage
- `features`: Technical indicators and feature engineering
- `models`: Machine learning models
- `trading`: Signal generation and order execution
- `config`: Configuration files
- `deps`: Dependencies

### Example Commit Messages

```
feat(mt5): add real-time tick data streaming

Implement asynchronous tick data reception from MT5
with proper buffering and backpressure handling

Refs: #12
```

```
fix(data): handle missing candles in historical data

Add forward-fill logic for gaps in MT5 historical data
Prevents NaN values in feature calculation

Fixes: #8
```

```
docs(features): add Persian documentation for RSI indicator

توضیحات فارسی برای اندیکاتور RSI اضافه شد
شامل فرمول محاسبه و پارامترهای قابل تنظیم
```

## Workflow

### 1. Check Repository Status
```bash
git status --short
```
- If no changes: inform user and stop
- If staged changes: use those
- If only unstaged: stage with `git add -A`

### 2. Analyze Changes
Review the diff to understand:
- Which modules are affected (mt5_client, data_pipeline, features, models, trading)
- Whether it's a feature, fix, refactor, etc.
- Any related issues or tickets

### 3. Generate Commit Message
Create a message following the convention above:
- Subject line ≤ 72 characters
- Body explains "why" not just "what"
- Reference issues when applicable
- Use Persian in body/docs for Iranian team members when helpful

### 4. Execute Commit
```bash
git commit -m "<subject>" -m "<body>"
```

### 5. Confirm
Show the result:
```bash
git log --oneline -1
git status --short
```

## Special Considerations for Trading Systems

- **Never commit**: API keys, credentials, or sensitive configuration
- **Always verify**: No hardcoded trading parameters that should be configurable
- **Check for**: Debug logging that should be removed before production
- **Warn about**: Changes to risk management logic without tests

## Safety Rules

- Never amend existing commits without asking
- Never force-push without explicit approval
- Never skip pre-commit hooks (`--no-verify`)
- Always sign commits if GPG is configured
- Ask before committing generated files or large data samples
