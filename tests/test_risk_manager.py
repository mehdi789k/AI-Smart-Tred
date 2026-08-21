from types import SimpleNamespace

import pytest

from smarttred.trading import RiskManager, kelly_fraction


def test_risk_manager_lot_size() -> None:
    manager = RiskManager(account_balance=10000.0, risk_per_trade=0.01)
    lot_size = manager.calculate_lot_size(entry_price=1.1000, stop_loss_price=1.0950, pip_value=10.0)
    assert lot_size > 0


def test_risk_manager_lot_size_rejects_invalid_input() -> None:
    manager = RiskManager(account_balance=10000.0, risk_per_trade=0.01)
    with pytest.raises(ValueError):
        manager.calculate_lot_size(balance=10000.0, risk_per_trade_pct=1.5, stop_loss_pips=10.0, pip_value=10.0)

    with pytest.raises(ValueError):
        manager.calculate_lot_size(entry_price=1.1000, stop_loss_price=1.1000, pip_value=10.0)


def test_risk_manager_dynamic_sl_tp() -> None:
    manager = RiskManager(account_balance=10000.0, risk_per_trade=0.01)
    symbol_info = SimpleNamespace(bid=100.0, ask=101.0, digits=5)
    sl, tp = manager.calculate_dynamic_sl_tp(symbol_info, atr_value=10.0, sl_atr_multiplier=1.5, tp_atr_multiplier=2.5)
    assert sl < tp
    assert sl == pytest.approx(85.5)
    assert tp == pytest.approx(125.5)


def test_kelly_fraction() -> None:
    value = kelly_fraction(win_rate=0.55, avg_win=1.2, avg_loss=1.0)
    assert value > 0


def test_kelly_fraction_invalid_input() -> None:
    with pytest.raises(ValueError):
        kelly_fraction(win_rate=1.5, avg_win=1.2, avg_loss=1.0)

    with pytest.raises(ValueError):
        kelly_fraction(win_rate=0.5, avg_win=1.2, avg_loss=0.0)
