from __future__ import annotations

from types import SimpleNamespace

import pytest

from smarttred.trading.live_execution import MT5OrderExecutor


class FakeMT5:
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 1
    ORDER_FILLING_IOC = 2
    ORDER_FILLING_RETURN = 3
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_DONE_PARTIAL = 10010
    TRADE_RETCODE_REJECT = 10008

    @staticmethod
    def initialize(**kwargs):
        return True

    @staticmethod
    def last_error():
        return "fake"

    @staticmethod
    def shutdown():
        return True

    @staticmethod
    def symbol_info(symbol):
        return SimpleNamespace(filling_mode=FakeMT5.ORDER_FILLING_RETURN, point=0.0001, volume_min=0.01, volume_max=50.0, volume_step=0.01, trade_contract_size=100000.0)

    @staticmethod
    def symbol_info_tick(symbol):
        return SimpleNamespace(bid=1.1000, ask=1.1010)

    @staticmethod
    def positions_get(symbol=None):
        return [SimpleNamespace(symbol=symbol)]

    @staticmethod
    def order_send(request):
        return SimpleNamespace(retcode=FakeMT5.TRADE_RETCODE_DONE, comment="ok")

    @staticmethod
    def account_info():
        return SimpleNamespace(balance=10000.0)


@pytest.fixture
def fake_executor(monkeypatch):
    import smarttred.trading.live_execution as live_module

    monkeypatch.setattr(live_module, "mt5", FakeMT5)
    executor = MT5OrderExecutor(settings=None)
    executor.connected = True
    return executor


def test_calculate_risk_lot_size(fake_executor):
    lot_size = fake_executor.calculate_risk_lot_size(
        entry_price=1.1000,
        stop_loss_price=1.0950,
        account_balance=10000.0,
        risk_per_trade=0.01,
        pip_value=10.0,
        pip_size=0.0001,
    )
    assert lot_size > 0


def test_get_open_positions(fake_executor):
    positions = fake_executor.get_open_positions("XAUUSD_l")
    assert len(positions) == 1
    assert positions[0].symbol == "XAUUSD_l"


def test_get_filling_mode(fake_executor):
    assert fake_executor.get_filling_mode("XAUUSD_l") == FakeMT5.ORDER_FILLING_RETURN


def test_execute_trade_rejected_order(fake_executor, monkeypatch):
    def fake_send(_request):
        return SimpleNamespace(retcode=FakeMT5.TRADE_RETCODE_REJECT, comment="rejected")

    monkeypatch.setattr(FakeMT5, "order_send", staticmethod(fake_send))
    result = fake_executor.execute_trade("XAUUSD_l", 1, 1.0950, 1.1100, 0.10)
    assert result is False


def test_place_market_order(fake_executor):
    result = fake_executor.place_market_order(
        symbol="XAUUSD_l",
        side=1,
        lot_size=0.10,
        stop_loss=1.0950,
        take_profit=1.1100,
        comment="test",
    )
    assert result.retcode == FakeMT5.TRADE_RETCODE_DONE


def test_place_risk_managed_order(fake_executor):
    result = fake_executor.place_risk_managed_order(
        symbol="XAUUSD_l",
        signal=1,
        entry_price=1.1000,
        stop_loss_pips=20.0,
        take_profit_pips=40.0,
        account_balance=10000.0,
        risk_per_trade=0.01,
    )
    assert result.retcode == FakeMT5.TRADE_RETCODE_DONE


def test_connect_raises_on_failed_init(monkeypatch):
    import smarttred.trading.live_execution as live_module
    from smarttred.config.settings import Settings

    class FailingMT5:
        @staticmethod
        def initialize(**kwargs):
            return False

        @staticmethod
        def last_error():
            return "bad-init"

    monkeypatch.setattr(live_module, "mt5", FailingMT5)
    
    # Provide mock settings to avoid validation error
    mock_settings = Settings(
        mt5_terminal_path="/fake/path",
        mt5_login=12345,
        mt5_password="fake_password",
        mt5_server="fake_server",
    )
    executor = MT5OrderExecutor(settings=mock_settings)
    with pytest.raises(ConnectionError):
        executor.connect()
