"""Trading costs (fees + slippage) must be charged on both sides."""
import pandas as pd
import pytest

from src.backtest.costs import net_pnl, trading_costs
from src.backtest.engine import BacktestEngine


def test_net_pnl_zero_costs_matches_gross():
    pnl_abs, pnl_pct = net_pnl(100.0, 110.0, 2.0, fee_pct=0.0, slippage_pct=0.0)
    assert pnl_abs == pytest.approx(20.0)
    assert pnl_pct == pytest.approx(0.10)


def test_net_pnl_flat_trade_loses_round_trip_costs():
    pnl_abs, pnl_pct = net_pnl(100.0, 100.0, 1.0, fee_pct=0.001, slippage_pct=0.0005)
    # buy 100.05 * 1.001, sell 99.95 * 0.999
    expected = 99.95 * 0.999 - 100.05 * 1.001
    assert pnl_abs == pytest.approx(expected)
    assert pnl_pct == pytest.approx(expected / (100.05 * 1.001))
    assert pnl_pct < -0.002


def test_trading_costs_defaults_and_overrides():
    assert trading_costs({}) == {"fee_pct": 0.001, "slippage_pct": 0.0005}
    cfg = {"costs": {"fee_pct": 0.002, "slippage_pct": 0.0}}
    assert trading_costs(cfg) == {"fee_pct": 0.002, "slippage_pct": 0.0}


def test_engine_close_position_charges_costs(monkeypatch):
    monkeypatch.setattr("src.backtest.engine.load_active_filters", lambda s: [])
    eng = BacktestEngine("darvas", {}, initial_capital=1000.0,
                         costs={"fee_pct": 0.001, "slippage_pct": 0.0})
    df = pd.DataFrame({"close": [100.0, 100.0]},
                      index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
    eng.position = {"pair": "BTC/IDR", "timeframe": "1h", "entry_time": "t0",
                    "entry_price": 100.0, "stop_loss": 90.0, "target": 120.0,
                    "position_size": 1.0, "features": {}}
    eng._close_position(df, 1, "end_of_data")
    trade = eng.trades[0]
    assert trade["pnl_pct"] < 0
    assert eng.capital == pytest.approx(1000.0 + trade["pnl_absolute"])
