"""Engine/runner support for a train/holdout split."""
import numpy as np
import pandas as pd
import pytest

import src.backtest.engine as eng_mod
from src.backtest.engine import BacktestEngine


def _df(n=200):
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    close = np.full(n, 100.0)
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1.0}, index=idx)


@pytest.fixture
def fake_strategy(monkeypatch):
    """Buy every 10th bar; stop 5 below, target 2 above (hit next bar)."""
    def detect(df, params):
        df = df.copy()
        df["sig"] = None
        df.iloc[::10, df.columns.get_loc("sig")] = "buy"
        return df

    def levels(df, i, params):
        c = df.iloc[i]["close"]
        return {"entry": c, "stop_loss": c - 5, "target": c + 0.5,
                "risk": 5, "risk_reward": 0.1}

    monkeypatch.setitem(eng_mod.STRATEGY_REGISTRY, "fake", (detect, "sig", levels))
    monkeypatch.setattr(eng_mod, "load_active_filters", lambda s: [])
    monkeypatch.setattr(eng_mod, "extract_features", lambda df, i: {"x": 1.0})
    return {"fake": {"min_risk_reward": 0.0}, "risk": {"max_position_pct": 0.05}}


def _run(params, **kw):
    df = _df()
    e = BacktestEngine("fake", params, 1_000_000,
                       costs={"fee_pct": 0.0, "slippage_pct": 0.0}, **kw.pop("ctor", {}))
    return e, e.run(df, "BTC/IDR", "1h", **kw)


def test_trade_start_blocks_earlier_entries(fake_strategy):
    df = _df()
    start = df.index[150]
    e, res = _run(fake_strategy, trade_start=start)
    assert e.trades
    assert all(pd.Timestamp(t["entry_time"]) >= start for t in e.trades)


def test_without_trade_start_trades_from_warmup(fake_strategy):
    e, _ = _run(fake_strategy)
    assert pd.Timestamp(e.trades[0]["entry_time"]) < _df().index[100]


def test_explicit_filters_override_db(fake_strategy):
    skip_all = [{"feature": "x", "operator": ">=", "threshold": 0.0,
                 "rule_type": "skip", "effect_size": 1.0, "rationale": ""}]
    e, res = _run(fake_strategy, ctor={"filters": skip_all})
    assert e.trades == []
    assert res["filtered_trades"] > 0
