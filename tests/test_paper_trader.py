"""End-to-end checks of PaperTrader._process_pair with a fake exchange."""
import os
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import src.execution.paper_trader as pt
from src.data.database import get_connection, init_db


@pytest.fixture
def db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    monkeypatch.setattr(pt, "get_connection", lambda: get_connection(path))
    yield path
    os.unlink(path)


def _candles(n=80, price=100.0):
    """n hourly candles ending with the currently-forming one."""
    now = datetime.now(timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    idx = pd.date_range(end=now, periods=n, freq="1h")
    df = pd.DataFrame({"open": price, "high": price + 1, "low": price - 1,
                       "close": price, "volume": 1.0}, index=idx)
    return df


def _trader(df, signal_rows, monkeypatch):
    def detect(frame, params):
        frame = frame.copy()
        frame["sig"] = None
        for r in signal_rows:
            frame.iloc[r, frame.columns.get_loc("sig")] = "buy"
        return frame

    def levels(frame, i, params):
        close = frame.iloc[i]["close"]
        return {"entry": close, "stop_loss": close - 10, "target": close + 40,
                "risk": 10, "risk_reward": 4.0}

    monkeypatch.setitem(pt.STRATEGY_REGISTRY, "fake", (detect, "sig", levels))
    monkeypatch.setattr(pt, "load_ohlcv", lambda conn, pair, tf: df)
    monkeypatch.setattr(pt, "upsert_ohlcv", lambda *a, **k: None)
    monkeypatch.setattr(pt, "extract_features", lambda frame, i: {})

    t = object.__new__(pt.PaperTrader)
    t.strategy = "fake"
    t.config = {}
    t.params = {"fake": {"min_risk_reward": 1.5, "trailing_atr_multiplier": 3.0},
                "risk": {"max_position_pct": 0.05, "max_positions": 3}}
    t.filters = []
    t.positions = {}
    t._last_signal_bar = {}
    t.costs = {"fee_pct": 0.001, "slippage_pct": 0.0}
    t.capital = t.initial_capital = 10_000_000
    t.exchange = type("X", (), {"fetch_ohlcv": lambda self, *a, **k: []})()
    t._notify = lambda msg: None
    return t


def test_signal_on_forming_candle_is_ignored(db, monkeypatch):
    df = _candles()
    t = _trader(df, signal_rows=[-1], monkeypatch=monkeypatch)
    t._process_pair("BTC/IDR", "1h")
    assert t.positions == {}


def test_signal_on_closed_candle_opens_once(db, monkeypatch):
    df = _candles()
    df.iloc[-1, df.columns.get_loc("close")] = 102.0  # current price moved
    t = _trader(df, signal_rows=[-2], monkeypatch=monkeypatch)
    t._process_pair("BTC/IDR", "1h")
    pos = t.positions["BTC/IDR"]
    assert pos["entry_price"] == 102.0      # filled at current price
    assert pos["stop_loss"] == 90.0         # stop from the signal bar
    assert pos["id"] is not None

    # Close it, then poll again within the same hour: no re-entry on the same bar.
    t._close_position(get_connection(db), "BTC/IDR", 103.0, "target")
    t._process_pair("BTC/IDR", "1h")
    assert "BTC/IDR" not in t.positions


def test_trailing_stop_ratchets_and_persists(db, monkeypatch):
    df = _candles()
    t = _trader(df, signal_rows=[-2], monkeypatch=monkeypatch)
    t._process_pair("BTC/IDR", "1h")
    pos_id = t.positions["BTC/IDR"]["id"]

    df.iloc[-1, df.columns.get_loc("high")] = 130.0   # big move up, ATR ~2
    t._process_pair("BTC/IDR", "1h")
    new_stop = t.positions["BTC/IDR"]["stop_loss"]
    assert new_stop > 90.0
    row = get_connection(db).execute(
        "SELECT stop_loss FROM trade_journal WHERE id=?", (pos_id,)).fetchone()
    assert row["stop_loss"] == pytest.approx(new_stop)


def test_close_position_charges_costs(db, monkeypatch):
    t = _trader(_candles(), signal_rows=[-2], monkeypatch=monkeypatch)
    t._process_pair("BTC/IDR", "1h")
    t._close_position(get_connection(db), "BTC/IDR", 100.0, "manual")  # flat exit
    row = get_connection(db).execute(
        "SELECT pnl_pct FROM trade_journal WHERE exit_time IS NOT NULL").fetchone()
    assert row["pnl_pct"] < 0
