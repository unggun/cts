"""Paper trading must decide on closed candles and trail stops like the backtest."""
from datetime import datetime

import pandas as pd
import pytest

from src.backtest.exits import trail_stop
from src.data.candles import last_closed_index, timeframe_seconds


def _df(times):
    idx = pd.to_datetime(times)
    return pd.DataFrame({"close": range(len(idx))}, index=idx)


def test_timeframe_seconds():
    assert timeframe_seconds("1h") == 3600
    assert timeframe_seconds("4h") == 14400
    assert timeframe_seconds("15m") == 900
    assert timeframe_seconds("1d") == 86400


def test_last_closed_index_skips_forming_candle():
    df = _df(["2026-11-07 00:00", "2026-11-07 01:00"])
    now = datetime(2026, 11, 7, 1, 21)  # 01:00 bar still open
    assert last_closed_index(df, "1h", now) == 0


def test_last_closed_index_when_last_bar_closed():
    df = _df(["2026-11-07 00:00", "2026-11-07 01:00"])
    now = datetime(2026, 11, 7, 2, 0, 5)
    assert last_closed_index(df, "1h", now) == 1


def test_last_closed_index_none_when_nothing_closed():
    df = _df(["2026-11-07 01:00"])
    assert last_closed_index(df, "1h", datetime(2026, 11, 7, 1, 30)) is None


def test_trail_stop_inactive_until_one_atr_profit():
    # highest 105 is not > entry 100 + atr 10 -> keep stop
    assert trail_stop(100.0, 90.0, 105.0, atr=10.0, trail_mult=3.0) == 90.0


def test_trail_stop_ratchets_up_never_down():
    # highest 150 > 110 -> trailing = 150 - 30 = 120
    assert trail_stop(100.0, 90.0, 150.0, atr=10.0, trail_mult=3.0) == pytest.approx(120.0)
    # a lower computed trail never lowers an existing stop
    assert trail_stop(100.0, 125.0, 150.0, atr=10.0, trail_mult=3.0) == 125.0
