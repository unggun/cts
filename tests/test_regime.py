"""Higher-timeframe regime features: correct, and never from the future."""
import numpy as np
import pandas as pd
import pytest

from src.indicators.features import extract_features
from src.indicators.regime import (
    REGIME_COLUMNS, _as_of, add_regime_columns, adx, resample, supertrend_up,
)


def _ohlcv(close, start="2026-01-01"):
    close = np.asarray(close, dtype=float)
    idx = pd.date_range(start, periods=len(close), freq="1h")
    return pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998,
                         "close": close, "volume": 1.0}, index=idx)


def _walk(n, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))


def test_supertrend_follows_trend():
    up = _ohlcv(np.linspace(100, 200, 300))
    down = _ohlcv(np.linspace(200, 100, 300))
    assert np.nanmin(supertrend_up(up)[20:]) == 1.0
    assert np.nanmax(supertrend_up(down)[20:]) == 0.0
    flip = _ohlcv(np.concatenate([np.linspace(100, 200, 150), np.linspace(200, 100, 150)]))
    d = supertrend_up(flip)
    assert d[140] == 1.0 and d[-1] == 0.0
    assert np.isnan(d[:9]).all()                       # ATR warm-up


def test_adx_high_in_trend_low_in_chop():
    trend = _ohlcv(np.linspace(100, 300, 300))
    chop = _ohlcv(100 + np.tile([0.0, 1.0], 150))
    assert adx(trend)[-1] > 40
    assert adx(chop)[-1] < 20


def test_higher_timeframe_value_only_after_its_candle_closes():
    df = _ohlcv(np.arange(24, dtype=float) + 100)
    h4 = resample(df, "4h")
    values = np.arange(len(h4), dtype=float)            # 4h candle k has value k
    got = _as_of(df, h4, "4h", values, "1h")
    # 1h bar opened 02:00 closes 03:00: no 4h candle has closed yet.
    assert np.isnan(got[2])
    # Bar opened 03:00 closes 04:00, together with the 00:00-04:00 candle.
    assert got[3] == 0 and got[6] == 0 and got[7] == 1


def test_truncation_never_changes_past_values():
    """No look-ahead: values at bar k are the same whether or not later bars exist."""
    n = 24 * 40
    pair, btc = _ohlcv(_walk(n, 1)), _ohlcv(_walk(n, 2, drift=0.0005))
    full = add_regime_columns(pair, btc)
    assert full[REGIME_COLUMNS].iloc[-1].notna().all()
    for k in (24 * 15 + 3, 24 * 20 + 11, 24 * 33 + 23, n - 2):
        part = add_regime_columns(pair.iloc[:k + 1], btc.iloc[:k + 1])
        pd.testing.assert_series_equal(part[REGIME_COLUMNS].iloc[k],
                                       full[REGIME_COLUMNS].iloc[k], check_names=False)


def test_forming_candle_does_not_leak_into_closed_bar():
    n = 24 * 30
    pair, btc = _ohlcv(_walk(n, 3)), _ohlcv(_walk(n, 4))
    base = add_regime_columns(pair, btc)
    wild = pair.copy()
    wild.iloc[-1, :4] = wild.iloc[-1, :4] * 3            # forming bar spikes
    btc_wild = btc.copy()
    btc_wild.iloc[-1, :4] = btc_wild.iloc[-1, :4] * 3
    spiked = add_regime_columns(wild, btc_wild)
    pd.testing.assert_frame_equal(spiked[REGIME_COLUMNS].iloc[:-1],
                                  base[REGIME_COLUMNS].iloc[:-1])


def test_btc_return_24h():
    btc = _ohlcv(np.concatenate([np.full(24, 100.0), np.full(48, 110.0)]))
    pair = _ohlcv(np.full(72, 5.0))
    out = add_regime_columns(pair, btc)
    assert np.isnan(out["btc_return_24h"].iloc[23])
    assert out["btc_return_24h"].iloc[24] == pytest.approx(0.10)
    assert out["btc_return_24h"].iloc[47] == pytest.approx(0.10)
    assert out["btc_return_24h"].iloc[48] == pytest.approx(0.0)


def test_without_btc_data_btc_features_are_missing():
    out = add_regime_columns(_ohlcv(_walk(24 * 30, 5)))
    assert out["btc_st_4h_up"].isna().all() and out["st_4h_up"].notna().any()
    feats = extract_features(out, len(out) - 1)
    assert "st_4h_up" in feats and "btc_st_4h_up" not in feats


def test_extract_features_includes_regime():
    n = 24 * 30
    out = add_regime_columns(_ohlcv(_walk(n, 6)), _ohlcv(_walk(n, 7)))
    feats = extract_features(out, n - 1)
    for col in REGIME_COLUMNS:
        assert isinstance(feats[col], float)
    assert feats["st_1d_up"] in (0.0, 1.0)


def test_engine_adds_regime_features_to_trades(monkeypatch):
    import src.backtest.engine as eng_mod
    from src.backtest.engine import BacktestEngine
    monkeypatch.setitem(eng_mod.STRATEGY_REGISTRY, "every", (
        lambda df, p: df.assign(sig=["buy" if i % 50 == 0 else None for i in range(len(df))]),
        "sig",
        lambda df, i, p: {"entry": df["close"].iat[i], "stop_loss": df["close"].iat[i] * 0.9,
                          "target": df["close"].iat[i] * 1.001, "risk": 1.0,
                          "risk_reward": 0.1}))
    df = _ohlcv(_walk(24 * 30, 8))
    e = BacktestEngine("every", {"every": {"min_risk_reward": 0.0}}, 1_000_000,
                       costs={"fee_pct": 0.0, "slippage_pct": 0.0}, filters=[])
    e.run(df, "BTC/IDR", "1h")                     # BTC is its own regime pair
    late = [t for t in e.trades if pd.Timestamp(t["entry_time"]) > df.index[24 * 15]]
    assert late and all("btc_st_4h_up" in t["features"] and "st_1d_up" in t["features"]
                        for t in late)
