import pandas as pd
import numpy as np
import pytest
from src.indicators.rsi_vwap import detect_rsi_vwap, calculate_rsi_vwap_levels


def _make_df(n=200, seed=42):
    """Create a synthetic OHLCV DataFrame for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="1h")
    close = 50000 + np.cumsum(rng.randn(n) * 100)
    high = close + rng.uniform(50, 200, n)
    low = close - rng.uniform(50, 200, n)
    opens = close + rng.randn(n) * 50
    volume = rng.uniform(1, 10, n) * 1e6
    return pd.DataFrame({
        "open": opens, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)


def test_detect_rsi_vwap_adds_columns():
    df = _make_df()
    result = detect_rsi_vwap(df)
    assert "rsi_14" in result.columns
    assert "vwap" in result.columns
    assert "rsi_vwap_signal" in result.columns


def test_detect_rsi_vwap_signals_are_valid():
    df = _make_df()
    result = detect_rsi_vwap(df)
    valid_values = {None, "buy"}
    signals = set(result["rsi_vwap_signal"].dropna().unique()) | {None}
    assert signals.issubset(valid_values)


def test_calculate_rsi_vwap_levels_returns_required_keys():
    df = _make_df()
    result = detect_rsi_vwap(df)
    signal_rows = result[result["rsi_vwap_signal"] == "buy"]
    idx = signal_rows.index[0] if len(signal_rows) > 0 else 100
    idx_pos = result.index.get_loc(idx)
    levels = calculate_rsi_vwap_levels(result, idx_pos)
    assert "entry" in levels
    assert "stop_loss" in levels
    assert "target" in levels
    assert "risk" in levels
    assert "risk_reward" in levels
    assert levels["risk"] > 0


def test_rsi_vwap_custom_params():
    df = _make_df()
    params = {"rsi_period": 10, "oversold_threshold": 25, "exit_rsi": 55}
    result = detect_rsi_vwap(df, params)
    assert "rsi_vwap_signal" in result.columns
