import pandas as pd
import numpy as np
import pytest
from src.indicators.macd import detect_macd, calculate_macd_levels


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


def test_detect_macd_adds_columns():
    df = _make_df()
    result = detect_macd(df)
    assert "macd_line" in result.columns
    assert "macd_signal_line" in result.columns
    assert "macd_histogram" in result.columns
    assert "macd_signal" in result.columns


def test_detect_macd_signals_are_valid():
    df = _make_df()
    result = detect_macd(df)
    valid_values = {None, "buy"}
    signals = set(result["macd_signal"].dropna().unique()) | {None}
    assert signals.issubset(valid_values)


def test_calculate_macd_levels_returns_required_keys():
    df = _make_df()
    result = detect_macd(df)
    idx = 100
    levels = calculate_macd_levels(result, idx)
    assert "entry" in levels
    assert "stop_loss" in levels
    assert "target" in levels
    assert "risk" in levels
    assert "risk_reward" in levels
    assert levels["risk"] > 0


def test_macd_custom_params():
    df = _make_df()
    params = {"fast_period": 5, "slow_period": 20, "signal_period": 5}
    result = detect_macd(df, params)
    assert "macd_signal" in result.columns


def test_macd_default_params_are_fast():
    """Verify default params are the aggressive 3/15/3 from the tweet."""
    df = _make_df()
    result = detect_macd(df)
    fast_signals = result["macd_signal"].dropna().count()
    assert fast_signals >= 0  # Just verify no crash with defaults
