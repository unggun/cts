import pandas as pd
import numpy as np
import pytest
from src.indicators.cvd import detect_cvd_divergence, calculate_cvd_levels


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


def test_detect_cvd_adds_columns():
    df = _make_df()
    result = detect_cvd_divergence(df)
    assert "cvd" in result.columns
    assert "cvd_signal" in result.columns


def test_detect_cvd_signals_are_valid():
    df = _make_df()
    result = detect_cvd_divergence(df)
    valid_values = {None, "buy"}
    signals = set(result["cvd_signal"].dropna().unique()) | {None}
    assert signals.issubset(valid_values)


def test_calculate_cvd_levels_returns_required_keys():
    df = _make_df()
    result = detect_cvd_divergence(df)
    idx = 100
    levels = calculate_cvd_levels(result, idx)
    assert "entry" in levels
    assert "stop_loss" in levels
    assert "target" in levels
    assert "risk" in levels
    assert "risk_reward" in levels
    assert levels["risk"] > 0


def test_cvd_custom_params():
    df = _make_df()
    params = {"lookback": 10, "divergence_bars": 8}
    result = detect_cvd_divergence(df, params)
    assert "cvd_signal" in result.columns
