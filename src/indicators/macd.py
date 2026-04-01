"""MACD Histogram crossover strategy with aggressive default parameters.

Default params: fast=3, slow=15, signal=3 (faster than classic 12/26/9).
These react faster to momentum shifts, suitable for crypto volatility.

- Entry: MACD line crosses above signal line (histogram goes positive)
- Exit: Reverse crossover or stop-loss/take-profit
- Volume confirmation optional
"""
import numpy as np
import pandas as pd


def _compute_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA for entire array."""
    ema = np.full(len(data), np.nan)
    if len(data) < period:
        return ema

    multiplier = 2 / (period + 1)
    ema[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        ema[i] = data[i] * multiplier + ema[i - 1] * (1 - multiplier)
    return ema


def _compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                 period: int = 14) -> np.ndarray:
    """Compute ATR for entire array."""
    atr = np.full(len(highs), np.nan)
    if len(highs) < period + 1:
        return atr

    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    tr = np.concatenate([[np.nan], tr])

    atr[period] = np.nanmean(tr[1:period + 1])
    for i in range(period + 1, len(highs)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def detect_macd(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Detect MACD histogram crossover signals.

    A buy signal is generated when the MACD line crosses above the signal
    line (histogram turns positive from negative).

    Args:
        df: OHLCV DataFrame
        params: Strategy parameters (fast_period, slow_period, signal_period)

    Returns:
        DataFrame with added columns: macd_line, macd_signal_line,
        macd_histogram, macd_signal
    """
    df = df.copy()
    if params is None:
        params = {}

    cfg = params if "fast_period" in params else params.get("macd", params)
    fast_period = cfg.get("fast_period", 3)
    slow_period = cfg.get("slow_period", 15)
    signal_period = cfg.get("signal_period", 3)
    volume_confirmation = cfg.get("volume_confirmation", True)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values

    fast_ema = _compute_ema(closes, fast_period)
    slow_ema = _compute_ema(closes, slow_period)
    macd_line = fast_ema - slow_ema

    signal_line = _compute_ema(
        np.where(np.isnan(macd_line), 0, macd_line), signal_period
    )
    signal_line = np.where(np.isnan(macd_line), np.nan, signal_line)

    histogram = macd_line - signal_line
    atr = _compute_atr(highs, lows, closes, 14)

    df["macd_line"] = macd_line
    df["macd_signal_line"] = signal_line
    df["macd_histogram"] = histogram
    df["atr_14"] = atr
    df["macd_signal"] = None

    vol_ma20 = pd.Series(volumes).rolling(20).mean().values

    for i in range(max(50, slow_period + signal_period), len(df)):
        if np.isnan(histogram[i]) or np.isnan(histogram[i - 1]):
            continue

        if histogram[i] > 0 and histogram[i - 1] <= 0:
            if volume_confirmation and vol_ma20[i] > 0:
                if volumes[i] / vol_ma20[i] < 0.8:
                    continue

            df.iloc[i, df.columns.get_loc("macd_signal")] = "buy"

    return df


def calculate_macd_levels(df: pd.DataFrame, idx: int,
                          params: dict = None) -> dict:
    """Calculate trade levels for a MACD signal.

    Stop is ATR-based. Target uses risk/reward ratio.
    """
    if params is None:
        params = {}

    cfg = params if "sl_atr_multiplier" in params else params.get("macd", params)
    sl_atr_mult = cfg.get("sl_atr_multiplier", 3.5)
    min_rr = cfg.get("min_risk_reward", 2.0)

    close = df.iloc[idx]["close"]
    atr = df.iloc[idx].get("atr_14", close * 0.02)
    if np.isnan(atr) or atr <= 0:
        atr = close * 0.02

    entry = close
    stop_loss = entry - (atr * sl_atr_mult)
    risk = entry - stop_loss
    target = entry + (risk * min_rr)
    risk_reward = (target - entry) / risk if risk > 0 else 0

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "target": target,
        "risk": risk,
        "risk_reward": risk_reward,
        "macd_line": df.iloc[idx].get("macd_line", None),
        "macd_histogram": df.iloc[idx].get("macd_histogram", None),
        "atr": atr,
    }
