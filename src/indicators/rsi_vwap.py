"""RSI Mean Reversion + VWAP exit strategy.

Detects oversold conditions via RSI and uses VWAP as a dynamic
take-profit target (fair value level).

- Entry: RSI drops below oversold threshold (default 30)
- Exit: Price reaches VWAP or RSI recovers above exit threshold (default 50)
- Stop: ATR-based stop below recent swing low
"""
import numpy as np
import pandas as pd


def _compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI for entire array."""
    rsi = np.full(len(closes), np.nan)
    if len(closes) < period + 1:
        return rsi

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - (100 / (1 + rs))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100 - (100 / (1 + rs))

    return rsi


def _compute_vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                  volumes: np.ndarray) -> np.ndarray:
    """Compute running VWAP using typical price."""
    typical_price = (highs + lows + closes) / 3
    cum_tp_vol = np.cumsum(typical_price * volumes)
    cum_vol = np.cumsum(volumes)
    vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, typical_price)
    return vwap


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


def detect_rsi_vwap(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Detect RSI oversold + VWAP mean reversion signals.

    Args:
        df: OHLCV DataFrame
        params: Strategy parameters (rsi_period, oversold_threshold, etc.)

    Returns:
        DataFrame with added columns: rsi_14, vwap, rsi_vwap_signal
    """
    df = df.copy()
    if params is None:
        params = {}

    cfg = params if "rsi_period" in params else params.get("rsi_vwap", params)
    rsi_period = cfg.get("rsi_period", 14)
    oversold = cfg.get("oversold_threshold", 30)
    volume_confirmation = cfg.get("volume_confirmation", True)
    volume_ratio_min = cfg.get("volume_ratio_min", 1.0)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values

    rsi = _compute_rsi(closes, rsi_period)
    vwap = _compute_vwap(highs, lows, closes, volumes)
    atr = _compute_atr(highs, lows, closes, 14)

    df["rsi_14"] = rsi
    df["vwap"] = vwap
    df["atr_14"] = atr
    df["rsi_vwap_signal"] = None

    vol_ma20 = pd.Series(volumes).rolling(20).mean().values

    for i in range(50, len(df)):
        if np.isnan(rsi[i]):
            continue

        if rsi[i] < oversold:
            if closes[i] < vwap[i]:
                if volume_confirmation and vol_ma20[i] > 0:
                    if volumes[i] / vol_ma20[i] < volume_ratio_min:
                        continue

                df.iloc[i, df.columns.get_loc("rsi_vwap_signal")] = "buy"

    return df


def calculate_rsi_vwap_levels(df: pd.DataFrame, idx: int,
                              params: dict = None) -> dict:
    """Calculate trade levels for an RSI+VWAP signal.

    Target is VWAP (dynamic fair value). Stop is ATR-based below entry.
    """
    if params is None:
        params = {}

    cfg = params if "sl_atr_multiplier" in params else params.get("rsi_vwap", params)
    sl_atr_mult = cfg.get("sl_atr_multiplier", 2.0)
    min_rr = cfg.get("min_risk_reward", 2.0)

    close = df.iloc[idx]["close"]
    vwap = df.iloc[idx].get("vwap", close)
    atr = df.iloc[idx].get("atr_14", close * 0.02)
    if np.isnan(atr) or atr <= 0:
        atr = close * 0.02

    entry = close
    stop_loss = entry - (atr * sl_atr_mult)

    risk = entry - stop_loss
    vwap_target = vwap
    min_target = entry + (risk * min_rr)
    target = max(vwap_target, min_target)

    risk_reward = (target - entry) / risk if risk > 0 else 0

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "target": target,
        "risk": risk,
        "risk_reward": risk_reward,
        "rsi": df.iloc[idx].get("rsi_14", None),
        "vwap": vwap,
        "atr": atr,
    }
