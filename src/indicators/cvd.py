"""Cumulative Volume Delta (CVD) divergence strategy.

CVD approximates buying vs selling pressure from OHLCV data.
Divergence between price and CVD reveals hidden pressure:
- Price makes lower low but CVD makes higher low → bullish divergence → buy
- Indicates hidden buying pressure despite falling price.

Volume delta is approximated using the close position within the bar:
  delta = volume * ((close - low) - (high - close)) / (high - low)
This gives positive delta when close is near the high (buying pressure)
and negative delta when close is near the low (selling pressure).
"""
import numpy as np
import pandas as pd


def _estimate_volume_delta(opens: np.ndarray, highs: np.ndarray,
                           lows: np.ndarray, closes: np.ndarray,
                           volumes: np.ndarray) -> np.ndarray:
    """Estimate volume delta from OHLCV using close position in bar."""
    bar_range = highs - lows
    safe_range = np.where(bar_range > 0, bar_range, 1.0)
    close_position = ((closes - lows) - (highs - closes)) / safe_range
    delta = volumes * close_position
    return delta


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


def detect_cvd_divergence(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Detect bullish CVD divergence signals.

    A bullish divergence occurs when price makes a lower low but CVD
    makes a higher low over the lookback window, indicating hidden
    buying pressure.

    Args:
        df: OHLCV DataFrame
        params: Strategy parameters

    Returns:
        DataFrame with added columns: cvd, volume_delta, cvd_signal
    """
    df = df.copy()
    if params is None:
        params = {}

    cfg = params if "lookback" in params else params.get("cvd", params)
    lookback = cfg.get("lookback", 14)
    divergence_bars = cfg.get("divergence_bars", 10)
    volume_confirmation = cfg.get("volume_confirmation", True)

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values

    delta = _estimate_volume_delta(opens, highs, lows, closes, volumes)
    cvd = np.cumsum(delta)
    atr = _compute_atr(highs, lows, closes, 14)

    df["volume_delta"] = delta
    df["cvd"] = cvd
    df["atr_14"] = atr
    df["cvd_signal"] = None

    vol_ma20 = pd.Series(volumes).rolling(20).mean().values

    for i in range(max(50, divergence_bars + lookback), len(df)):
        if np.isnan(atr[i]):
            continue

        price_window = lows[i - divergence_bars:i + 1]
        cvd_window = cvd[i - divergence_bars:i + 1]

        if lows[i] > np.min(price_window) * 1.005:
            continue

        prev_low_idx_rel = np.argmin(price_window[:-1])
        if len(price_window) < 2:
            continue

        price_divergence = lows[i] <= price_window[prev_low_idx_rel] * 1.002
        cvd_divergence = cvd[i] > cvd_window[prev_low_idx_rel] * 1.0

        if price_divergence and cvd_divergence:
            if volume_confirmation and vol_ma20[i] > 0:
                if volumes[i] / vol_ma20[i] < 0.8:
                    continue

            df.iloc[i, df.columns.get_loc("cvd_signal")] = "buy"

    return df


def calculate_cvd_levels(df: pd.DataFrame, idx: int,
                         params: dict = None) -> dict:
    """Calculate trade levels for a CVD divergence signal.

    Stop below recent swing low. Target based on ATR and risk/reward.
    """
    if params is None:
        params = {}

    cfg = params if "sl_atr_multiplier" in params else params.get("cvd", params)
    sl_atr_mult = cfg.get("sl_atr_multiplier", 2.0)
    min_rr = cfg.get("min_risk_reward", 2.0)
    lookback = cfg.get("lookback", 14)

    close = df.iloc[idx]["close"]
    atr = df.iloc[idx].get("atr_14", close * 0.02)
    if np.isnan(atr) or atr <= 0:
        atr = close * 0.02

    start = max(0, idx - lookback)
    recent_low = df.iloc[start:idx + 1]["low"].min()
    stop_loss = min(recent_low - (atr * 0.5), close - (atr * sl_atr_mult))

    entry = close
    risk = entry - stop_loss
    target = entry + (risk * min_rr)
    risk_reward = (target - entry) / risk if risk > 0 else 0

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "target": target,
        "risk": risk,
        "risk_reward": risk_reward,
        "cvd": df.iloc[idx].get("cvd", None),
        "volume_delta": df.iloc[idx].get("volume_delta", None),
        "atr": atr,
    }
