"""Z-score mean reversion strategy with RSI confirmation.

Buys when price deviates significantly below its rolling mean (z-score < threshold)
AND RSI confirms oversold conditions. This avoids catching falling knives by
requiring momentum exhaustion before entry.

Exit target is the rolling mean itself — the price level we expect to revert to.

- Entry: z-score < threshold AND RSI < rsi_threshold
- Target: rolling mean (natural mean reversion target)
- Stop: ATR-based with trailing stop support
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


def detect_mean_reversion(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Detect mean reversion buy signals using z-score + RSI confirmation.

    A buy signal is generated when:
    1. Z-score is below the threshold (price significantly below rolling mean)
    2. RSI confirms oversold conditions (momentum exhaustion)
    3. Optional volume confirmation

    Args:
        df: OHLCV DataFrame
        params: Strategy parameters

    Returns:
        DataFrame with added columns: zscore, rolling_mean, rsi_14,
        atr_14, mr_signal
    """
    df = df.copy()
    if params is None:
        params = {}

    cfg = params if "zscore_threshold" in params else params.get("mean_reversion", params)
    window = cfg.get("window", 20)
    zscore_threshold = cfg.get("zscore_threshold", -1.5)
    rsi_period = cfg.get("rsi_period", 14)
    rsi_threshold = cfg.get("rsi_threshold", 35)
    volume_confirmation = cfg.get("volume_confirmation", True)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values

    # Rolling mean and std for z-score
    rolling_mean = pd.Series(closes).rolling(window=window).mean().values
    rolling_std = pd.Series(closes).rolling(window=window).std().values

    # Z-score: how far price is from its rolling mean in std devs
    zscore = np.where(rolling_std > 0,
                      (closes - rolling_mean) / rolling_std,
                      0.0)

    rsi = _compute_rsi(closes, rsi_period)
    atr = _compute_atr(highs, lows, closes, 14)

    df["zscore"] = zscore
    df["rolling_mean"] = rolling_mean
    df["rolling_std"] = rolling_std
    df["rsi_14"] = rsi
    df["atr_14"] = atr
    df["mr_signal"] = None

    vol_ma20 = pd.Series(volumes).rolling(20).mean().values

    for i in range(50, len(df)):
        if np.isnan(zscore[i]) or np.isnan(rsi[i]):
            continue

        # Z-score below threshold (price significantly below mean)
        if zscore[i] < zscore_threshold:
            # RSI confirms oversold (momentum exhaustion)
            if rsi[i] < rsi_threshold:
                # Optional volume confirmation
                if volume_confirmation and vol_ma20[i] > 0:
                    if volumes[i] / vol_ma20[i] < 0.8:
                        continue

                df.iloc[i, df.columns.get_loc("mr_signal")] = "buy"

    return df


def calculate_mean_reversion_levels(df: pd.DataFrame, idx: int,
                                    params: dict = None) -> dict:
    """Calculate trade levels for a mean reversion signal.

    Target is the rolling mean — the natural reversion target.
    Stop is ATR-based with wide multiplier for crypto volatility.
    """
    if params is None:
        params = {}

    cfg = params if "sl_atr_multiplier" in params else params.get("mean_reversion", params)
    sl_atr_mult = cfg.get("sl_atr_multiplier", 7.0)
    min_rr = cfg.get("min_risk_reward", 1.0)

    close = df.iloc[idx]["close"]
    atr = df.iloc[idx].get("atr_14", close * 0.02)
    if np.isnan(atr) or atr <= 0:
        atr = close * 0.02

    rolling_mean = df.iloc[idx].get("rolling_mean", close)
    rolling_std = df.iloc[idx].get("rolling_std", atr)
    if np.isnan(rolling_mean):
        rolling_mean = close
    if np.isnan(rolling_std) or rolling_std <= 0:
        rolling_std = atr

    entry = close

    # Stop: use tighter stop for mean reversion — these are counter-trend
    # trades expecting a bounce, so a smaller stop is appropriate
    stop_loss = entry - (atr * sl_atr_mult)

    # Target: rolling mean + overshoot allowance (price often overshoots
    # the mean on reversion). Target the mean plus 0.5 std devs above it.
    target = rolling_mean + (rolling_std * 0.5)
    if target <= entry:
        # Fallback if somehow target is below entry
        target = entry + (atr * 2.0)

    risk_reward = (target - entry) / risk if risk > 0 else 0

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "target": target,
        "risk": risk,
        "risk_reward": risk_reward,
        "zscore": df.iloc[idx].get("zscore", None),
        "rsi": df.iloc[idx].get("rsi_14", None),
        "rolling_mean": rolling_mean,
        "atr": atr,
    }
