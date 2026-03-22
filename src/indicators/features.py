"""Extract 40+ features per trade for winner/loser pattern analysis.

Features cover: price action, volume, momentum, volatility, trend,
time-based patterns, and candle structure.
"""
import pandas as pd
import numpy as np
from typing import Optional


def extract_features(df: pd.DataFrame, entry_idx: int,
                     exit_idx: Optional[int] = None) -> dict:
    """Extract features at the point of trade entry (and optionally exit).

    Args:
        df: OHLCV DataFrame with technical indicators already computed
        entry_idx: Integer position of the entry candle
        exit_idx: Optional integer position of exit candle

    Returns:
        dict with 40+ feature values
    """
    features = {}
    i = entry_idx

    # Guard: need enough history
    if i < 50:
        return features

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    volumes = df["volume"].values

    c = closes[i]
    h = highs[i]
    l = lows[i]
    o = opens[i]
    v = volumes[i]

    # ── 1. PRICE ACTION FEATURES ──

    # Returns over different lookbacks
    features["return_1"] = (c - closes[i-1]) / closes[i-1] if closes[i-1] > 0 else 0
    features["return_5"] = (c - closes[i-5]) / closes[i-5] if closes[i-5] > 0 else 0
    features["return_10"] = (c - closes[i-10]) / closes[i-10] if closes[i-10] > 0 else 0
    features["return_20"] = (c - closes[i-20]) / closes[i-20] if closes[i-20] > 0 else 0

    # Distance from recent high/low
    high_20 = np.max(highs[i-20:i+1])
    low_20 = np.min(lows[i-20:i+1])
    features["dist_from_20_high"] = (c - high_20) / high_20 if high_20 > 0 else 0
    features["dist_from_20_low"] = (c - low_20) / low_20 if low_20 > 0 else 0

    # ── 2. MOVING AVERAGE FEATURES ──

    ema_8 = _ema(closes[:i+1], 8)
    ema_21 = _ema(closes[:i+1], 21)
    ema_50 = _ema(closes[:i+1], 50) if i >= 50 else c
    sma_20 = np.mean(closes[i-19:i+1])

    features["price_vs_ema8"] = (c - ema_8) / ema_8 if ema_8 > 0 else 0
    features["price_vs_ema21"] = (c - ema_21) / ema_21 if ema_21 > 0 else 0
    features["price_vs_ema50"] = (c - ema_50) / ema_50 if ema_50 > 0 else 0
    features["price_vs_sma20"] = (c - sma_20) / sma_20 if sma_20 > 0 else 0

    # EMA alignment
    features["ema_bullish_aligned"] = 1 if ema_8 > ema_21 > ema_50 else 0
    features["ema_bearish_aligned"] = 1 if ema_8 < ema_21 < ema_50 else 0
    features["ema_mixed"] = 1 if not (features["ema_bullish_aligned"] or features["ema_bearish_aligned"]) else 0

    # ── 3. VOLUME FEATURES ──

    vol_ma_20 = np.mean(volumes[i-19:i+1])
    vol_ma_5 = np.mean(volumes[i-4:i+1])
    features["volume_ratio_20"] = v / vol_ma_20 if vol_ma_20 > 0 else 0
    features["volume_ratio_5"] = v / vol_ma_5 if vol_ma_5 > 0 else 0
    features["volume_trend"] = vol_ma_5 / vol_ma_20 if vol_ma_20 > 0 else 0

    # Volume on up vs down candles (last 10)
    up_vol = sum(volumes[j] for j in range(i-9, i+1) if closes[j] > opens[j])
    down_vol = sum(volumes[j] for j in range(i-9, i+1) if closes[j] <= opens[j])
    features["up_down_volume_ratio"] = up_vol / down_vol if down_vol > 0 else 10.0

    # ── 4. MOMENTUM / RSI FEATURES ──

    rsi_14 = _rsi(closes[:i+1], 14)
    features["rsi_14"] = rsi_14
    features["rsi_zone"] = (
        "oversold" if rsi_14 < 30 else
        "weak" if rsi_14 < 45 else
        "neutral" if rsi_14 < 55 else
        "strong" if rsi_14 < 70 else
        "overbought"
    )

    # RSI divergence (simple check)
    if i >= 14:
        rsi_5_ago = _rsi(closes[:i-4], 14)
        features["rsi_momentum"] = rsi_14 - rsi_5_ago
        # Price making higher low but RSI making lower low = bearish divergence
        features["rsi_bull_divergence"] = 1 if (lows[i] > lows[i-5] and rsi_14 < rsi_5_ago) else 0
    else:
        features["rsi_momentum"] = 0
        features["rsi_bull_divergence"] = 0

    # ── 5. VOLATILITY FEATURES ──

    returns_20 = np.diff(closes[i-20:i+1]) / closes[i-19:i+1]
    features["volatility_20"] = np.std(returns_20) if len(returns_20) > 1 else 0

    atr_14 = _atr(highs[:i+1], lows[:i+1], closes[:i+1], 14)
    features["atr_14"] = atr_14
    features["atr_pct"] = atr_14 / c if c > 0 else 0

    # Bollinger Band position
    bb_std = np.std(closes[i-19:i+1])
    features["bb_position"] = (c - sma_20) / (2 * bb_std) if bb_std > 0 else 0

    # ── 6. CANDLE STRUCTURE ──

    body = abs(c - o)
    upper_wick = h - max(c, o)
    lower_wick = min(c, o) - l
    total_range = h - l

    features["body_pct"] = body / total_range if total_range > 0 else 0
    features["upper_wick_pct"] = upper_wick / total_range if total_range > 0 else 0
    features["lower_wick_pct"] = lower_wick / total_range if total_range > 0 else 0
    features["is_bullish_candle"] = 1 if c > o else 0

    # Candle size relative to ATR
    features["candle_atr_ratio"] = total_range / atr_14 if atr_14 > 0 else 0

    # ── 7. TIME-BASED FEATURES ──

    if hasattr(df.index, 'hour'):
        dt = df.index[i]
        features["hour"] = dt.hour
        features["day_of_week"] = dt.dayofweek
        features["is_weekend"] = 1 if dt.dayofweek >= 5 else 0
        features["session"] = (
            "asian" if 0 <= dt.hour < 8 else
            "european" if 8 <= dt.hour < 16 else
            "american"
        )
    else:
        features["hour"] = -1
        features["day_of_week"] = -1
        features["is_weekend"] = 0
        features["session"] = "unknown"

    # ── 8. TREND CONTEXT ──

    # Higher highs / higher lows count (last 10 candles)
    hh_count = sum(1 for j in range(i-9, i+1) if highs[j] > highs[j-1])
    hl_count = sum(1 for j in range(i-9, i+1) if lows[j] > lows[j-1])
    features["higher_highs_10"] = hh_count
    features["higher_lows_10"] = hl_count
    features["trend_strength"] = (hh_count + hl_count) / 20  # 0 to 1

    # Consecutive candles in same direction
    consec = 0
    direction = 1 if c > o else -1
    for j in range(i, max(i-10, 0), -1):
        if (closes[j] > opens[j]) == (direction > 0):
            consec += 1
        else:
            break
    features["consecutive_candles"] = consec * direction

    return features


# ── Helper functions ──

def _ema(data: np.ndarray, period: int) -> float:
    """Calculate EMA of the last value."""
    if len(data) < period:
        return data[-1]
    multiplier = 2 / (period + 1)
    ema = data[0]
    for val in data[1:]:
        ema = val * multiplier + ema * (1 - multiplier)
    return ema


def _rsi(data: np.ndarray, period: int = 14) -> float:
    """Calculate RSI."""
    if len(data) < period + 1:
        return 50.0
    deltas = np.diff(data[-period-1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         period: int = 14) -> float:
    """Calculate Average True Range."""
    if len(highs) < period + 1:
        return highs[-1] - lows[-1]
    trs = []
    for j in range(-period, 0):
        tr = max(
            highs[j] - lows[j],
            abs(highs[j] - closes[j-1]),
            abs(lows[j] - closes[j-1])
        )
        trs.append(tr)
    return np.mean(trs)
