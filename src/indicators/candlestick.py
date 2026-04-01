"""Candlestick reversal and continuation pattern detection.

Detects:
- Hammer / Inverted hammer (bullish reversal)
- Bullish / Bearish engulfing
- Doji (indecision → reversal)
- Morning star / Evening star (3-candle reversal)
- Three white soldiers / Three black crows (strong continuation)
- Piercing line / Dark cloud cover (2-candle reversal)
- Shooting star (bearish reversal)

Each pattern is scored by context (trend, volume, support/resistance proximity)
to filter out weak signals.
"""
import numpy as np
import pandas as pd


# ── Individual pattern detectors ──

def _is_doji(o, h, l, c, threshold=0.05):
    body = abs(c - o)
    total = h - l
    return total > 0 and body / total < threshold


def _is_hammer(o, h, l, c):
    body = abs(c - o)
    total = h - l
    if total == 0:
        return False
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    return (lower_wick >= 2 * body and
            upper_wick <= body * 0.3 and
            body / total < 0.35)


def _is_inverted_hammer(o, h, l, c):
    body = abs(c - o)
    total = h - l
    if total == 0:
        return False
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    return (upper_wick >= 2 * body and
            lower_wick <= body * 0.3 and
            body / total < 0.35)


def _is_shooting_star(o, h, l, c):
    return _is_inverted_hammer(o, h, l, c) and c < o


def _is_bullish_engulfing(o1, h1, l1, c1, o2, h2, l2, c2):
    prev_bearish = c1 < o1
    curr_bullish = c2 > o2
    curr_body = abs(c2 - o2)
    prev_body = abs(c1 - o1)
    return (prev_bearish and curr_bullish and
            o2 <= c1 and c2 >= o1 and
            curr_body > prev_body)


def _is_bearish_engulfing(o1, h1, l1, c1, o2, h2, l2, c2):
    prev_bullish = c1 > o1
    curr_bearish = c2 < o2
    curr_body = abs(c2 - o2)
    prev_body = abs(c1 - o1)
    return (prev_bullish and curr_bearish and
            o2 >= c1 and c2 <= o1 and
            curr_body > prev_body)


def _is_morning_star(o1, h1, l1, c1, o2, h2, l2, c2, o3, h3, l3, c3):
    first_bearish = c1 < o1 and abs(c1 - o1) / (h1 - l1 + 1e-10) > 0.5
    middle_small = abs(c2 - o2) < abs(c1 - o1) * 0.3
    third_bullish = c3 > o3 and abs(c3 - o3) / (h3 - l3 + 1e-10) > 0.5
    recovery = c3 > (o1 + c1) / 2
    return first_bearish and middle_small and third_bullish and recovery


def _is_evening_star(o1, h1, l1, c1, o2, h2, l2, c2, o3, h3, l3, c3):
    first_bullish = c1 > o1 and abs(c1 - o1) / (h1 - l1 + 1e-10) > 0.5
    middle_small = abs(c2 - o2) < abs(c1 - o1) * 0.3
    third_bearish = c3 < o3 and abs(c3 - o3) / (h3 - l3 + 1e-10) > 0.5
    recovery = c3 < (o1 + c1) / 2
    return first_bullish and middle_small and third_bearish and recovery


def _is_three_white_soldiers(candles):
    if len(candles) < 3:
        return False
    for i in range(3):
        o, h, l, c = candles[i]
        if c <= o:
            return False
        if (h - l) > 0 and (c - o) / (h - l) < 0.5:
            return False
    return (candles[1][3] > candles[0][3] and
            candles[2][3] > candles[1][3] and
            candles[1][0] >= candles[0][0] and candles[1][0] <= candles[0][3] and
            candles[2][0] >= candles[1][0] and candles[2][0] <= candles[1][3])


def _is_three_black_crows(candles):
    if len(candles) < 3:
        return False
    for i in range(3):
        o, h, l, c = candles[i]
        if c >= o:
            return False
        if (h - l) > 0 and (o - c) / (h - l) < 0.5:
            return False
    return (candles[1][3] < candles[0][3] and
            candles[2][3] < candles[1][3])


def _is_piercing_line(o1, h1, l1, c1, o2, h2, l2, c2):
    prev_bearish = c1 < o1
    curr_bullish = c2 > o2
    opens_below = o2 < c1
    midpoint = (o1 + c1) / 2
    closes_above_mid = c2 > midpoint and c2 < o1
    return prev_bearish and curr_bullish and opens_below and closes_above_mid


def _is_dark_cloud_cover(o1, h1, l1, c1, o2, h2, l2, c2):
    prev_bullish = c1 > o1
    curr_bearish = c2 < o2
    opens_above = o2 > c1
    midpoint = (o1 + c1) / 2
    closes_below_mid = c2 < midpoint and c2 > o1
    return prev_bullish and curr_bearish and opens_above and closes_below_mid


# ── Trend context ──

def _get_trend(closes, idx, lookback=10):
    if idx < lookback:
        return "flat"
    change = (closes[idx] - closes[idx - lookback]) / closes[idx - lookback]
    if change > 0.02:
        return "up"
    elif change < -0.02:
        return "down"
    return "flat"


# ── Main detection ──

def detect_candlestick_patterns(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Detect candlestick patterns with trend context filtering.

    Returns DataFrame with: candle_pattern, candle_signal, candle_strength
    """
    if params is None:
        params = {"require_trend_context": True, "volume_confirmation": True,
                  "min_strength": "moderate"}

    cs = params if "require_trend_context" in params else params.get("candlestick", params)
    require_trend = cs.get("require_trend_context", True)
    vol_confirm = cs.get("volume_confirmation", True)
    min_strength = cs.get("min_strength", "moderate")
    strength_order = {"weak": 0, "moderate": 1, "strong": 2}

    n = len(df)
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values
    vol_ma = df["volume"].rolling(20).mean().values

    patterns = [None] * n
    signals = [None] * n
    strengths = [None] * n

    for i in range(3, n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        o1, h1, l1, c1 = opens[i-1], highs[i-1], lows[i-1], closes[i-1]
        o2, h2, l2, c2 = opens[i-2], highs[i-2], lows[i-2], closes[i-2]

        trend = _get_trend(closes, i)
        vol_ratio = volumes[i] / vol_ma[i] if vol_ma[i] and vol_ma[i] > 0 else 1.0

        detected = None
        signal = None
        strength = "weak"

        # ── Bullish patterns ──
        if _is_hammer(o, h, l, c) and (not require_trend or trend == "down"):
            detected, signal = "hammer", "buy"
            strength = "strong" if vol_ratio > 1.2 else "moderate"

        elif _is_inverted_hammer(o, h, l, c) and c > o and (not require_trend or trend == "down"):
            detected, signal = "inverted_hammer", "buy"
            strength = "moderate" if vol_ratio > 1.0 else "weak"

        elif _is_bullish_engulfing(o1, h1, l1, c1, o, h, l, c) and (not require_trend or trend == "down"):
            detected, signal = "bullish_engulfing", "buy"
            strength = "strong" if vol_ratio > 1.3 else "moderate"

        elif _is_morning_star(o2, h2, l2, c2, o1, h1, l1, c1, o, h, l, c):
            detected, signal, strength = "morning_star", "buy", "strong"

        elif _is_piercing_line(o1, h1, l1, c1, o, h, l, c) and (not require_trend or trend == "down"):
            detected, signal = "piercing_line", "buy"
            strength = "moderate" if vol_ratio > 1.0 else "weak"

        elif _is_three_white_soldiers([(opens[i-2], highs[i-2], lows[i-2], closes[i-2]),
                                        (opens[i-1], highs[i-1], lows[i-1], closes[i-1]),
                                        (opens[i], highs[i], lows[i], closes[i])]):
            detected, signal, strength = "three_white_soldiers", "buy", "strong"

        # ── Bearish patterns ──
        if detected is None:
            if _is_shooting_star(o, h, l, c) and (not require_trend or trend == "up"):
                detected, signal = "shooting_star", "sell"
                strength = "strong" if vol_ratio > 1.2 else "moderate"

            elif _is_bearish_engulfing(o1, h1, l1, c1, o, h, l, c) and (not require_trend or trend == "up"):
                detected, signal = "bearish_engulfing", "sell"
                strength = "strong" if vol_ratio > 1.3 else "moderate"

            elif _is_evening_star(o2, h2, l2, c2, o1, h1, l1, c1, o, h, l, c):
                detected, signal, strength = "evening_star", "sell", "strong"

            elif _is_dark_cloud_cover(o1, h1, l1, c1, o, h, l, c) and (not require_trend or trend == "up"):
                detected, signal = "dark_cloud_cover", "sell"
                strength = "moderate" if vol_ratio > 1.0 else "weak"

            elif _is_three_black_crows([(opens[i-2], highs[i-2], lows[i-2], closes[i-2]),
                                         (opens[i-1], highs[i-1], lows[i-1], closes[i-1]),
                                         (opens[i], highs[i], lows[i], closes[i])]):
                detected, signal, strength = "three_black_crows", "sell", "strong"

        # Doji (context-dependent)
        if detected is None and _is_doji(o, h, l, c):
            detected = "doji"
            if trend == "up":
                signal, strength = "sell", "weak"
            elif trend == "down":
                signal, strength = "buy", "weak"

        # Volume filter
        if vol_confirm and detected and vol_ratio < 0.8:
            strength = "weak"

        # Minimum strength filter
        if detected and strength_order.get(strength, 0) < strength_order.get(min_strength, 1):
            signal = None

        patterns[i] = detected
        signals[i] = signal
        strengths[i] = strength

    df = df.copy()
    df["candle_pattern"] = patterns
    df["candle_signal"] = signals
    df["candle_strength"] = strengths
    return df


def calculate_candle_levels(df: pd.DataFrame, idx: int, params: dict = None) -> dict:
    """Calculate trade levels for a candlestick signal using ATR-based stops."""
    if params is None:
        params = {"atr_sl_multiplier": 3.0, "min_risk_reward": 2.0}

    cs = params if "atr_sl_multiplier" in params else params.get("candlestick", params)
    sl_mult = cs.get("atr_sl_multiplier", 3.0)
    min_rr = cs.get("min_risk_reward", 2.0)

    from src.indicators.features import _atr
    entry = df.iloc[idx]["close"]
    atr = _atr(df["high"].values[:idx+1], df["low"].values[:idx+1],
               df["close"].values[:idx+1], 14)

    signal = df.iloc[idx]["candle_signal"]
    if signal == "buy":
        stop_loss = entry - (atr * sl_mult)
        risk = entry - stop_loss
        target = entry + (risk * min_rr)
    else:
        return {"entry": entry, "stop_loss": entry, "target": entry,
                "risk": 0, "risk_reward": 0}

    return {
        "entry": entry, "stop_loss": stop_loss, "target": target,
        "risk": abs(risk), "risk_reward": min_rr,
        "pattern": df.iloc[idx].get("candle_pattern"),
        "strength": df.iloc[idx].get("candle_strength"), "atr": atr,
    }
