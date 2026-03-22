"""EMA/SMA crossover detection.

Detects moving average crossover signals:

Single crossovers:
- EMA 8/21 crossover (fast, for scalping/swing)
- EMA 21/50 crossover (medium, for swing)
- SMA 50/200 crossover (slow: golden cross / death cross)

Multi-EMA alignment:
- Bullish alignment: 8 > 21 > 50 (strong uptrend)
- Bearish alignment: 8 < 21 < 50 (strong downtrend)
- Convergence/divergence detection

Signal strength is boosted when:
- Multiple timeframe crossovers align
- Volume confirms the crossover
- Price is above/below all MAs
"""
import numpy as np
import pandas as pd
from typing import Optional


def detect_ma_crossovers(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Detect moving average crossover signals.

    Args:
        df: OHLCV DataFrame
        params: Crossover parameters

    Returns:
        DataFrame with added columns:
        - ma_cross_type: name of crossover ('ema8_21_bull', 'golden_cross', etc.)
        - ma_alignment: 'bullish', 'bearish', 'mixed'
        - ma_strength: 1-3
        - signal: 'buy' or 'sell'
        - Also adds the MA columns themselves for reference
    """
    if params is None:
        params = {
            "fast_ema": 8,
            "mid_ema": 21,
            "slow_ema": 50,
            "sma_medium": 50,
            "sma_slow": 200,
            "volume_confirmation": True,
            "price_above_ma_bonus": True,
            "min_strength_for_signal": 2,
        }

    mp = params if "fast_ema" in params else params.get("ma_crossover", params)
    fast = mp.get("fast_ema", 8)
    mid = mp.get("mid_ema", 21)
    slow = mp.get("slow_ema", 50)
    sma_med = mp.get("sma_medium", 50)
    sma_slow_period = mp.get("sma_slow", 200)
    vol_confirm = mp.get("volume_confirmation", True)
    price_bonus = mp.get("price_above_ma_bonus", True)
    min_strength = mp.get("min_strength_for_signal", 2)

    df = df.copy()
    closes = df["close"].values
    volumes = df["volume"].values
    n = len(df)

    # ── Calculate all moving averages ──
    df[f"ema_{fast}"] = _ema_series(closes, fast)
    df[f"ema_{mid}"] = _ema_series(closes, mid)
    df[f"ema_{slow}"] = _ema_series(closes, slow)
    df[f"sma_{sma_med}"] = pd.Series(closes).rolling(sma_med).mean().values
    df[f"sma_{sma_slow_period}"] = pd.Series(closes).rolling(sma_slow_period).mean().values

    ema_fast = df[f"ema_{fast}"].values
    ema_mid = df[f"ema_{mid}"].values
    ema_slow = df[f"ema_{slow}"].values
    sma_medium = df[f"sma_{sma_med}"].values
    sma_200 = df[f"sma_{sma_slow_period}"].values

    vol_ma = pd.Series(volumes).rolling(20).mean().values

    cross_types = [None] * n
    alignments = [None] * n
    strengths = np.zeros(n, dtype=int)
    signals = [None] * n

    for i in range(max(sma_slow_period, slow) + 1, n):
        c = closes[i]
        vol_ratio = volumes[i] / vol_ma[i] if vol_ma[i] and vol_ma[i] > 0 else 1.0

        # ── Detect crossovers ──
        cross = None
        direction = None
        strength = 0

        # EMA fast/mid crossover (8/21)
        if (ema_fast[i] > ema_mid[i] and ema_fast[i-1] <= ema_mid[i-1]):
            cross = f"ema{fast}_{mid}_bull"
            direction = "bullish"
            strength = 1
        elif (ema_fast[i] < ema_mid[i] and ema_fast[i-1] >= ema_mid[i-1]):
            cross = f"ema{fast}_{mid}_bear"
            direction = "bearish"
            strength = 1

        # EMA mid/slow crossover (21/50)
        if (ema_mid[i] > ema_slow[i] and ema_mid[i-1] <= ema_slow[i-1]):
            cross = f"ema{mid}_{slow}_bull"
            direction = "bullish"
            strength = 2
        elif (ema_mid[i] < ema_slow[i] and ema_mid[i-1] >= ema_slow[i-1]):
            cross = f"ema{mid}_{slow}_bear"
            direction = "bearish"
            strength = 2

        # Golden cross / Death cross (SMA 50/200)
        if (sma_medium[i] > sma_200[i] and sma_medium[i-1] <= sma_200[i-1]):
            cross = "golden_cross"
            direction = "bullish"
            strength = 3
        elif (sma_medium[i] < sma_200[i] and sma_medium[i-1] >= sma_200[i-1]):
            cross = "death_cross"
            direction = "bearish"
            strength = 3

        # ── EMA alignment ──
        if ema_fast[i] > ema_mid[i] > ema_slow[i]:
            alignment = "bullish"
        elif ema_fast[i] < ema_mid[i] < ema_slow[i]:
            alignment = "bearish"
        else:
            alignment = "mixed"

        # ── Strength adjustments ──
        if cross and strength > 0:
            # Volume confirmation
            if vol_confirm and vol_ratio >= 1.3:
                strength = min(strength + 1, 3)

            # Price position bonus
            if price_bonus:
                if (direction == "bullish" and
                    c > ema_fast[i] and c > ema_mid[i] and c > ema_slow[i]):
                    strength = min(strength + 1, 3)
                elif (direction == "bearish" and
                      c < ema_fast[i] and c < ema_mid[i] and c < ema_slow[i]):
                    strength = min(strength + 1, 3)

            # Alignment bonus
            if direction == "bullish" and alignment == "bullish":
                strength = min(strength + 1, 3)
            elif direction == "bearish" and alignment == "bearish":
                strength = min(strength + 1, 3)

        cross_types[i] = cross
        alignments[i] = alignment
        strengths[i] = strength

        # Generate signal
        if cross and strength >= min_strength:
            if direction == "bullish":
                signals[i] = "buy"
            elif direction == "bearish":
                signals[i] = "sell"

    df["ma_cross_type"] = cross_types
    df["ma_alignment"] = alignments
    df["ma_strength"] = strengths
    df["signal"] = signals

    return df


def calculate_ma_levels(df: pd.DataFrame, idx: int,
                        params: dict = None) -> dict:
    """Calculate trade levels for a MA crossover signal.

    Stop loss uses the slower MA as dynamic support/resistance.
    Target uses ATR-based projection.
    """
    if params is None:
        params = {
            "fast_ema": 8, "mid_ema": 21, "slow_ema": 50,
            "atr_target_multiplier": 3.0,
            "min_risk_reward": 2.0,
        }

    mp = params if "fast_ema" in params else params.get("ma_crossover", params)
    slow = mp.get("slow_ema", 50)
    atr_mult = mp.get("atr_target_multiplier", 3.0)

    entry = df.iloc[idx]["close"]
    cross_type = df.iloc[idx].get("ma_cross_type", "")
    is_bullish = "bull" in str(cross_type) or cross_type == "golden_cross"

    # ATR for dynamic levels
    atr = _calc_atr(df, idx, 14)

    # Use the slow EMA as dynamic stop loss level
    slow_ema_col = f"ema_{slow}"
    if slow_ema_col in df.columns:
        slow_ma = df.iloc[idx][slow_ema_col]
    else:
        slow_ma = entry

    if is_bullish:
        stop_loss = min(slow_ma, entry - atr * 1.5)
        risk = entry - stop_loss
        target = entry + atr * atr_mult
    else:
        stop_loss = max(slow_ma, entry + atr * 1.5)
        risk = stop_loss - entry
        target = entry - atr * atr_mult

    risk_reward = abs(target - entry) / abs(risk) if risk != 0 else 0

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "target": target,
        "risk": abs(risk),
        "risk_reward": risk_reward,
        "cross_type": cross_type,
        "alignment": df.iloc[idx].get("ma_alignment", "mixed"),
        "strength": int(df.iloc[idx].get("ma_strength", 0)),
    }


def _ema_series(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate EMA for the entire series."""
    result = np.full(len(data), np.nan)
    if len(data) < period:
        return result

    multiplier = 2 / (period + 1)

    # Initialize with SMA
    result[period - 1] = np.mean(data[:period])

    for i in range(period, len(data)):
        result[i] = data[i] * multiplier + result[i-1] * (1 - multiplier)

    return result


def _calc_atr(df: pd.DataFrame, idx: int, period: int = 14) -> float:
    """Calculate ATR at given index."""
    if idx < period + 1:
        return df.iloc[idx]["high"] - df.iloc[idx]["low"]
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    trs = []
    for j in range(idx - period, idx):
        tr = max(
            highs[j] - lows[j],
            abs(highs[j] - closes[j-1]),
            abs(lows[j] - closes[j-1])
        )
        trs.append(tr)
    return np.mean(trs)
