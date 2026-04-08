"""Support and resistance level detection with breakout signals.

Identifies horizontal S/R levels by finding price zones where price
has repeatedly reversed (swing highs/lows clustering). Generates
breakout signals when price decisively breaks through these levels
with volume confirmation.

Methods:
- Swing high/low clustering (primary)
- Volume profile zones (secondary confirmation)
- Touch count scoring (more touches = stronger level)
"""
import numpy as np
import pandas as pd
from typing import List, Tuple


def find_swing_points(highs: np.ndarray, lows: np.ndarray,
                      lookback: int = 5) -> Tuple[List, List]:
    """Find swing highs and swing lows.

    A swing high is a high that is higher than `lookback` candles on each side.
    A swing low is a low that is lower than `lookback` candles on each side.
    """
    n = len(highs)
    swing_highs = []  # (index, price)
    swing_lows = []   # (index, price)

    for i in range(lookback, n - lookback):
        # Swing high
        is_high = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_high = False
                break
        if is_high:
            swing_highs.append((i, highs[i]))

        # Swing low
        is_low = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_low = False
                break
        if is_low:
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def cluster_levels(points: List[Tuple[int, float]],
                   tolerance_pct: float = 0.005) -> List[dict]:
    """Cluster swing points into support/resistance levels.

    Points within `tolerance_pct` of each other are grouped.
    Returns levels sorted by touch count (strongest first).
    """
    if not points:
        return []

    # Sort by price
    sorted_points = sorted(points, key=lambda x: x[1])
    levels = []
    used = set()

    for i, (idx_i, price_i) in enumerate(sorted_points):
        if i in used:
            continue

        cluster = [(idx_i, price_i)]
        used.add(i)

        for j, (idx_j, price_j) in enumerate(sorted_points):
            if j in used:
                continue
            if abs(price_j - price_i) / price_i <= tolerance_pct:
                cluster.append((idx_j, price_j))
                used.add(j)

        if len(cluster) >= 2:  # Need at least 2 touches
            prices = [p for _, p in cluster]
            indices = [idx for idx, _ in cluster]
            levels.append({
                "price": np.mean(prices),
                "touches": len(cluster),
                "first_touch_idx": min(indices),
                "last_touch_idx": max(indices),
                "price_range": (min(prices), max(prices)),
            })

    # Sort by touch count (strongest first)
    levels.sort(key=lambda x: x["touches"], reverse=True)
    return levels


def detect_sr_breakouts(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Detect support/resistance levels and breakout signals.

    Args:
        df: OHLCV DataFrame
        params: Config with sr settings

    Returns:
        DataFrame with added columns:
        - sr_nearest_resistance: nearest resistance level above current price
        - sr_nearest_support: nearest support level below current price
        - sr_signal: 'buy' on resistance breakout, 'sell' on support break
        - sr_level_strength: number of touches on the broken level
        - sr_breakout_type: 'resistance_break' or 'support_break'
    """
    if params is None:
        params = {
            "swing_lookback": 5,
            "cluster_tolerance_pct": 0.005,
            "min_touches": 2,
            "volume_confirmation": True,
            "breakout_threshold_pct": 0.002,
            "recalc_interval": 50,
        }

    sr = params if "swing_lookback" in params else params.get("support_resistance", params)
    swing_lb = sr.get("swing_lookback", 5)
    tolerance = sr.get("cluster_tolerance_pct", 0.005)
    min_touches = sr.get("min_touches", 2)
    vol_confirm = sr.get("volume_confirmation", True)
    breakout_thresh = sr.get("breakout_threshold_pct", 0.002)
    recalc_every = sr.get("recalc_interval", 50)

    n = len(df)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values
    vol_ma = df["volume"].rolling(20).mean().values

    nearest_res = np.full(n, np.nan)
    nearest_sup = np.full(n, np.nan)
    sr_signals = [None] * n
    sr_strength = np.zeros(n, dtype=int)
    sr_type = [None] * n

    levels = []
    last_calc = 0

    for i in range(swing_lb * 2 + 10, n):
        # Recalculate levels periodically (expensive operation)
        if i - last_calc >= recalc_every or not levels:
            # Use data up to current point
            sh, sl = find_swing_points(highs[:i+1], lows[:i+1], swing_lb)

            # Cluster into levels
            all_points = sh + sl
            levels = cluster_levels(all_points, tolerance)
            levels = [lv for lv in levels if lv["touches"] >= min_touches]
            last_calc = i

        if not levels:
            continue

        price = closes[i]

        # Find nearest resistance and support
        resistances = [lv for lv in levels if lv["price"] > price * (1 + breakout_thresh)]
        supports = [lv for lv in levels if lv["price"] < price * (1 - breakout_thresh)]

        if resistances:
            nearest_r = min(resistances, key=lambda x: x["price"])
            nearest_res[i] = nearest_r["price"]
        if supports:
            nearest_s = max(supports, key=lambda x: x["price"])
            nearest_sup[i] = nearest_s["price"]

        # Check for breakout
        prev_close = closes[i - 1] if i > 0 else price

        # Resistance breakout (bullish)
        for lv in levels:
            lv_price = lv["price"]
            # Was below, now above
            if (prev_close <= lv_price * (1 + breakout_thresh) and
                price > lv_price * (1 + breakout_thresh)):

                vol_ok = True
                if vol_confirm and vol_ma[i] and vol_ma[i] > 0:
                    vol_ok = volumes[i] / vol_ma[i] >= 1.2

                if vol_ok and lv["touches"] >= min_touches:
                    sr_signals[i] = "buy"
                    sr_strength[i] = lv["touches"]
                    sr_type[i] = "resistance_break"
                    break

            # Support break (bearish — useful for exits or shorts)
            elif (prev_close >= lv_price * (1 - breakout_thresh) and
                  price < lv_price * (1 - breakout_thresh)):

                vol_ok = True
                if vol_confirm and vol_ma[i] and vol_ma[i] > 0:
                    vol_ok = volumes[i] / vol_ma[i] >= 1.2

                if vol_ok and lv["touches"] >= min_touches:
                    sr_signals[i] = "sell"
                    sr_strength[i] = lv["touches"]
                    sr_type[i] = "support_break"
                    break

    df = df.copy()
    df["sr_nearest_resistance"] = nearest_res
    df["sr_nearest_support"] = nearest_sup
    df["sr_signal"] = sr_signals
    df["sr_level_strength"] = sr_strength
    df["sr_breakout_type"] = sr_type

    return df


def calculate_sr_levels(df: pd.DataFrame, idx: int, params: dict = None) -> dict:
    """Calculate trade levels for an S/R breakout signal.

    For resistance break: stop below the broken level, target = next resistance
    For support break: stop above the broken level, target = next support
    """
    if params is None:
        params = {"stop_buffer_pct": 0.015, "min_risk_reward": 2.0}

    sr = params if "stop_buffer_pct" in params else params.get("support_resistance", params)
    stop_buffer = sr.get("stop_buffer_pct", 0.015)
    min_rr = sr.get("min_risk_reward", 2.0)

    entry = df.iloc[idx]["close"]
    breakout_type = df.iloc[idx].get("sr_breakout_type")
    touches = df.iloc[idx].get("sr_level_strength", 0)

    if breakout_type == "resistance_break":
        # Stop below the resistance level we just broke (now support)
        resistance = df.iloc[idx].get("sr_nearest_resistance", entry)
        # The resistance we broke is roughly at entry or slightly below
        # Use nearest support as reference for stop
        support = df.iloc[idx].get("sr_nearest_support")
        if support and not np.isnan(support):
            stop_loss = support * (1 - stop_buffer)
        else:
            # Fallback: ATR-based stop
            from src.indicators.features import _atr
            atr = _atr(df["high"].values[:idx+1], df["low"].values[:idx+1],
                       df["close"].values[:idx+1], 14)
            sl_mult = sr.get("sl_atr_multiplier", 6.0)
            stop_loss = entry - (sl_mult * atr)

        risk = entry - stop_loss
        target = entry + (risk * min_rr)

    elif breakout_type == "support_break":
        # For now, only generate buy-side levels
        return {"entry": entry, "stop_loss": entry, "target": entry,
                "risk": 0, "risk_reward": 0}
    else:
        return {"entry": entry, "stop_loss": entry, "target": entry,
                "risk": 0, "risk_reward": 0}

    risk_reward = (target - entry) / risk if risk > 0 else 0

    return {
        "entry": entry, "stop_loss": stop_loss, "target": target,
        "risk": abs(risk), "risk_reward": risk_reward,
        "breakout_type": breakout_type, "level_touches": touches,
    }
