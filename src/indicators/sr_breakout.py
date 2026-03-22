"""Support and resistance level detection with breakout signals.

Identifies key horizontal price levels using:
- Swing highs / swing lows (local extremes)
- Touch frequency (more touches = stronger level)
- Level clustering (merge nearby levels into zones)

Generates signals when price breaks above resistance or below support
with volume confirmation.
"""
import numpy as np
import pandas as pd
from typing import Optional


def detect_sr_levels(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Detect support/resistance levels and breakout signals.

    Args:
        df: OHLCV DataFrame
        params: Detection parameters

    Returns:
        DataFrame with added columns:
        - sr_nearest_resistance: closest resistance level above
        - sr_nearest_support: closest support level below
        - sr_breakout_type: 'resistance' or 'support' or None
        - sr_level_strength: number of touches on the broken level
        - signal: 'buy' on resistance break, 'sell' on support break
    """
    if params is None:
        params = {
            "swing_lookback": 5,
            "level_tolerance_pct": 0.01,
            "min_touches": 2,
            "volume_ratio_min": 1.2,
            "breakout_confirmation_candles": 1,
        }

    sp = params if "swing_lookback" in params else params.get("sr_breakout", params)
    swing_lb = sp.get("swing_lookback", 5)
    tolerance = sp.get("level_tolerance_pct", 0.01)
    min_touches = sp.get("min_touches", 2)
    vol_min = sp.get("volume_ratio_min", 1.2)
    confirm_candles = sp.get("breakout_confirmation_candles", 1)

    n = len(df)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values
    vol_ma = pd.Series(volumes).rolling(20).mean().values

    sr_resistance = np.full(n, np.nan)
    sr_support = np.full(n, np.nan)
    breakout_type = [None] * n
    level_strength = np.zeros(n, dtype=int)
    signals = [None] * n

    for i in range(swing_lb * 4, n):
        # ── Find swing points up to current candle ──
        swing_highs = []
        swing_lows = []

        for j in range(swing_lb, i - swing_lb):
            # Swing high: highest in its neighborhood
            if highs[j] == max(highs[j - swing_lb:j + swing_lb + 1]):
                swing_highs.append(highs[j])
            # Swing low: lowest in its neighborhood
            if lows[j] == min(lows[j - swing_lb:j + swing_lb + 1]):
                swing_lows.append(lows[j])

        if not swing_highs and not swing_lows:
            continue

        # ── Cluster nearby levels into zones ──
        all_levels = swing_highs + swing_lows
        levels = _cluster_levels(all_levels, tolerance)

        # ── Count touches for each level ──
        level_touches = {}
        for level in levels:
            touches = 0
            for j in range(max(0, i - 100), i):
                # Price touched this level (within tolerance)
                if (abs(highs[j] - level) / level < tolerance or
                    abs(lows[j] - level) / level < tolerance or
                    abs(closes[j] - level) / level < tolerance):
                    touches += 1
            level_touches[level] = touches

        # ── Find nearest S/R to current price ──
        current = closes[i]

        resistance_levels = [
            (lv, level_touches.get(lv, 0))
            for lv in levels if lv > current * (1 + tolerance * 0.5)
        ]
        support_levels = [
            (lv, level_touches.get(lv, 0))
            for lv in levels if lv < current * (1 - tolerance * 0.5)
        ]

        # Sort by distance to current price
        resistance_levels.sort(key=lambda x: x[0])
        support_levels.sort(key=lambda x: -x[0])

        if resistance_levels:
            sr_resistance[i] = resistance_levels[0][0]
        if support_levels:
            sr_support[i] = support_levels[0][0]

        # ── Check for breakouts ──
        vol_ratio = volumes[i] / vol_ma[i] if vol_ma[i] and vol_ma[i] > 0 else 0

        # Resistance breakout
        if resistance_levels:
            res_level, res_touches = resistance_levels[0]

            if (closes[i] > res_level and
                res_touches >= min_touches and
                vol_ratio >= vol_min):

                # Confirm: previous candles were below
                confirmed = True
                for k in range(1, confirm_candles + 1):
                    if i - k >= 0 and closes[i - k] > res_level:
                        confirmed = False
                        break

                if confirmed:
                    breakout_type[i] = "resistance"
                    level_strength[i] = res_touches
                    signals[i] = "buy"

        # Support breakdown
        if support_levels:
            sup_level, sup_touches = support_levels[0]

            if (closes[i] < sup_level and
                sup_touches >= min_touches and
                vol_ratio >= vol_min):

                confirmed = True
                for k in range(1, confirm_candles + 1):
                    if i - k >= 0 and closes[i - k] < sup_level:
                        confirmed = False
                        break

                if confirmed:
                    breakout_type[i] = "support"
                    level_strength[i] = sup_touches
                    signals[i] = "sell"

    df = df.copy()
    df["sr_nearest_resistance"] = sr_resistance
    df["sr_nearest_support"] = sr_support
    df["sr_breakout_type"] = breakout_type
    df["sr_level_strength"] = level_strength
    df["signal"] = signals

    return df


def calculate_sr_levels(df: pd.DataFrame, idx: int,
                        params: dict = None) -> dict:
    """Calculate trade levels for a S/R breakout signal.

    For resistance breakout (buy):
      - Entry at close
      - Stop loss below the broken resistance (now support)
      - Target = distance from nearest support to resistance, projected above

    For support breakdown (sell):
      - Entry at close
      - Stop loss above the broken support (now resistance)
      - Target = distance projected below
    """
    if params is None:
        params = {"stop_loss_buffer_pct": 0.005, "min_risk_reward": 2.0}

    sp = params if "stop_loss_buffer_pct" in params else params.get("sr_breakout", params)
    sl_buffer = sp.get("stop_loss_buffer_pct", 0.005)
    min_rr = sp.get("min_risk_reward", 2.0)

    entry = df.iloc[idx]["close"]
    breakout = df.iloc[idx].get("sr_breakout_type", "resistance")
    resistance = df.iloc[idx].get("sr_nearest_resistance")
    support = df.iloc[idx].get("sr_nearest_support")

    if breakout == "resistance" and resistance is not None:
        # Broken resistance becomes support
        stop_loss = resistance * (1 - sl_buffer)
        risk = entry - stop_loss

        if support is not None and not np.isnan(support):
            # Project the S/R range above
            sr_range = resistance - support
            target = entry + sr_range
        else:
            target = entry + risk * min_rr

        risk_reward = (target - entry) / risk if risk > 0 else 0

    elif breakout == "support" and support is not None:
        stop_loss = support * (1 + sl_buffer)
        risk = stop_loss - entry

        if resistance is not None and not np.isnan(resistance):
            sr_range = resistance - support
            target = entry - sr_range
        else:
            target = entry - risk * min_rr

        risk_reward = (entry - target) / risk if risk > 0 else 0

    else:
        return {"entry": entry, "stop_loss": entry, "target": entry,
                "risk": 0, "risk_reward": 0}

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "target": target,
        "risk": abs(risk),
        "risk_reward": abs(risk_reward),
        "breakout_type": breakout,
        "level_strength": int(df.iloc[idx].get("sr_level_strength", 0)),
        "resistance": resistance,
        "support": support,
    }


def _cluster_levels(levels: list, tolerance: float) -> list:
    """Merge nearby price levels into zones.

    Groups levels within tolerance % of each other and returns
    the average of each group.
    """
    if not levels:
        return []

    sorted_levels = sorted(levels)
    clusters = []
    current_cluster = [sorted_levels[0]]

    for i in range(1, len(sorted_levels)):
        if (sorted_levels[i] - current_cluster[-1]) / current_cluster[-1] <= tolerance:
            current_cluster.append(sorted_levels[i])
        else:
            clusters.append(np.mean(current_cluster))
            current_cluster = [sorted_levels[i]]

    clusters.append(np.mean(current_cluster))
    return clusters
