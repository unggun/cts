"""DBW (Double Bottom / W-pattern) detection.

Identifies W-shaped reversal patterns where price:
1. Makes a low (first bottom)
2. Bounces to a middle peak (neckline)
3. Retests near the first bottom (second bottom)
4. Breaks above the neckline with volume confirmation
"""
import pandas as pd
import numpy as np


def detect_dbw(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Detect Double Bottom / W-pattern formations.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        params: Strategy parameters with dbw config

    Returns:
        DataFrame with added columns:
        - dbw_bottom1_idx: Index of first bottom
        - dbw_bottom2_idx: Index of second bottom
        - dbw_neckline: Neckline (middle peak) price
        - dbw_signal: 'buy' when neckline is broken with volume
    """
    if params is None:
        params = {
            "lookback_period": 20,
            "breakout_threshold": 0.02,
            "volume_confirmation": True,
            "bottom_tolerance_pct": 0.03,
        }

    dbw = params if "lookback_period" in params else params.get("dbw", params)
    lookback = dbw.get("lookback_period", 20)
    tolerance = dbw.get("bottom_tolerance_pct", 0.03)
    vol_confirm = dbw.get("volume_confirmation", True)

    n = len(df)
    lows = df["low"].values
    highs = df["high"].values
    closes = df["close"].values
    volumes = df["volume"].values
    vol_ma = df["volume"].rolling(20).mean().values

    dbw_signal = [None] * n
    dbw_neckline = np.full(n, np.nan)
    dbw_bottom1 = np.full(n, np.nan, dtype=float)
    dbw_bottom2 = np.full(n, np.nan, dtype=float)

    for i in range(lookback * 2, n):
        window = slice(i - lookback * 2, i)
        w_lows = lows[window]
        w_highs = highs[window]

        # Find local minima (potential bottoms)
        bottoms = []
        for j in range(2, len(w_lows) - 2):
            if (w_lows[j] <= w_lows[j-1] and w_lows[j] <= w_lows[j-2] and
                w_lows[j] <= w_lows[j+1] and w_lows[j] <= w_lows[j+2]):
                bottoms.append((j, w_lows[j]))

        if len(bottoms) < 2:
            continue

        # Check pairs of bottoms for W-pattern
        for b1_idx in range(len(bottoms) - 1):
            b1_pos, b1_price = bottoms[b1_idx]
            for b2_idx in range(b1_idx + 1, len(bottoms)):
                b2_pos, b2_price = bottoms[b2_idx]

                # Bottoms should be at similar levels
                if abs(b1_price - b2_price) / b1_price > tolerance:
                    continue

                # Must have some separation
                if b2_pos - b1_pos < lookback // 3:
                    continue

                # Find the neckline (highest point between bottoms)
                mid_highs = w_highs[b1_pos:b2_pos + 1]
                neckline = np.max(mid_highs)

                # Check if current candle breaks above neckline
                if closes[i] > neckline:
                    vol_ok = True
                    if vol_confirm and vol_ma[i] > 0:
                        vol_ok = volumes[i] / vol_ma[i] >= 1.2

                    if vol_ok:
                        dbw_signal[i] = "buy"
                        dbw_neckline[i] = neckline
                        actual_b1 = i - lookback * 2 + b1_pos
                        actual_b2 = i - lookback * 2 + b2_pos
                        dbw_bottom1[i] = float(actual_b1)
                        dbw_bottom2[i] = float(actual_b2)
                        break
            if dbw_signal[i] == "buy":
                break

    df = df.copy()
    df["dbw_signal"] = dbw_signal
    df["dbw_neckline"] = dbw_neckline
    df["dbw_bottom1_idx"] = dbw_bottom1
    df["dbw_bottom2_idx"] = dbw_bottom2

    return df


def calculate_dbw_levels(df: pd.DataFrame, idx: int,
                         params: dict = None) -> dict:
    """Calculate trade levels for a DBW breakout.

    Stop is ATR-based just below the pattern bottom (volatility-aware).
    Target uses a configurable multiplier on the pattern height for more
    aggressive profit targets (default 1.5× height above neckline).
    """
    import numpy as np

    if params is None:
        params = {}

    dbw = params if "sl_atr_multiplier" in params else params.get("dbw", params)
    sl_atr_mult = dbw.get("sl_atr_multiplier", 1.5)
    target_multiplier = dbw.get("target_multiplier", 1.5)

    entry = df.iloc[idx]["close"]
    neckline = df.iloc[idx]["dbw_neckline"]
    bottom1_idx = int(df.iloc[idx]["dbw_bottom1_idx"])
    bottom2_idx = int(df.iloc[idx]["dbw_bottom2_idx"])
    bottom_price = min(df.iloc[bottom1_idx]["low"], df.iloc[bottom2_idx]["low"])

    atr = df.iloc[idx].get("atr_14", entry * 0.02)
    if np.isnan(atr) or atr <= 0:
        atr = entry * 0.02

    pattern_height = neckline - bottom_price
    target = neckline + (pattern_height * target_multiplier)
    stop_loss = bottom_price - (atr * sl_atr_mult)
    risk = entry - stop_loss
    risk_reward = (target - entry) / risk if risk > 0 else 0

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "target": target,
        "risk": risk,
        "risk_reward": risk_reward,
        "neckline": neckline,
        "bottom_price": bottom_price,
        "pattern_height": pattern_height,
    }
