"""Darvas Box pattern detection.

The Darvas Box method identifies consolidation zones where price trades within
a defined range for a period (4-7 days by default), then triggers when price
breaks above the box top (resistance) with volume confirmation.

Variations:
- Standard: Break above box top after consolidation
- ATH breakout: When the box top IS the all-time high
"""
import pandas as pd
import numpy as np


def detect_darvas_boxes(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Detect Darvas box formations in OHLCV data.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        params: Strategy parameters dict with darvas config

    Returns:
        DataFrame with added columns:
        - box_top: Current box resistance level
        - box_bottom: Current box support level
        - box_days: How many candles the current box has lasted
        - is_breakout: True when price breaks above box_top with volume
        - is_ath_breakout: True when breakout is at all-time high
        - signal: 'buy' on breakout, None otherwise
    """
    if params is None:
        params = {
            "min_consolidation_days": 4,
            "max_consolidation_days": 7,
            "volume_ratio_min": 1.2,
        }

    darvas = params if "min_consolidation_days" in params else params.get("darvas", params)
    min_days = darvas.get("min_consolidation_days", 4)
    volume_ratio_min = darvas.get("volume_ratio_min", 1.2)

    n = len(df)
    box_top = np.full(n, np.nan)
    box_bottom = np.full(n, np.nan)
    box_days = np.zeros(n, dtype=int)
    is_breakout = np.zeros(n, dtype=bool)
    is_ath_breakout = np.zeros(n, dtype=bool)

    # Volume moving average (20-period)
    vol_ma = df["volume"].rolling(20).mean().values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values

    # Track the forming box
    current_top = None
    current_bottom = None
    days_in_box = 0
    all_time_high = 0.0

    for i in range(20, n):  # Start after vol_ma is available
        h = highs[i]
        l = lows[i]
        c = closes[i]

        # Update all-time high
        if h > all_time_high:
            all_time_high = h

        if current_top is None:
            # Start a new potential box
            current_top = h
            current_bottom = l
            days_in_box = 1
        else:
            # Check if we're still within the box
            if h <= current_top * 1.001:  # Small tolerance
                # Still inside — update bottom if new low is higher
                if l > current_bottom:
                    pass  # Box bottom stays (we want the lowest low)
                else:
                    current_bottom = l
                days_in_box += 1
            elif h > current_top:
                # Potential breakout — check if box was long enough
                vol_ratio = volumes[i] / vol_ma[i] if vol_ma[i] > 0 else 0

                if days_in_box >= min_days and vol_ratio >= volume_ratio_min:
                    is_breakout[i] = True
                    is_ath_breakout[i] = (abs(current_top - all_time_high) / all_time_high < 0.01)

                # Start new box from this candle
                current_top = h
                current_bottom = l
                days_in_box = 1

        box_top[i] = current_top
        box_bottom[i] = current_bottom
        box_days[i] = days_in_box

    df = df.copy()
    df["box_top"] = box_top
    df["box_bottom"] = box_bottom
    df["box_days"] = box_days
    df["is_breakout"] = is_breakout
    df["is_ath_breakout"] = is_ath_breakout
    df["signal"] = np.where(is_breakout, "buy", None)

    return df


def calculate_trade_levels(df: pd.DataFrame, idx: int,
                           params: dict = None) -> dict:
    """Calculate entry, stop loss, and target for a breakout signal.

    Args:
        df: DataFrame with Darvas columns
        idx: Integer index of the breakout candle
        params: Strategy parameters

    Returns:
        dict with entry, stop_loss, target, risk_reward, is_ath
    """
    if params is None:
        params = {"stop_loss_pct": 0.07, "min_risk_reward": 2.0}

    darvas = params if "stop_loss_pct" in params else params.get("darvas", params)
    stop_loss_pct = darvas.get("stop_loss_pct", 0.07)
    min_rr = darvas.get("min_risk_reward", 2.0)

    entry = df.iloc[idx]["close"]
    box_bottom = df.iloc[idx]["box_bottom"]
    box_top = df.iloc[idx]["box_top"]
    is_ath = df.iloc[idx].get("is_ath_breakout", False)

    # Stop loss below box bottom
    stop_loss = box_bottom * (1 - stop_loss_pct)
    risk = entry - stop_loss

    if is_ath:
        # ATH breakout — let it run, use trailing stop concept
        target = None
        risk_reward = float("inf")
    else:
        # Standard breakout — target = box height projected above breakout
        box_height = box_top - box_bottom
        target = entry + (box_height * min_rr)
        risk_reward = (target - entry) / risk if risk > 0 else 0

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "target": target,
        "risk": risk,
        "risk_reward": risk_reward,
        "is_ath": bool(is_ath),
        "box_top": box_top,
        "box_bottom": box_bottom,
    }
