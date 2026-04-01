"""EMA/SMA crossover detection with multi-timeframe confirmation.

Detects:
- Golden cross (fast EMA crosses above slow EMA) → bullish
- Death cross (fast EMA crosses below slow EMA) → bearish
- Triple EMA alignment (8/21/50) for trend confirmation
- Price + EMA pullback entries (price pulls back to EMA then bounces)

Uses volume and trend strength to filter false crossovers.
"""
import numpy as np
import pandas as pd


def _compute_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA for entire array."""
    ema = np.full(len(data), np.nan)
    if len(data) < period:
        return ema

    multiplier = 2 / (period + 1)
    ema[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        ema[i] = data[i] * multiplier + ema[i - 1] * (1 - multiplier)
    return ema


def _compute_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Compute SMA for entire array."""
    sma = np.full(len(data), np.nan)
    for i in range(period - 1, len(data)):
        sma[i] = np.mean(data[i - period + 1:i + 1])
    return sma


def detect_ema_crossovers(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Detect EMA/SMA crossover signals.

    Args:
        df: OHLCV DataFrame
        params: Config with ema_crossover settings

    Returns:
        DataFrame with added columns:
        - ema_fast: fast EMA values
        - ema_slow: slow EMA values
        - ema_trend: trend EMA values (long-term context)
        - ema_signal: 'buy' on golden cross, 'sell' on death cross
        - ema_alignment: 'bullish_aligned', 'bearish_aligned', 'mixed'
        - ema_crossover_type: 'golden_cross', 'death_cross', 'pullback_bounce', etc.
        - ema_signal_strength: 'strong', 'moderate', 'weak'
    """
    if params is None:
        params = {
            "fast_period": 8,
            "slow_period": 21,
            "trend_period": 50,
            "use_ema": True,
            "volume_confirmation": True,
            "require_alignment": False,
            "pullback_enabled": True,
            "pullback_tolerance_atr": 0.3,
        }

    ec = params if "fast_period" in params else params.get("ema_crossover", params)
    fast_p = ec.get("fast_period", 8)
    slow_p = ec.get("slow_period", 21)
    trend_p = ec.get("trend_period", 50)
    use_ema = ec.get("use_ema", True)
    vol_confirm = ec.get("volume_confirmation", True)
    require_align = ec.get("require_alignment", False)
    pullback_on = ec.get("pullback_enabled", True)
    pullback_tol = ec.get("pullback_tolerance_atr", 0.3)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values

    # Compute moving averages
    compute = _compute_ema if use_ema else _compute_sma
    fast = compute(closes, fast_p)
    slow = compute(closes, slow_p)
    trend = compute(closes, trend_p)

    vol_ma = _compute_sma(volumes, 20)

    n = len(df)
    signals = [None] * n
    alignments = [None] * n
    cross_types = [None] * n
    strengths = [None] * n

    # ATR for pullback tolerance
    from src.indicators.features import _atr

    for i in range(max(trend_p, 20) + 1, n):
        if np.isnan(fast[i]) or np.isnan(slow[i]) or np.isnan(trend[i]):
            continue

        # ── EMA alignment ──
        if fast[i] > slow[i] > trend[i]:
            alignments[i] = "bullish_aligned"
        elif fast[i] < slow[i] < trend[i]:
            alignments[i] = "bearish_aligned"
        else:
            alignments[i] = "mixed"

        vol_ratio = volumes[i] / vol_ma[i] if vol_ma[i] and vol_ma[i] > 0 else 1.0

        # ── Crossover detection ──
        # Golden cross: fast was below slow, now above
        if fast[i] > slow[i] and fast[i - 1] <= slow[i - 1]:
            signal = "buy"
            cross_type = "golden_cross"

            # Strength scoring
            if alignments[i] == "bullish_aligned" and vol_ratio > 1.2:
                strength = "strong"
            elif vol_ratio > 1.0:
                strength = "moderate"
            else:
                strength = "weak"

            # Alignment filter
            if require_align and closes[i] < trend[i]:
                signal = None  # Below long-term trend, skip

            if signal:
                signals[i] = signal
                cross_types[i] = cross_type
                strengths[i] = strength

        # Death cross: fast was above slow, now below
        elif fast[i] < slow[i] and fast[i - 1] >= slow[i - 1]:
            signal = "sell"
            cross_type = "death_cross"

            if alignments[i] == "bearish_aligned" and vol_ratio > 1.2:
                strength = "strong"
            elif vol_ratio > 1.0:
                strength = "moderate"
            else:
                strength = "weak"

            if require_align and closes[i] > trend[i]:
                signal = None

            if signal:
                signals[i] = signal
                cross_types[i] = cross_type
                strengths[i] = strength

        # ── Pullback bounce (buy when price pulls back to fast EMA in uptrend) ──
        elif pullback_on and signals[i] is None:
            atr = _atr(highs[:i+1], lows[:i+1], closes[:i+1], 14)
            tol = atr * pullback_tol

            if (alignments[i] == "bullish_aligned" and
                abs(lows[i] - fast[i]) <= tol and
                closes[i] > fast[i] and
                closes[i - 1] > fast[i - 1] * 0.998):
                # Price touched fast EMA from above and bounced
                # Confirm with bullish candle
                if closes[i] > df.iloc[i]["open"]:
                    signals[i] = "buy"
                    cross_types[i] = "pullback_bounce"
                    strengths[i] = "moderate" if vol_ratio > 1.0 else "weak"

            elif (alignments[i] == "bearish_aligned" and
                  abs(highs[i] - fast[i]) <= tol and
                  closes[i] < fast[i] and
                  closes[i - 1] < fast[i - 1] * 1.002):
                if closes[i] < df.iloc[i]["open"]:
                    signals[i] = "sell"
                    cross_types[i] = "pullback_bounce_bear"
                    strengths[i] = "moderate" if vol_ratio > 1.0 else "weak"

    df = df.copy()
    df["ema_fast"] = fast
    df["ema_slow"] = slow
    df["ema_trend"] = trend
    df["ema_signal"] = signals
    df["ema_alignment"] = alignments
    df["ema_crossover_type"] = cross_types
    df["ema_signal_strength"] = strengths

    return df


def calculate_ema_levels(df: pd.DataFrame, idx: int, params: dict = None) -> dict:
    """Calculate trade levels for an EMA crossover signal.

    Stop loss: below the slow EMA (for buys) or above (for sells)
    Target: based on risk-reward ratio
    """
    if params is None:
        params = {"sl_atr_multiplier": 3.5, "min_risk_reward": 2.0}

    ec = params if "sl_atr_multiplier" in params else params.get("ema_crossover", params)
    sl_mult = ec.get("sl_atr_multiplier", 3.5)
    min_rr = ec.get("min_risk_reward", 2.0)

    from src.indicators.features import _atr

    entry = df.iloc[idx]["close"]
    atr = _atr(df["high"].values[:idx+1], df["low"].values[:idx+1],
               df["close"].values[:idx+1], 14)

    signal = df.iloc[idx]["ema_signal"]
    cross_type = df.iloc[idx].get("ema_crossover_type", "")

    slow_ema = df.iloc[idx].get("ema_slow", entry)

    if signal == "buy":
        # Stop below slow EMA or ATR-based, whichever is tighter
        sl_ema = slow_ema - (atr * 0.5)
        sl_atr = entry - (atr * sl_mult)
        stop_loss = max(sl_ema, sl_atr)  # Tighter of the two

        risk = entry - stop_loss
        target = entry + (risk * min_rr)
    elif signal == "sell":
        # For now only generate buy signals for long-only system
        return {"entry": entry, "stop_loss": entry, "target": entry,
                "risk": 0, "risk_reward": 0}
    else:
        return {"entry": entry, "stop_loss": entry, "target": entry,
                "risk": 0, "risk_reward": 0}

    risk_reward = (target - entry) / risk if risk > 0 else 0

    return {
        "entry": entry, "stop_loss": stop_loss, "target": target,
        "risk": abs(risk), "risk_reward": risk_reward,
        "crossover_type": cross_type,
        "alignment": df.iloc[idx].get("ema_alignment"),
        "strength": df.iloc[idx].get("ema_signal_strength"),
        "atr": atr,
    }
