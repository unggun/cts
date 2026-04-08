"""Bollinger Band Squeeze breakout strategy (TTM Squeeze).

Detects volatility compression using the TTM Squeeze method: Bollinger Bands
contract inside Keltner Channels, indicating a coiling market about to make
a big move. Entry on squeeze release when price breaks above the upper BB.

The TTM Squeeze is the gold standard for volatility breakout detection,
widely used by institutional traders. It works especially well on crypto
where volatility cycles are pronounced.

- Squeeze: BB inside KC (BB upper < KC upper AND BB lower > KC lower)
- Entry: Squeeze releases AND price breaks above upper BB with volume
- Stop: Lower Keltner Channel or ATR-based, whichever is wider
- Exit: Trailing stop (breakouts can run far)
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


def detect_bb_squeeze(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Detect Bollinger Band Squeeze breakout signals.

    A buy signal is generated when:
    1. BB has been inside KC for at least min_squeeze_bars (squeeze)
    2. Squeeze releases (BB expands outside KC)
    3. Price breaks above upper Bollinger Band (upward breakout)
    4. Optional volume confirmation

    Args:
        df: OHLCV DataFrame
        params: Strategy parameters

    Returns:
        DataFrame with added columns: bb_upper, bb_lower, bb_mid,
        kc_upper, kc_lower, squeeze, atr_14, bbs_signal
    """
    df = df.copy()
    if params is None:
        params = {}

    cfg = params if "bb_period" in params else params.get("bb_squeeze", params)
    bb_period = cfg.get("bb_period", 20)
    bb_std = cfg.get("bb_std", 2.0)
    kc_period = cfg.get("kc_period", 20)
    kc_atr_mult = cfg.get("kc_atr_mult", 1.5)
    min_squeeze_bars = cfg.get("min_squeeze_bars", 3)
    volume_confirmation = cfg.get("volume_confirmation", True)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values

    # Bollinger Bands: SMA +/- N std devs
    bb_mid = pd.Series(closes).rolling(window=bb_period).mean().values
    bb_std_vals = pd.Series(closes).rolling(window=bb_period).std().values
    bb_upper = bb_mid + (bb_std * bb_std_vals)
    bb_lower = bb_mid - (bb_std * bb_std_vals)

    # Keltner Channels: EMA +/- N * ATR
    kc_mid = _compute_ema(closes, kc_period)
    atr = _compute_atr(highs, lows, closes, 14)
    kc_upper = kc_mid + (kc_atr_mult * atr)
    kc_lower = kc_mid - (kc_atr_mult * atr)

    # Squeeze detection: BB inside KC
    squeeze = np.zeros(len(df), dtype=int)
    for i in range(50, len(df)):
        if (not np.isnan(bb_upper[i]) and not np.isnan(kc_upper[i])):
            if bb_upper[i] < kc_upper[i] and bb_lower[i] > kc_lower[i]:
                squeeze[i] = 1

    # Count consecutive squeeze bars
    squeeze_count = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        if squeeze[i] == 1:
            squeeze_count[i] = squeeze_count[i - 1] + 1
        else:
            squeeze_count[i] = 0

    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["bb_mid"] = bb_mid
    df["kc_upper"] = kc_upper
    df["kc_lower"] = kc_lower
    df["squeeze"] = squeeze
    df["squeeze_count"] = squeeze_count
    df["atr_14"] = atr
    df["bbs_signal"] = None

    vol_ma20 = pd.Series(volumes).rolling(20).mean().values

    for i in range(51, len(df)):
        if np.isnan(bb_upper[i]) or np.isnan(kc_upper[i]):
            continue

        # Squeeze just released: was in squeeze, now out
        was_squeezing = squeeze_count[i - 1] >= min_squeeze_bars
        squeeze_released = squeeze[i] == 0 and was_squeezing

        if squeeze_released:
            # Price breaking above upper BB confirms upward breakout
            if closes[i] > bb_upper[i]:
                # Volume confirmation
                if volume_confirmation and vol_ma20[i] > 0:
                    if volumes[i] / vol_ma20[i] < 0.8:
                        continue

                df.iloc[i, df.columns.get_loc("bbs_signal")] = "buy"

    return df


def calculate_bb_squeeze_levels(df: pd.DataFrame, idx: int,
                                params: dict = None) -> dict:
    """Calculate trade levels for a BB squeeze breakout signal.

    Stop at lower Keltner Channel or ATR-based, whichever is wider.
    Target uses risk-reward ratio — trailing stop does the heavy lifting
    for capturing extended breakout moves.
    """
    if params is None:
        params = {}

    cfg = params if "sl_atr_multiplier" in params else params.get("bb_squeeze", params)
    sl_atr_mult = cfg.get("sl_atr_multiplier", 5.0)
    min_rr = cfg.get("min_risk_reward", 1.5)

    close = df.iloc[idx]["close"]
    atr = df.iloc[idx].get("atr_14", close * 0.02)
    if np.isnan(atr) or atr <= 0:
        atr = close * 0.02

    kc_lower = df.iloc[idx].get("kc_lower", close - atr * 2)
    if np.isnan(kc_lower):
        kc_lower = close - atr * 2

    entry = close

    # Stop: lower KC or ATR-based, whichever gives more room
    sl_kc = kc_lower
    sl_atr = entry - (atr * sl_atr_mult)
    stop_loss = min(sl_kc, sl_atr)

    risk = entry - stop_loss
    target = entry + (risk * min_rr)
    risk_reward = (target - entry) / risk if risk > 0 else 0

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "target": target,
        "risk": risk,
        "risk_reward": risk_reward,
        "squeeze_bars": int(df.iloc[idx].get("squeeze_count", 0)),
        "bb_upper": df.iloc[idx].get("bb_upper", None),
        "kc_lower": kc_lower,
        "atr": atr,
    }
