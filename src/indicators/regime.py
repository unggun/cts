"""Higher-timeframe trend context for each candle (market regime).

4h and daily candles are built from the 1h data, so backtests and paper
trading use the same source (the stored 4h/1d candles lag behind). A 1h bar
only sees higher-timeframe candles that had closed by the time the 1h bar
closed: no look-ahead.

Columns added (NaN until enough history):
    st_4h_up, st_1d_up   Supertrend(10, 3) direction: 1 up, 0 down
    adx_4h               ADX(14) on 4h candles (trend strength)
    btc_st_4h_up         BTC's 4h Supertrend direction
    btc_return_24h       BTC's return over the last 24 hours
"""
import numpy as np
import pandas as pd

from src.data.candles import timeframe_seconds

REGIME_COLUMNS = ["st_4h_up", "st_1d_up", "adx_4h", "btc_st_4h_up", "btc_return_24h"]
DEFAULT_REGIME_PAIR = "BTC/IDR"


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """OHLCV candles of a higher timeframe, indexed by open time."""
    out = df[["open", "high", "low", "close", "volume"]].resample(
        rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna(subset=["close"])


def _rma(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder's moving average (NaN until ``period`` values)."""
    out = np.full(len(x), np.nan)
    if len(x) < period:
        return out
    out[period - 1] = np.nanmean(x[:period])
    for i in range(period, len(x)):
        out[i] = (out[i - 1] * (period - 1) + x[i]) / period
    return out


def _true_range(h, l, c):
    prev = np.concatenate([[np.nan], c[:-1]])
    return np.nanmax(np.vstack([h - l, np.abs(h - prev), np.abs(l - prev)]), axis=0)


def supertrend_up(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> np.ndarray:
    """Supertrend direction per candle: 1.0 up, 0.0 down, NaN during warm-up."""
    h, l, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    atr = _rma(_true_range(h, l, c), period)
    mid = (h + l) / 2
    upper, lower = mid + mult * atr, mid - mult * atr
    direction = np.full(len(c), np.nan)
    fu = fl = np.nan
    up = True
    for i in range(len(c)):
        if np.isnan(atr[i]):
            continue
        if np.isnan(fu):                       # first bar with an ATR
            fu, fl = upper[i], lower[i]
            up = c[i] >= mid[i]
        else:
            fu = upper[i] if (upper[i] < fu or c[i - 1] > fu) else fu
            fl = lower[i] if (lower[i] > fl or c[i - 1] < fl) else fl
            if up and c[i] < fl:
                up = False
            elif not up and c[i] > fu:
                up = True
        direction[i] = 1.0 if up else 0.0
    return direction


def adx(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Average Directional Index (Wilder)."""
    h, l, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    up_move = np.diff(h, prepend=np.nan)
    down_move = -np.diff(l, prepend=np.nan)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = _true_range(h, l, c)
    # Skip the first bar (no previous candle) before smoothing.
    atr = np.concatenate([[np.nan], _rma(tr[1:], period)])
    pdi = 100 * np.concatenate([[np.nan], _rma(plus_dm[1:], period)]) / atr
    mdi = 100 * np.concatenate([[np.nan], _rma(minus_dm[1:], period)]) / atr
    with np.errstate(invalid="ignore", divide="ignore"):
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    start = np.flatnonzero(~np.isnan(dx))
    out = np.full(len(c), np.nan)
    if len(start):
        s = start[0]
        out[s:] = _rma(np.nan_to_num(dx[s:]), period)
    return out


def _as_of(df: pd.DataFrame, htf: pd.DataFrame, rule: str, values: np.ndarray,
           timeframe: str) -> np.ndarray:
    """Value of the newest higher-timeframe candle closed when each bar closed."""
    if htf.empty or df.empty:
        return np.full(len(df), np.nan)
    avail = pd.DataFrame({"t": htf.index + pd.Timedelta(rule), "v": values}).dropna(subset=["v"])
    bars = pd.DataFrame({"t": df.index + pd.Timedelta(seconds=timeframe_seconds(timeframe)),
                         "pos": np.arange(len(df))})
    merged = pd.merge_asof(bars.sort_values("t"), avail.sort_values("t"), on="t",
                           direction="backward")
    return merged.sort_values("pos")["v"].to_numpy(dtype=float)


def add_regime_columns(df: pd.DataFrame, btc_df: pd.DataFrame = None,
                       timeframe: str = "1h") -> pd.DataFrame:
    """Return ``df`` with the REGIME_COLUMNS added (see module docstring)."""
    df = df.copy()
    if df.empty:
        for col in REGIME_COLUMNS:
            df[col] = np.nan
        return df
    h4, d1 = resample(df, "4h"), resample(df, "1D")
    df["st_4h_up"] = _as_of(df, h4, "4h", supertrend_up(h4), timeframe)
    df["st_1d_up"] = _as_of(df, d1, "1D", supertrend_up(d1), timeframe)
    df["adx_4h"] = _as_of(df, h4, "4h", adx(h4), timeframe)

    if btc_df is None or btc_df.empty:
        df["btc_st_4h_up"] = np.nan
        df["btc_return_24h"] = np.nan
        return df
    b4 = resample(btc_df, "4h")
    df["btc_st_4h_up"] = _as_of(df, b4, "4h", supertrend_up(b4), timeframe)
    # BTC 1h bars close at the same time as this pair's bars.
    close = btc_df["close"].astype(float)
    ago = close.reindex(close.index - pd.Timedelta(hours=24)).to_numpy()
    ret = pd.Series(close.to_numpy() / ago - 1, index=close.index)
    tf = timeframe_seconds(timeframe)
    df["btc_return_24h"] = _as_of(df, ret.to_frame("close"), f"{tf}s", ret.to_numpy(), timeframe)
    return df


def regime_pair(config: dict) -> str:
    return (config.get("trading", {}) or {}).get("regime_pair", DEFAULT_REGIME_PAIR)
