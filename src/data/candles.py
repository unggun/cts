"""Candle timing helpers.

``fetch_ohlcv`` returns the still-forming candle as its last row. Signals
computed on it can appear and vanish within the hour, while the backtest
only ever sees closed bars, so entry decisions must use closed candles.
"""
from datetime import datetime, timezone

_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


def timeframe_seconds(timeframe: str) -> int:
    """'15m' -> 900, '1h' -> 3600, '4h' -> 14400, '1d' -> 86400."""
    return int(timeframe[:-1]) * _UNIT_SECONDS[timeframe[-1]]


def last_closed_index(df, timeframe: str, now: datetime = None):
    """Positional index of the newest fully closed candle, or None.

    ``df`` is indexed by candle open time as naive UTC (as ``load_ohlcv``
    returns it); ``now`` defaults to the current UTC time.
    """
    if now is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
    duration = timeframe_seconds(timeframe)
    for i in range(len(df) - 1, -1, -1):
        opened = df.index[i].to_pydatetime()
        if (now - opened).total_seconds() >= duration:
            return i
    return None
