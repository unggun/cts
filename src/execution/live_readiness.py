"""Evaluate per-strategy readiness to flip from paper to live trading."""
import json
from datetime import datetime, timedelta

from src.data.database import get_connection, init_db

KNOWN_STRATEGIES = [
    "darvas", "dbw", "candlestick", "macd", "ma_crossover",
    "mean_reversion", "sr_breakout", "cvd", "bb_squeeze", "rsi_vwap",
]

MIN_MC_TRADES = 50
MAX_RUIN_PROB = 0.03
MIN_PAPER_TRADES = 30
MIN_PAPER_WR = 0.50
PAPER_LOOKBACK_DAYS = 30
