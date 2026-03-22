"""Download OHLCV data from Tokocrypto via CCXT and store in SQLite."""
import argparse
import time
from datetime import datetime, timedelta

import ccxt

from src.config import load_config
from src.data.database import get_connection, init_db, upsert_ohlcv


def create_exchange(config: dict = None) -> ccxt.Exchange:
    """Create a CCXT exchange instance for Tokocrypto."""
    if config is None:
        config = load_config()

    exc_cfg = config["exchange"]

    # Tokocrypto is Binance-powered — CCXT supports it directly
    # If 'tokocrypto' isn't available in your ccxt version, fall back to binance
    exchange_id = exc_cfg.get("name", "tokocrypto")
    try:
        exchange_class = getattr(ccxt, exchange_id)
    except AttributeError:
        print(f"Exchange '{exchange_id}' not found in ccxt, falling back to 'binance'")
        exchange_class = ccxt.binance

    exchange = exchange_class({
        "apiKey": exc_cfg.get("api_key", ""),
        "secret": exc_cfg.get("secret", ""),
        "enableRateLimit": exc_cfg.get("rate_limit", True),
        "options": {
            "defaultType": "spot",
        },
    })

    return exchange


def download_ohlcv(pair: str, timeframe: str, days: int = 365,
                   exchange: ccxt.Exchange = None, config: dict = None):
    """Download historical OHLCV data and store in SQLite.

    Args:
        pair: Trading pair like 'BTC/IDR'
        timeframe: Candle timeframe like '1h', '4h', '1d'
        days: How many days of history to fetch
        exchange: Optional pre-created exchange instance
        config: Optional config dict
    """
    if config is None:
        config = load_config()
    if exchange is None:
        exchange = create_exchange(config)

    init_db()
    conn = get_connection()

    # Calculate time range
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    now = int(datetime.utcnow().timestamp() * 1000)

    # Map timeframe to milliseconds for pagination
    tf_ms = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
        "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000, "1w": 604_800_000,
    }
    candle_ms = tf_ms.get(timeframe, 3_600_000)
    batch_size = 500  # Most exchanges return max 500-1000 per request

    print(f"Downloading {pair} {timeframe} data ({days} days)...")
    total_candles = 0
    current_since = since

    while current_since < now:
        try:
            candles = exchange.fetch_ohlcv(
                symbol=pair,
                timeframe=timeframe,
                since=current_since,
                limit=batch_size
            )
        except ccxt.NetworkError as e:
            print(f"  Network error, retrying in 5s: {e}")
            time.sleep(5)
            continue
        except ccxt.ExchangeError as e:
            print(f"  Exchange error: {e}")
            break

        if not candles:
            break

        count = upsert_ohlcv(conn, pair, timeframe, candles)
        total_candles += len(candles)

        last_ts = candles[-1][0]
        last_dt = datetime.utcfromtimestamp(last_ts / 1000)
        print(f"  Fetched {len(candles)} candles up to {last_dt.isoformat()}"
              f" (total: {total_candles})")

        # Move past the last candle
        current_since = last_ts + candle_ms

        # Respect rate limits
        time.sleep(exchange.rateLimit / 1000)

    conn.close()
    print(f"Done. {total_candles} candles stored for {pair} {timeframe}")
    return total_candles


def download_all(config: dict = None):
    """Download data for all configured pairs and timeframes."""
    if config is None:
        config = load_config()

    exchange = create_exchange(config)
    pairs = config["trading"]["pairs"]
    timeframes = config["trading"]["timeframes"]
    days = config["trading"].get("history_days", 365)

    for pair in pairs:
        for tf in timeframes:
            download_ohlcv(pair, tf, days, exchange=exchange, config=config)
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OHLCV data")
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Trading pairs (e.g. BTC/IDR ETH/IDR)")
    parser.add_argument("--timeframe", default="1h",
                        help="Candle timeframe (1h, 4h, 1d)")
    parser.add_argument("--days", type=int, default=365,
                        help="Days of history to download")
    args = parser.parse_args()

    config = load_config()
    exchange = create_exchange(config)

    pairs = args.pairs or config["trading"]["pairs"]
    for pair in pairs:
        download_ohlcv(pair, args.timeframe, args.days,
                       exchange=exchange, config=config)
