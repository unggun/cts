"""Paper trading engine — simulate live trading without real money.

Connects to Tokocrypto via CCXT for real-time price data but executes
trades only in the local database. This validates the strategy in
live market conditions before risking real capital.
"""
import json
import time
import signal
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.downloader import create_exchange
from src.data.database import get_connection, init_db, load_ohlcv, upsert_ohlcv, load_latest_parameters
from src.backtest.engine import STRATEGY_REGISTRY
from src.indicators.features import extract_features
from src.learning.filter_rules import load_active_filters, apply_filters


class PaperTrader:
    """Live paper trading engine."""

    def __init__(self, strategy: str = "darvas", config: dict = None):
        self.config = config or load_config()
        self.strategy = strategy
        self.exchange = create_exchange(self.config)
        self.running = False

        # Load strategy parameters (prefer learned, fallback to config)
        init_db()
        conn = get_connection()
        self.params = load_latest_parameters(conn, strategy)
        if self.params is None:
            self.params = self.config.get("strategy", {}).get(strategy, {})
        self.params["risk"] = self.config.get("risk", {})
        conn.close()

        # Load active filter rules
        self.filters = load_active_filters(strategy)
        if self.filters:
            print(f"  Loaded {len(self.filters)} active filter rules")

        # State
        self.positions = {}  # pair -> position dict
        self.capital = self.config.get("simulation", {}).get("initial_capital", 10_000_000)
        self.initial_capital = self.capital

        # Restore persisted positions from DB
        self._restore_positions()

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        print("\nShutting down paper trader...")
        self.running = False

    def start(self):
        """Start the paper trading loop."""
        pairs = self.config["trading"]["pairs"]
        timeframe = self.config["trading"]["timeframes"][0]  # Primary timeframe

        print("=" * 60)
        print(f"PAPER TRADER STARTED")
        print(f"Strategy: {self.strategy}")
        print(f"Pairs: {', '.join(pairs)}")
        print(f"Timeframe: {timeframe}")
        print(f"Initial capital: {self.capital:,.0f} IDR")
        print(f"Parameters: {json.dumps(self.params, indent=2, default=str)}")
        print("=" * 60)
        print("Press Ctrl+C to stop.\n")

        self.running = True

        while self.running:
            try:
                for pair in pairs:
                    self._process_pair(pair, timeframe)

                # Status update
                total_equity = self._calculate_equity(pairs)
                pnl = (total_equity - self.initial_capital) / self.initial_capital
                positions_str = ", ".join(
                    f"{p}: {pos['entry_price']:,.0f}"
                    for p, pos in self.positions.items()
                ) or "none"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Equity: {total_equity:,.0f} IDR ({pnl:+.2%}) | "
                      f"Positions: {positions_str}")

                # Sleep until next candle (simplified — sleep for a portion of timeframe)
                sleep_seconds = self._get_sleep_seconds(timeframe)
                time.sleep(sleep_seconds)

            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(30)

        print(f"\nPaper trader stopped. Final equity: {self._calculate_equity(pairs):,.0f} IDR")

    def _process_pair(self, pair: str, timeframe: str):
        """Process a single pair — check exits and entries."""
        conn = get_connection()

        # Fetch latest candles with retry
        candles = None
        for attempt in range(3):
            try:
                candles = self.exchange.fetch_ohlcv(pair, timeframe, limit=200)
                break
            except Exception as e:
                if attempt < 2:
                    wait = (attempt + 1) * 5
                    print(f"  Retry {attempt+1}/3 fetching {pair} in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    print(f"  Failed fetching {pair} after 3 attempts: {e}")
                    conn.close()
                    return

        if candles:
            upsert_ohlcv(conn, pair, timeframe, candles)

        # Load data from DB for analysis
        df = load_ohlcv(conn, pair, timeframe)
        if len(df) < 60:
            conn.close()
            return

        # Apply pattern detection using registry
        if self.strategy not in STRATEGY_REGISTRY:
            conn.close()
            return

        detect_fn, signal_col, levels_fn = STRATEGY_REGISTRY[self.strategy]
        df = detect_fn(df, self.params)

        last_idx = len(df) - 1
        current_price = df.iloc[last_idx]["close"]

        # ── Check exit for open position ──
        if pair in self.positions:
            pos = self.positions[pair]
            low = df.iloc[last_idx]["low"]
            high = df.iloc[last_idx]["high"]

            exit_price = None
            exit_reason = None

            if low <= pos["stop_loss"]:
                exit_price = pos["stop_loss"]
                exit_reason = "stop_loss"
            elif pos.get("target") and high >= pos["target"]:
                exit_price = pos["target"]
                exit_reason = "target"

            # ATH trailing stop
            if pos.get("is_ath") and exit_price is None:
                from src.indicators.features import _atr
                atr = _atr(df["high"].values, df["low"].values, df["close"].values, 14)
                trailing = high - (2 * atr)
                if trailing > pos["stop_loss"]:
                    pos["stop_loss"] = trailing

            if exit_price:
                self._close_position(conn, pair, exit_price, exit_reason)

        # ── Check entry ──
        if pair not in self.positions:
            if df.iloc[last_idx].get(signal_col) == "buy":
                strategy_params = self.params.get(self.strategy, self.params)
                levels = levels_fn(df, last_idx, strategy_params)

                min_rr = strategy_params.get("min_risk_reward", 2.0)
                if levels["risk_reward"] >= min_rr or levels.get("target") is None:
                    # Check max positions
                    max_pos = self.params.get("risk", {}).get("max_positions", 3)
                    if len(self.positions) < max_pos:
                        features = extract_features(df, last_idx)

                        # Apply filter rules
                        if self.filters:
                            should_skip, skip_reason = apply_filters(features, self.filters)
                            if should_skip:
                                print(f"  ⏭️ FILTERED {pair}: {skip_reason}")
                                conn.close()
                                return

                        self._open_position(conn, pair, levels, features)

        conn.close()

    def _open_position(self, conn, pair: str, levels: dict, features: dict):
        """Open a paper trading position."""
        max_pos_pct = self.params.get("risk", {}).get("max_position_pct", 0.05)
        risk_amount = self.capital * max_pos_pct

        if levels["risk"] > 0:
            position_size = risk_amount / levels["risk"]
        else:
            return

        cost = position_size * levels["entry"]
        if cost > self.capital * 0.95:
            position_size = self.capital * 0.95 / levels["entry"]

        self.positions[pair] = {
            "entry_time": datetime.now().isoformat(),
            "entry_price": levels["entry"],
            "stop_loss": levels["stop_loss"],
            "target": levels.get("target"),
            "position_size": position_size,
            "is_ath": levels.get("is_ath", False),
            "features": features,
        }

        # Persist to DB
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trade_journal
            (mode, pair, strategy, direction, entry_time, entry_price,
             stop_loss, take_profit, position_size, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "paper", pair, self.strategy, "long",
            self.positions[pair]["entry_time"], levels["entry"],
            levels["stop_loss"], levels.get("target"), position_size,
            json.dumps(features)
        ))
        conn.commit()

        target_str = f"{levels['target']:,.0f}" if levels.get("target") else "ATH trail"
        print(f"  📈 OPENED {pair} @ {levels['entry']:,.0f} | "
              f"SL: {levels['stop_loss']:,.0f} | Target: {target_str} | "
              f"Size: {position_size:.6f}")

        self._notify(
            f"📈 *PAPER BUY {pair}*\n"
            f"Strategy: {self.strategy}\n"
            f"Entry: {levels['entry']:,.0f}\n"
            f"Stop loss: {levels['stop_loss']:,.0f}\n"
            f"Target: {target_str}\n"
            f"Size: {position_size:.6f}"
        )

    def _close_position(self, conn, pair: str, exit_price: float, reason: str):
        """Close a paper trading position and journal it."""
        pos = self.positions.pop(pair)

        pnl_per_unit = exit_price - pos["entry_price"]
        pnl_absolute = pnl_per_unit * pos["position_size"]
        pnl_pct = pnl_per_unit / pos["entry_price"]
        self.capital += pnl_absolute

        emoji = "✅" if pnl_pct > 0 else "❌"
        print(f"  {emoji} CLOSED {pair} @ {exit_price:,.0f} | "
              f"PnL: {pnl_pct:+.2%} ({pnl_absolute:+,.0f} IDR) | "
              f"Reason: {reason}")

        # Update the open journal entry with exit info
        cur = conn.cursor()
        cur.execute("""
            UPDATE trade_journal SET
                exit_time=?, exit_price=?, exit_reason=?,
                pnl_pct=?, pnl_absolute=?
            WHERE mode='paper' AND pair=? AND strategy=?
                AND exit_time IS NULL
            ORDER BY created_at DESC LIMIT 1
        """, (
            datetime.now().isoformat(), exit_price, reason,
            pnl_pct, pnl_absolute, pair, self.strategy
        ))
        conn.commit()

        self._notify(
            f"{emoji} *PAPER SELL {pair}* ({reason})\n"
            f"Strategy: {self.strategy}\n"
            f"Exit: {exit_price:,.0f}\n"
            f"PnL: {pnl_pct:+.2%} ({pnl_absolute:+,.0f} IDR)"
        )

    def _restore_positions(self):
        """Restore open positions from DB on startup."""
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT pair, entry_time, entry_price, stop_loss, take_profit,
                   position_size, features_json
            FROM trade_journal
            WHERE mode='paper' AND strategy=? AND exit_time IS NULL
        """, (self.strategy,))
        rows = cur.fetchall()
        for row in rows:
            self.positions[row["pair"]] = {
                "entry_time": row["entry_time"],
                "entry_price": row["entry_price"],
                "stop_loss": row["stop_loss"],
                "target": row["take_profit"],
                "position_size": row["position_size"],
                "is_ath": False,
                "features": json.loads(row["features_json"] or "{}"),
            }
        conn.close()
        if self.positions:
            print(f"  Restored {len(self.positions)} open positions: "
                  f"{', '.join(self.positions.keys())}")

    def _notify(self, message: str):
        """Send Telegram notification."""
        tg = self.config.get("notifications", {}).get("telegram", {})
        if not tg.get("enabled"):
            return
        token = tg.get("bot_token", "")
        chat_id = tg.get("chat_id", "")
        if not token or not chat_id or "YOUR" in token:
            return
        try:
            import requests
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
                timeout=10,
            )
        except Exception:
            pass

    def _calculate_equity(self, pairs: list) -> float:
        """Calculate total equity including unrealized P&L."""
        equity = self.capital
        for pair, pos in self.positions.items():
            try:
                ticker = self.exchange.fetch_ticker(pair)
                current_price = ticker["last"]
                unrealized = (current_price - pos["entry_price"]) * pos["position_size"]
                equity += unrealized
            except Exception:
                pass
        return equity

    def _get_sleep_seconds(self, timeframe: str) -> int:
        """Get sleep duration based on timeframe."""
        tf_seconds = {
            "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 600,   # Check every 10 min for 1h candles
            "4h": 1800,  # Check every 30 min for 4h candles
            "1d": 3600,  # Check every hour for daily
        }
        return tf_seconds.get(timeframe, 600)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Paper trading")
    parser.add_argument("--strategy", default="darvas",
                        choices=["darvas", "dbw", "candlestick", "sr_breakout", "ma_crossover", "rsi_vwap", "cvd", "macd"])
    args = parser.parse_args()

    trader = PaperTrader(strategy=args.strategy)
    trader.start()
