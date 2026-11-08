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
from src.data.database import (
    get_connection, init_db, load_ohlcv, upsert_ohlcv, load_latest_parameter_row,
)
from src.backtest.engine import STRATEGY_REGISTRY
from src.backtest.costs import net_pnl, trading_costs
from src.backtest.exits import trail_stop
from src.data.candles import last_closed_index
from src.indicators.features import extract_features
from src.indicators.regime import add_regime_columns, regime_pair
from src.learning.filter_rules import load_active_filters, apply_filters


class PaperTrader:
    """Live paper trading engine."""

    def __init__(self, strategy: str = "darvas", config: dict = None):
        self.config = config or load_config()
        self.strategy = strategy
        self.exchange = create_exchange(self.config)
        self.running = False

        # Load strategy parameters (prefer learned, fallback to config) and
        # the active filter rules
        init_db()
        row, self.filters = self._read_learned_state()
        self._apply_params(row)
        if self.filters:
            print(f"  Loaded {len(self.filters)} active filter rules")

        # State
        self.positions = {}  # pair -> position dict
        self._last_signal_bar = {}  # pair -> closed candle already acted on
        self.costs = trading_costs(self.config)
        self.capital =self.config.get("simulation", {}).get("initial_capital", 10_000_000)
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
                self._refresh_learned_state()
                self._run_cycle(pairs, timeframe)

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

    def _run_cycle(self, pairs: list, timeframe: str):
        """Process every pair once, the regime pair (BTC) first: every pair's
        btc_* features then see its candles from this cycle, not the last."""
        btc_pair = regime_pair(self.config)
        if btc_pair not in pairs:
            self._refresh_candles(btc_pair, timeframe)
        for pair in sorted(pairs, key=lambda p: p != btc_pair):
            self._process_pair(pair, timeframe)

    def _refresh_candles(self, pair: str, timeframe: str):
        """Fetch and store the latest candles of a pair that isn't traded."""
        conn = get_connection()
        try:
            candles = self._fetch_candles(pair, timeframe)
            if candles:
                upsert_ohlcv(conn, pair, timeframe, candles)
        finally:
            conn.close()

    def _fetch_candles(self, pair: str, timeframe: str):
        """Latest candles from the exchange (3 tries), or None."""
        for attempt in range(3):
            try:
                return self.exchange.fetch_ohlcv(pair, timeframe, limit=200)
            except Exception as e:
                if attempt < 2:
                    wait = (attempt + 1) * 5
                    print(f"  Retry {attempt+1}/3 fetching {pair} in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    print(f"  Failed fetching {pair} after 3 attempts: {e}")
        return None

    def _process_pair(self, pair: str, timeframe: str):
        """Process a single pair — check exits and entries."""
        conn = get_connection()

        candles = self._fetch_candles(pair, timeframe)
        if candles is None:
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
        # Same higher-timeframe/BTC context as the backtest (regime features).
        btc_pair = regime_pair(self.config)
        btc_df = df if pair == btc_pair else load_ohlcv(conn, btc_pair, timeframe)
        df = add_regime_columns(df, btc_df, timeframe)

        last_idx = len(df) - 1
        current_price = df.iloc[last_idx]["close"]
        strategy_params = self.params.get(self.strategy, self.params)

        # ── Check exit for open position ──
        # Exits use the live (possibly still-forming) candle's high/low,
        # like a resting stop/target order would.
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

            if exit_price:
                self._close_position(conn, pair, exit_price, exit_reason)
            else:
                # Same ATR trailing stop as the backtest engine, for all positions.
                from src.indicators.features import _atr
                pos["highest_high"] = max(pos.get("highest_high", pos["entry_price"]), high)
                atr = _atr(df["high"].values, df["low"].values, df["close"].values, 14)
                trail_mult = strategy_params.get("trailing_atr_multiplier", 3.0)
                new_stop = trail_stop(pos["entry_price"], pos["stop_loss"],
                                      pos["highest_high"], atr, trail_mult)
                if new_stop > pos["stop_loss"]:
                    pos["stop_loss"] = new_stop
                    self._persist_stop(conn, pos)
                    print(f"  ↗ {pair} trailing stop raised to {new_stop:,.0f}")

        # ── Check entry ──
        # Signals are only acted on once their candle has closed, matching the
        # backtest; acting on the forming candle lets signals repaint.
        signal_idx = last_closed_index(df, timeframe)
        if signal_idx is None:
            conn.close()
            return
        signal_bar = df.index[signal_idx]

        if pair not in self.positions and self._last_signal_bar.get(pair) != signal_bar:
            if df.iloc[signal_idx].get(signal_col) == "buy":
                self._last_signal_bar[pair] = signal_bar
                levels = dict(levels_fn(df, signal_idx, strategy_params))
                # Fill at the current price, keeping the signal's stop/target.
                levels["entry"] = current_price
                levels["risk"] = current_price - levels["stop_loss"]
                if levels.get("target") is not None and levels["risk"] > 0:
                    levels["risk_reward"] = (levels["target"] - current_price) / levels["risk"]

                min_rr = strategy_params.get("min_risk_reward", 2.0)
                if levels["risk"] <= 0:
                    print(f"  ⏭️ SKIPPED {pair}: price already below signal stop")
                elif levels["risk_reward"] >= min_rr or levels.get("target") is None:
                    # Check max positions
                    max_pos = self.params.get("risk", {}).get("max_positions", 3)
                    if len(self.positions) < max_pos:
                        features = extract_features(df, signal_idx)

                        # Apply filter rules
                        if self.filters:
                            should_skip, skip_reason = apply_filters(features, self.filters)
                            if should_skip:
                                print(f"  ⏭️ FILTERED {pair}: {skip_reason}")
                                conn.close()
                                return

                        self._open_position(conn, pair, levels, features)

        conn.close()

    def _read_learned_state(self):
        """Newest parameter row and active filters, read from one DB snapshot.

        A promotion writes both in one transaction; reading them together
        means the trader never mixes new params with old filters.
        """
        conn = get_connection()
        try:
            conn.execute("BEGIN")
            row = load_latest_parameter_row(conn, self.strategy)
            filters = load_active_filters(self.strategy, conn=conn)
            conn.commit()
        finally:
            conn.close()
        return row, filters

    def _apply_params(self, row):
        """Use a saved parameter row, or the config params when none is saved."""
        if row is None:
            params = dict(self.config.get("strategy", {}).get(self.strategy, {}))
        else:
            params = row["parameters"]
        params["risk"] = self.config.get("risk", {})
        self.params = params
        self.param_row_id = row["id"] if row else None
        self.param_version = row["version"] if row else 0

    def _refresh_learned_state(self):
        """Pick up promoted parameter versions and filter changes without a restart."""
        row, filters = self._read_learned_state()
        # Compare row ids, not versions: a reset can make a new row reuse a version.
        if (row["id"] if row else None) != self.param_row_id:
            self._apply_params(row)
            label = (f"parameter version {self.param_version}" if row
                     else "config parameters (no saved version)")
            print(f"  🔄 Loaded {label} for {self.strategy}")
            self._notify(f"🔄 *PAPER {self.strategy}*: now trading {label}")

        key = lambda fs: [(f["feature"], f["operator"], str(f["threshold"])) for f in fs]
        if key(filters) != key(self.filters):
            self.filters = filters
            print(f"  🔄 Filter rules updated ({len(filters)} active)")

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
            "highest_high": levels["entry"],
            "features": features,
        }

        # Persist to DB
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trade_journal
            (mode, pair, strategy, direction, entry_time, entry_price,
             stop_loss, take_profit, position_size, features_json,
             protection_mode, param_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "paper", pair, self.strategy, "long",
            self.positions[pair]["entry_time"], levels["entry"],
            levels["stop_loss"], levels.get("target"), position_size,
            json.dumps(features), "simulated", self.param_row_id,
        ))
        self.positions[pair]["id"] = cur.lastrowid
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

        pnl_absolute, pnl_pct = net_pnl(
            pos["entry_price"], exit_price, pos["position_size"],
            self.costs["fee_pct"], self.costs["slippage_pct"],
        )
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

    def _persist_stop(self, conn, pos: dict):
        """Save a ratcheted trailing stop so it survives restarts."""
        if pos.get("id") is None:
            return
        conn.execute("UPDATE trade_journal SET stop_loss=? WHERE id=?",
                     (pos["stop_loss"], pos["id"]))
        conn.commit()

    def _restore_positions(self):
        """Restore open positions from DB on startup."""
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, pair, entry_time, entry_price, stop_loss, take_profit,
                   position_size, features_json
            FROM trade_journal
            WHERE mode='paper' AND strategy=? AND exit_time IS NULL
        """, (self.strategy,))
        rows = cur.fetchall()
        for row in rows:
            self.positions[row["pair"]] = {
                "id": row["id"],
                # The persisted stop already holds any trailing ratchet, so
                # restarting the high-water mark at entry is safe.
                "highest_high": row["entry_price"],
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
                        choices=["darvas", "dbw", "candlestick", "sr_breakout", "ma_crossover",
                                 "rsi_vwap", "cvd", "macd", "mean_reversion", "bb_squeeze"])
    args = parser.parse_args()

    trader = PaperTrader(strategy=args.strategy)
    trader.start()
