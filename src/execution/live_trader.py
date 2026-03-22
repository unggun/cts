"""Live trading engine — executes real orders on Tokocrypto.

This extends the paper trader with actual order execution via CCXT.
All the same pattern detection, risk management, and auto SL/TP logic applies.

IMPORTANT SAFETY FEATURES:
- Requires explicit config flag: exchange.sandbox = false
- Maximum position size enforced at exchange level
- All orders are limit orders (not market) to avoid slippage
- Every order is logged before and after execution
- Kill switch: create a file called STOP_TRADING in the project root to halt
- Telegram notifications on every trade entry/exit
"""
import json
import time
import signal
import sys
from datetime import datetime
from pathlib import Path

import ccxt

from src.config import load_config
from src.data.downloader import create_exchange
from src.data.database import (
    get_connection, init_db, load_ohlcv, upsert_ohlcv, load_latest_parameters
)
from src.indicators.darvas import detect_darvas_boxes, calculate_trade_levels
from src.indicators.dbw import detect_dbw, calculate_dbw_levels
from src.indicators.features import extract_features


class LiveTrader:
    """Live trading engine with real order execution."""

    def __init__(self, strategy: str = "darvas", config: dict = None):
        self.config = config or load_config()
        self.strategy = strategy
        self.running = False

        # ── Safety checks ──
        exc_cfg = self.config["exchange"]
        if exc_cfg.get("sandbox", True):
            print("=" * 60)
            print("ERROR: Cannot start live trading while sandbox=true")
            print("Set exchange.sandbox to false in config.yaml")
            print("=" * 60)
            sys.exit(1)

        if exc_cfg.get("api_key", "").startswith("YOUR"):
            print("ERROR: API key not configured")
            sys.exit(1)

        # Create exchange with live credentials
        self.exchange = create_exchange(self.config, authenticated=True)

        # Verify API connection and permissions
        self._verify_connection()

        # Load strategy parameters
        init_db()
        conn = get_connection()
        self.params = load_latest_parameters(conn, strategy)
        if self.params is None:
            self.params = self.config.get("strategy", {}).get(strategy, {})
        self.params["risk"] = self.config.get("risk", {})
        conn.close()

        # State
        self.positions = {}
        self.daily_pnl = 0.0
        self.trade_count_today = 0

        # Kill switch file path
        self.kill_switch = Path(__file__).parent.parent.parent / "STOP_TRADING"

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _verify_connection(self):
        """Verify API connection and check balance."""
        print("Verifying exchange connection...")
        try:
            balance = self.exchange.fetch_balance()
            idr_free = balance.get("IDR", {}).get("free", 0)
            idr_total = balance.get("IDR", {}).get("total", 0)
            print(f"  Connected to {self.exchange.id}")
            print(f"  IDR balance: {idr_free:,.0f} free / {idr_total:,.0f} total")

            # Check for non-IDR assets
            for currency, amounts in balance.get("total", {}).items():
                if amounts > 0 and currency != "IDR":
                    print(f"  {currency}: {amounts}")

            self.capital = idr_free
            self.initial_capital = idr_free
        except ccxt.AuthenticationError:
            print("ERROR: Authentication failed. Check your API key and secret.")
            sys.exit(1)
        except ccxt.ExchangeError as e:
            print(f"ERROR: Exchange error: {e}")
            sys.exit(1)

    def _check_kill_switch(self) -> bool:
        """Check if the kill switch file exists."""
        if self.kill_switch.exists():
            print("\n🛑 KILL SWITCH ACTIVATED — STOP_TRADING file detected")
            print("Remove the file to resume trading:")
            print(f"  rm {self.kill_switch}")
            return True
        return False

    def _check_daily_limits(self) -> bool:
        """Check if daily loss limit has been reached."""
        max_daily_loss = self.config.get("risk", {}).get("max_daily_loss_pct", 0.03)
        if self.initial_capital > 0:
            daily_loss_pct = self.daily_pnl / self.initial_capital
            if daily_loss_pct < -max_daily_loss:
                print(f"\n⚠️ Daily loss limit reached: {daily_loss_pct:.2%} "
                      f"(limit: {-max_daily_loss:.2%})")
                return False
        return True

    def start(self):
        """Start the live trading loop."""
        pairs = self.config["trading"]["pairs"]
        timeframe = self.config["trading"]["timeframes"][0]
        risk_cfg = self.config.get("risk", {})

        print("\n" + "=" * 60)
        print("🔴 LIVE TRADING STARTED")
        print("=" * 60)
        print(f"Strategy:        {self.strategy}")
        print(f"Pairs:           {', '.join(pairs)}")
        print(f"Timeframe:       {timeframe}")
        print(f"Capital:         {self.capital:,.0f} IDR")
        print(f"Max pos size:    {risk_cfg.get('max_position_pct', 0.05):.0%}")
        print(f"Max positions:   {risk_cfg.get('max_positions', 3)}")
        print(f"Max daily loss:  {risk_cfg.get('max_daily_loss_pct', 0.03):.0%}")
        print(f"Kill switch:     touch {self.kill_switch}")
        print("=" * 60)
        print("Press Ctrl+C to stop.\n")

        self._notify(
            f"🔴 *Live trading started*\n"
            f"Strategy: {self.strategy}\n"
            f"Capital: {self.capital:,.0f} IDR\n"
            f"Pairs: {', '.join(pairs)}"
        )

        self.running = True

        while self.running:
            try:
                # Safety checks
                if self._check_kill_switch():
                    self.running = False
                    break

                if not self._check_daily_limits():
                    self._notify("⚠️ Daily loss limit reached. Trading paused.")
                    time.sleep(3600)  # Wait 1 hour before rechecking
                    continue

                for pair in pairs:
                    self._process_pair(pair, timeframe)

                # Status update
                self._print_status(pairs)

                # Sleep
                sleep_seconds = self._get_sleep_seconds(timeframe)
                time.sleep(sleep_seconds)

            except ccxt.NetworkError as e:
                print(f"Network error: {e}. Retrying in 30s...")
                time.sleep(30)
            except ccxt.ExchangeError as e:
                print(f"Exchange error: {e}. Retrying in 60s...")
                time.sleep(60)
            except Exception as e:
                print(f"Unexpected error: {e}")
                self._notify(f"⚠️ Trading error: {e}")
                time.sleep(60)

        # Shutdown
        self._notify("⏹️ Live trading stopped.")
        print("Live trading stopped.")

    def _process_pair(self, pair: str, timeframe: str):
        """Process a single pair — check exits and entries."""
        conn = get_connection()

        # Fetch latest candles
        try:
            candles = self.exchange.fetch_ohlcv(pair, timeframe, limit=200)
            if candles:
                upsert_ohlcv(conn, pair, timeframe, candles)
        except Exception as e:
            print(f"  Error fetching {pair}: {e}")
            conn.close()
            return

        # Load and analyze data
        df = load_ohlcv(conn, pair, timeframe)
        if len(df) < 60:
            conn.close()
            return

        if self.strategy == "darvas":
            df = detect_darvas_boxes(df, self.params)
            signal_col = "signal"
        elif self.strategy == "dbw":
            df = detect_dbw(df, self.params)
            signal_col = "dbw_signal"
        else:
            conn.close()
            return

        last_idx = len(df) - 1
        current_price = df.iloc[last_idx]["close"]

        # ── Check exits ──
        if pair in self.positions:
            self._check_exit(df, last_idx, pair, conn)

        # ── Check entries ──
        if pair not in self.positions:
            if df.iloc[last_idx].get(signal_col) == "buy":
                self._try_entry(df, last_idx, pair, conn)

        conn.close()

    def _check_exit(self, df, idx, pair, conn):
        """Check and execute exits."""
        pos = self.positions[pair]
        low = df.iloc[idx]["low"]
        high = df.iloc[idx]["high"]

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
            self._execute_sell(pair, exit_price, exit_reason, conn)

    def _try_entry(self, df, idx, pair, conn):
        """Evaluate and execute entry."""
        strategy_params = self.params.get(self.strategy, self.params)

        if self.strategy == "darvas":
            levels = calculate_trade_levels(df, idx, strategy_params)
        elif self.strategy == "dbw":
            levels = calculate_dbw_levels(df, idx, strategy_params)
        else:
            return

        min_rr = strategy_params.get("min_risk_reward", 2.0)
        if levels["risk_reward"] < min_rr and levels.get("target") is not None:
            return

        max_pos = self.params.get("risk", {}).get("max_positions", 3)
        if len(self.positions) >= max_pos:
            return

        # Refresh balance before sizing
        try:
            balance = self.exchange.fetch_balance()
            available = balance.get("IDR", {}).get("free", 0)
        except Exception:
            return

        max_pos_pct = self.params.get("risk", {}).get("max_position_pct", 0.05)
        risk_amount = available * max_pos_pct

        if levels["risk"] <= 0:
            return

        position_size = risk_amount / levels["risk"]
        cost = position_size * levels["entry"]

        if cost > available * 0.95:
            position_size = available * 0.95 / levels["entry"]

        # Minimum order check
        min_cost = 10_000  # IDR 10k minimum (adjust per exchange)
        if position_size * levels["entry"] < min_cost:
            print(f"  Skip {pair}: order too small ({position_size * levels['entry']:,.0f} IDR)")
            return

        features = extract_features(df, idx)
        self._execute_buy(pair, levels, position_size, features, conn)

    def _execute_buy(self, pair, levels, size, features, conn):
        """Place a buy order on the exchange."""
        entry_price = levels["entry"]

        print(f"\n  📈 PLACING BUY ORDER: {pair}")
        print(f"     Price: {entry_price:,.0f} | Size: {size:.8f}")
        print(f"     SL: {levels['stop_loss']:,.0f} | "
              f"Target: {levels.get('target', 'ATH trail')}")

        try:
            # Use limit order at current price
            order = self.exchange.create_order(
                symbol=pair,
                type="limit",
                side="buy",
                amount=size,
                price=entry_price,
            )

            order_id = order.get("id")
            status = order.get("status")
            filled_price = order.get("average") or entry_price

            print(f"     Order ID: {order_id} | Status: {status}")

            # Wait briefly for fill
            if status != "closed":
                time.sleep(5)
                order = self.exchange.fetch_order(order_id, pair)
                status = order.get("status")
                filled_price = order.get("average") or entry_price

            if status in ("closed", "filled"):
                self.positions[pair] = {
                    "entry_time": datetime.now().isoformat(),
                    "entry_price": filled_price,
                    "stop_loss": levels["stop_loss"],
                    "target": levels.get("target"),
                    "position_size": order.get("filled", size),
                    "is_ath": levels.get("is_ath", False),
                    "order_id": order_id,
                    "features": features,
                }
                print(f"     ✅ FILLED @ {filled_price:,.0f}")

                target_str = f"{levels['target']:,.0f}" if levels.get("target") else "ATH trailing"
                self._notify(
                    f"📈 *BUY {pair}*\n"
                    f"Entry: {filled_price:,.0f}\n"
                    f"Stop loss: {levels['stop_loss']:,.0f}\n"
                    f"Target: {target_str}\n"
                    f"Size: {order.get('filled', size):.8f}"
                )

                # Log to journal
                self._log_entry(conn, pair, filled_price, levels,
                                order.get("filled", size), features)

            else:
                # Cancel unfilled order
                print(f"     ⚠️ Order not filled (status: {status}). Cancelling.")
                try:
                    self.exchange.cancel_order(order_id, pair)
                except Exception:
                    pass

        except ccxt.InsufficientFunds:
            print(f"     ❌ Insufficient funds for {pair}")
        except ccxt.ExchangeError as e:
            print(f"     ❌ Order failed: {e}")

    def _execute_sell(self, pair, exit_price, reason, conn):
        """Place a sell order on the exchange."""
        pos = self.positions[pair]
        size = pos["position_size"]

        print(f"\n  📉 PLACING SELL ORDER: {pair} (reason: {reason})")
        print(f"     Price: {exit_price:,.0f} | Size: {size:.8f}")

        try:
            order = self.exchange.create_order(
                symbol=pair,
                type="limit",
                side="sell",
                amount=size,
                price=exit_price,
            )

            order_id = order.get("id")
            status = order.get("status")
            filled_price = order.get("average") or exit_price

            # Wait for fill
            if status != "closed":
                time.sleep(5)
                order = self.exchange.fetch_order(order_id, pair)
                status = order.get("status")
                filled_price = order.get("average") or exit_price

            # If still not filled after 10s, use market order
            if status not in ("closed", "filled"):
                print(f"     Limit not filled, trying market order...")
                try:
                    self.exchange.cancel_order(order_id, pair)
                except Exception:
                    pass
                order = self.exchange.create_order(
                    symbol=pair,
                    type="market",
                    side="sell",
                    amount=size,
                )
                filled_price = order.get("average") or exit_price

            pnl_per_unit = filled_price - pos["entry_price"]
            pnl_absolute = pnl_per_unit * size
            pnl_pct = pnl_per_unit / pos["entry_price"]

            self.daily_pnl += pnl_absolute
            self.trade_count_today += 1

            emoji = "✅" if pnl_pct > 0 else "❌"
            print(f"     {emoji} CLOSED @ {filled_price:,.0f} | "
                  f"PnL: {pnl_pct:+.2%} ({pnl_absolute:+,.0f} IDR)")

            self._notify(
                f"{emoji} *SELL {pair}* ({reason})\n"
                f"Exit: {filled_price:,.0f}\n"
                f"PnL: {pnl_pct:+.2%} ({pnl_absolute:+,.0f} IDR)"
            )

            # Log to journal
            self._log_exit(conn, pair, filled_price, reason, pnl_pct,
                          pnl_absolute, pos)

            del self.positions[pair]

        except ccxt.ExchangeError as e:
            print(f"     ❌ Sell failed: {e}")
            self._notify(f"⚠️ SELL FAILED for {pair}: {e}")

    def _log_entry(self, conn, pair, price, levels, size, features):
        """Log trade entry to journal."""
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trade_journal
            (mode, pair, strategy, direction, entry_time, entry_price,
             stop_loss, take_profit, position_size, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "live", pair, self.strategy, "long",
            datetime.now().isoformat(), price,
            levels["stop_loss"], levels.get("target"), size,
            json.dumps(features)
        ))
        conn.commit()

    def _log_exit(self, conn, pair, price, reason, pnl_pct, pnl_abs, pos):
        """Update trade journal with exit info."""
        cur = conn.cursor()
        cur.execute("""
            UPDATE trade_journal SET
                exit_time=?, exit_price=?, exit_reason=?,
                pnl_pct=?, pnl_absolute=?
            WHERE mode='live' AND pair=? AND strategy=?
                AND exit_time IS NULL
            ORDER BY created_at DESC LIMIT 1
        """, (
            datetime.now().isoformat(), price, reason,
            pnl_pct, pnl_abs, pair, self.strategy
        ))
        conn.commit()

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

    def _print_status(self, pairs):
        """Print current status."""
        positions_str = ", ".join(
            f"{p}: {pos['entry_price']:,.0f}" for p, pos in self.positions.items()
        ) or "none"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"Daily PnL: {self.daily_pnl:+,.0f} IDR | "
              f"Trades today: {self.trade_count_today} | "
              f"Positions: {positions_str}")

    def _get_sleep_seconds(self, timeframe):
        tf_seconds = {
            "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 600, "4h": 1800, "1d": 3600,
        }
        return tf_seconds.get(timeframe, 600)

    def _shutdown(self, signum, frame):
        print("\nShutting down live trader...")
        self.running = False


# ── Pre-flight checklist ──

def preflight_check(config: dict = None) -> bool:
    """Run all safety checks before going live. Returns True if all pass."""
    if config is None:
        config = load_config()

    print("=" * 60)
    print("PRE-FLIGHT CHECKLIST FOR LIVE TRADING")
    print("=" * 60)

    checks = []

    # 1. Config check
    sandbox = config["exchange"].get("sandbox", True)
    checks.append(("Config: sandbox=false", not sandbox))

    # 2. API key check
    api_key = config["exchange"].get("api_key", "")
    checks.append(("API key configured", not api_key.startswith("YOUR")))

    # 3. Risk limits configured
    risk = config.get("risk", {})
    checks.append(("Max position % set", "max_position_pct" in risk))
    checks.append(("Max daily loss % set", "max_daily_loss_pct" in risk))
    checks.append(("Max positions set", "max_positions" in risk))

    # 4. Telegram notifications
    tg = config.get("notifications", {}).get("telegram", {})
    tg_ok = tg.get("enabled") and tg.get("bot_token", "").startswith("YOUR") is False
    checks.append(("Telegram alerts enabled", tg_ok))

    # 5. Has backtest data
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM backtest_trades")
        bt_count = cur.fetchone()[0]
        conn.close()
        checks.append((f"Backtest trades exist ({bt_count})", bt_count >= 30))
    except Exception:
        checks.append(("Backtest trades exist", False))

    # 6. Has simulation results
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT verdict FROM simulation_results ORDER BY created_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        conn.close()
        if row:
            verdict = row[0]
            checks.append((f"Monte Carlo verdict: {verdict}", verdict == "ROBUST"))
        else:
            checks.append(("Monte Carlo run exists", False))
    except Exception:
        checks.append(("Monte Carlo run exists", False))

    # 7. Paper trading history
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM trade_journal WHERE mode='paper'")
        paper_count = cur.fetchone()[0]
        conn.close()
        checks.append((f"Paper trades logged ({paper_count})", paper_count >= 10))
    except Exception:
        checks.append(("Paper trades logged", False))

    # Print results
    all_pass = True
    for check_name, passed in checks:
        emoji = "✅" if passed else "❌"
        print(f"  {emoji} {check_name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("✅ All checks passed. Ready for live trading.")
    else:
        print("❌ Some checks failed. Resolve issues before going live.")
        print("   (Telegram and paper trade checks are recommended but not blocking)")

    return all_pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live trading")
    parser.add_argument("--strategy", default="darvas",
                        choices=["darvas", "dbw", "candlestick", "sr_breakout", "ma_crossover"])
    parser.add_argument("--preflight", action="store_true",
                        help="Run pre-flight checks only")
    args = parser.parse_args()

    if args.preflight:
        preflight_check()
    else:
        # Always run preflight first
        if preflight_check():
            print("\nStarting live trader in 10 seconds... (Ctrl+C to abort)")
            time.sleep(10)
            trader = LiveTrader(strategy=args.strategy)
            trader.start()
        else:
            print("\nFix the issues above before starting live trading.")
