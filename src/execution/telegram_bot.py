"""Telegram bot for querying trading status via chat commands.

Runs as a long-polling bot that listens for commands:
  /positions          — show all open positions
  /positions paper    — paper positions only
  /positions live     — live positions only
  /equity             — show equity summary per strategy
  /help               — show available commands
"""
import time
import signal
import sys
import json
from datetime import datetime

import requests as req

from src.config import load_config
from src.data.database import get_connection, init_db
from src.data.downloader import create_exchange


class TelegramBot:
    """Simple Telegram bot using long polling."""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.running = False

        tg = self.config.get("notifications", {}).get("telegram", {})
        self.bot_token = tg.get("bot_token", "")
        self.chat_id = str(tg.get("chat_id", ""))
        self.api_base = f"https://api.telegram.org/bot{self.bot_token}"

        if not self.bot_token or "YOUR" in self.bot_token:
            print("ERROR: Telegram bot_token not configured in config.yaml")
            sys.exit(1)

        self.offset = 0  # Track last processed update

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        print("\nShutting down Telegram bot...")
        self.running = False

    def start(self):
        """Start the bot polling loop."""
        # Verify bot token
        try:
            resp = req.get(f"{self.api_base}/getMe", timeout=10)
            bot_info = resp.json()
            if not bot_info.get("ok"):
                print(f"ERROR: Invalid bot token: {bot_info}")
                sys.exit(1)
            bot_name = bot_info["result"]["username"]
            print(f"Telegram bot @{bot_name} started. Listening for commands...")
        except Exception as e:
            print(f"ERROR: Cannot connect to Telegram API: {e}")
            sys.exit(1)

        self.running = True
        while self.running:
            try:
                self._poll()
            except Exception as e:
                print(f"Polling error: {e}")
                time.sleep(5)

        print("Telegram bot stopped.")

    def _poll(self):
        """Long-poll for new messages."""
        try:
            resp = req.get(
                f"{self.api_base}/getUpdates",
                params={"offset": self.offset, "timeout": 30},
                timeout=35,
            )
            data = resp.json()
        except req.exceptions.Timeout:
            return
        except Exception:
            time.sleep(2)
            return

        if not data.get("ok"):
            return

        for update in data.get("result", []):
            self.offset = update["update_id"] + 1
            msg = update.get("message", {})
            chat_id = str(msg.get("chat", {}).get("id", ""))
            text = msg.get("text", "").strip()

            # Only respond to the configured chat
            if chat_id != self.chat_id:
                continue

            if not text.startswith("/"):
                continue

            self._handle_command(chat_id, text)

    def _handle_command(self, chat_id: str, text: str):
        """Route commands to handlers."""
        parts = text.split()
        command = parts[0].lower().split("@")[0]  # Strip @botname suffix
        args = parts[1:]

        handlers = {
            "/positions": self._cmd_positions,
            "/pos": self._cmd_positions,
            "/equity": self._cmd_equity,
            "/help": self._cmd_help,
            "/start": self._cmd_help,
        }

        handler = handlers.get(command)
        if handler:
            try:
                response = handler(args)
            except Exception as e:
                print(f"Command error ({command}): {e}")
                response = f"Error processing {command}: {e}"
        else:
            response = f"Unknown command: {command}\nType /help for available commands."

        self._send(chat_id, response)

    def _send(self, chat_id: str, text: str):
        """Send a message. Falls back to plain text if Markdown fails."""
        try:
            resp = req.post(
                f"{self.api_base}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
            if not resp.json().get("ok"):
                # Retry without Markdown
                req.post(
                    f"{self.api_base}/sendMessage",
                    json={"chat_id": chat_id, "text": text},
                    timeout=10,
                )
        except Exception as e:
            print(f"Send error: {e}")

    def _cmd_help(self, args: list) -> str:
        """Show available commands."""
        return (
            "🤖 *CTS Trading Bot*\n\n"
            "Available commands:\n"
            "`/positions` — all open positions\n"
            "`/positions paper` — paper positions only\n"
            "`/positions live` — live positions only\n"
            "`/positions <strategy>` — filter by strategy\n"
            "`/equity` — equity summary per strategy\n"
            "`/help` — this message"
        )

    def _cmd_positions(self, args: list) -> str:
        """Show open positions."""
        init_db()
        conn = get_connection()
        cur = conn.cursor()

        query = "SELECT * FROM trade_journal WHERE exit_time IS NULL"
        params = []

        # Parse filter args
        mode_filter = None
        strategy_filter = None
        for arg in args:
            a = arg.lower()
            if a in ("paper", "live"):
                mode_filter = a
            else:
                strategy_filter = a

        if mode_filter:
            query += " AND mode=?"
            params.append(mode_filter)
        if strategy_filter:
            query += " AND strategy=?"
            params.append(strategy_filter)

        query += " ORDER BY mode, strategy, pair"
        cur.execute(query, params)
        rows = cur.fetchall()
        conn.close()

        if not rows:
            filter_desc = ""
            if mode_filter:
                filter_desc += f" ({mode_filter})"
            if strategy_filter:
                filter_desc += f" [{strategy_filter}]"
            return f"No open positions{filter_desc}."

        lines = []
        current_mode = None
        current_strategy = None

        for row in rows:
            r = dict(row)

            if r["mode"] != current_mode:
                current_mode = r["mode"]
                icon = "🔴" if current_mode == "live" else "📄"
                lines.append(f"\n{icon} *{current_mode.upper()} TRADING*")

            if r["strategy"] != current_strategy:
                current_strategy = r["strategy"]
                lines.append(f"\n📊 _{current_strategy.upper()}_")

            target = f"{r['take_profit']:,.0f}" if r.get("take_profit") else "ATH"
            lines.append(
                f"  *{r['pair']}*\n"
                f"    Entry: {r['entry_price']:,.0f}\n"
                f"    SL: {r['stop_loss']:,.0f} | TP: {target}\n"
                f"    Size: {r['position_size']:.6f}"
            )

        lines.append(f"\n_Total: {len(rows)} position(s)_")
        return "\n".join(lines)

    def _cmd_equity(self, args: list) -> str:
        """Show equity summary by reading running paper trader state."""
        init_db()
        conn = get_connection()
        cur = conn.cursor()

        # Get open positions grouped by strategy
        cur.execute("""
            SELECT mode, strategy, COUNT(*) as count,
                   GROUP_CONCAT(pair, ', ') as pairs
            FROM trade_journal
            WHERE exit_time IS NULL
            GROUP BY mode, strategy
            ORDER BY mode, strategy
        """)
        open_rows = cur.fetchall()

        # Get recent closed trades (last 7 days)
        cur.execute("""
            SELECT mode, strategy, COUNT(*) as count,
                   SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(pnl_absolute) as total_pnl
            FROM trade_journal
            WHERE exit_time IS NOT NULL
                AND exit_time >= datetime('now', '-7 days')
            GROUP BY mode, strategy
            ORDER BY mode, strategy
        """)
        closed_rows = cur.fetchall()
        conn.close()

        lines = ["📊 *Equity Summary*\n"]

        if open_rows:
            lines.append("*Open Positions:*")
            for row in open_rows:
                r = dict(row)
                icon = "🔴" if r["mode"] == "live" else "📄"
                lines.append(
                    f"  {icon} {r['strategy'].upper()}: "
                    f"{r['count']} pos ({r['pairs']})"
                )
        else:
            lines.append("_No open positions_")

        if closed_rows:
            lines.append("\n*Last 7 Days (closed):*")
            for row in closed_rows:
                r = dict(row)
                icon = "🔴" if r["mode"] == "live" else "📄"
                wr = r["wins"] / r["count"] * 100 if r["count"] > 0 else 0
                pnl = r["total_pnl"] or 0
                emoji = "📈" if pnl > 0 else "📉"
                lines.append(
                    f"  {icon} {r['strategy'].upper()}: "
                    f"{r['count']} trades | WR: {wr:.0f}% | "
                    f"{emoji} {pnl:+,.0f} IDR"
                )
        else:
            lines.append("\n_No closed trades in last 7 days_")

        return "\n".join(lines)


if __name__ == "__main__":
    bot = TelegramBot()
    bot.start()
