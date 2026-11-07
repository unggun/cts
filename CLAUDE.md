# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Crypto Auto-Learning Trading System (CTS) — a self-improving crypto trading system targeting Tokocrypto exchange (IDR pairs). It backtests strategies, runs Monte Carlo simulations, then uses Claude API to analyze results and auto-tune strategy parameters.

## Commands

```bash
# Setup
pip install -r requirements.txt
cp config.example.yaml config.yaml
python3 -c "from src.data.database import init_db; init_db()"

# Download data
python3 -m src.data.downloader --pairs BTC/IDR ETH/IDR --timeframe 1h --days 365

# Run backtest
python3 -m src.backtest.runner --strategy darvas --pair BTC/IDR

# Monte Carlo simulation
python3 -m src.simulation.monte_carlo --strategy darvas

# Analyze winner/loser patterns
python3 -m src.learning.analyzer --strategy darvas

# Claude auto-review (dry run shows prompt without API call)
python3 -m src.learning.claude_review --strategy darvas --dry-run

# Full auto-learning cycle
python3 -m src.run_learning_cycle --strategy darvas --dry-run

# Paper trading
python3 -m src.execution.paper_trader --strategy darvas

# Live trading (requires sandbox=false and API keys)
python3 -m src.execution.live_trader --strategy darvas --preflight
python3 -m src.execution.live_trader --strategy darvas

# Query open positions (paper + live)
python3 -m src.execution.positions
python3 -m src.execution.positions --mode paper
python3 -m src.execution.positions --strategy rsi_vwap
python3 -m src.execution.positions --json

# Check live-trading readiness (per strategy: READY / WATCH / BLOCKED / INSUFFICIENT_DATA)
python3 -m src.execution.live_readiness
python3 -m src.execution.live_readiness --strategy dbw
python3 -m src.execution.live_readiness --json

# Telegram bot (listens for /positions, /equity, /help commands)
python3 -m src.execution.telegram_bot

# Run all strategies in one cycle (with combined Telegram summary)
python3 -m src.run_learning_cycle --all-strategies --dry-run

# Parameter version comparison
python3 -m src.learning.parameter_store --strategy darvas --compare

# Performance trend across versions
python3 -m src.learning.parameter_store --strategy darvas --trend
```

## Architecture

All modules run as `python -m src.<module>` from the project root. Config is loaded from `config.yaml` (falls back to `config.example.yaml`). Secrets can be overridden via env vars: `TOKOCRYPTO_API_KEY`, `TOKOCRYPTO_SECRET`, `CLAUDE_API_KEY`, `TELEGRAM_BOT_TOKEN`.

**Data flow**: CCXT download → SQLite (`data/trading.db`) → backtest engine → Monte Carlo sim → Claude API review → parameter update → repeat.

### Strategy Registry

`src/backtest/engine.py` defines `STRATEGY_REGISTRY` mapping strategy names to `(detect_fn, signal_col, levels_fn)` tuples. Available strategies: `darvas`, `dbw`, `candlestick`, `support_resistance`, `ema_crossover`. Each strategy has a detector in `src/indicators/` that adds signal columns to the DataFrame, and a levels function that returns `entry`, `stop_loss`, `target`, `risk_reward`.

### Parameter Versioning

Strategy parameters are versioned in `strategy_parameters` table. New versions are only saved by promotion (`src/learning/promotion.py`) after a proposal wins on held-out data. `load_latest_parameters()` retrieves the most recent version for a strategy; paper traders reload it automatically when a new version appears.

### Auto-Learning Loop

`src/run_learning_cycle.py` orchestrates the full cycle: download → backtest all pairs on the train window → analyze winners/losers → Monte Carlo → Claude review (proposal only) → holdout validation and promotion → Telegram notification. The `--dry-run` flag skips the Claude API call and shows the prompt instead; it still evaluates the analyzer's filters on the holdout but never promotes. A dry run still writes the train-window `backtest_trades` and the `simulation_results` row; it saves no parameter version, filter change or `learning_sessions` row.

### Propose → Verify → Promote

The newest `learning.promotion.holdout_days` (90) of data are held out from analysis and Claude. Claude's proposal (params + filter set, capped like live filters) and the current version are both backtested on that holdout after costs; the proposal is promoted only if it improves expectancy per trade by `min_expectancy_gain`, keeps at least `min_holdout_trades` and `min_trade_ratio` of the current trades, and doesn't worsen max drawdown by more than `max_drawdown_worsening`. Quality gates and the Monte Carlo verdict are reported but don't block promotion. Each verdict is stored in `learning_sessions.evaluation_json` (and `applied`), and the last 3 are shown to Claude in the next prompt. Design: `docs/superpowers/specs/2026-11-07-walk-forward-promotion-design.md`.

### Filter Rules

`src/learning/filter_rules.py` manages structured trade filters (stored in `filter_rules` table). The analyzer and Claude review both generate filter rules that gate trade entries — trades matching "skip" rules are rejected before opening. Proposed rule sets only become active when promoted (see Propose → Verify → Promote).

Only the strongest `learning.max_filter_rules` (default 3) skip rules are applied, one per feature (`select_filters`). If filters leave fewer than `min_trades_for_analysis` trades, the cycle re-runs the backtest unfiltered so the analyzer and Claude see the full signal population (`filters_starved_cycle`). The active filters stay in place as the incumbent until a promotion replaces them.

### Costs and Paper/Backtest Parity

Backtests and paper trades charge `costs.fee_pct` and `costs.slippage_pct` per side (`src/backtest/costs.py`). The paper trader enters only on closed candles (`src/data/candles.py`) and uses the same ATR trailing stop as the engine (`src/backtest/exits.py`), persisting ratcheted stops to `trade_journal`. The live trader still acts on the forming candle and does not trail fixed-target (OCO) positions.

### Confidence-Gated Updates

Claude returns a confidence level (low/medium/high). When confidence is "low", the proposal is not evaluated or promoted. Filter rules change only through promotion (or a manual `--reset-filters`).

### Database Schema

Seven tables in SQLite: `ohlcv` (candle data), `backtest_trades`, `strategy_parameters` (versioned), `learning_sessions` (Claude review history), `simulation_results`, `trade_journal` (paper/live trades), `filter_rules` (active trade filters). Schema is in `src/data/database.py:init_db()` and `src/learning/filter_rules.py:init_filter_rules_table()`.

### Paper & Live Trading

`src/execution/paper_trader.py` connects to Tokocrypto for real-time prices but executes trades in the local database. `src/execution/live_trader.py` places real orders. Both traders:
- Persist open positions to `trade_journal` (with `exit_time IS NULL` for open trades)
- Restore positions from DB on startup (survives restarts)
- Send Telegram notifications on every buy/sell
- Use learned parameters and filter rules from the auto-learning loop

The learning system only uses `backtest_trades` for analysis, not paper/live trades. Paper/live trading validates the strategy in real-time conditions.

### Exchange-Side Protection (OCO / Stop-Loss-Limit)

`src/execution/protection.py` places real protective sell orders on Tokocrypto
after a buy fill: an **OCO** for fixed-target strategies, a **STOP_LOSS_LIMIT**
for Darvas ATH-breakout positions (which trail). Polling-based exits in
`live_trader._check_exit` are now a fallback path used only when placement
fails (`protection_mode="polling"`) or for legacy positions.

Protection state is observable via the `protection_mode` column on
`trade_journal` (`oco` | `stop_only` | `polling` | `simulated`) and via
`/positions` (CLI) and the Telegram bot.

Master switch: `risk.protection.enabled` in `config.yaml` — set to `false`
to revert to today's polling-only behavior without code changes.

On bot restart, `_restore_positions` reconciles each restored position with
the exchange (still active / closed-while-down / naked) and re-places
protection if it's missing.

### Position Queries

`src/execution/positions.py` provides a CLI to query open positions from `trade_journal`. Open positions are identified by `exit_time IS NULL`.

### Live-Readiness Check

`src/execution/live_readiness.py` evaluates per-strategy readiness to flip from paper to live trading. Read-only, checks four gates: Monte Carlo verdict/ruin probability, Claude confidence, and current-param paper-trade stats (split around the latest `strategy_parameters.created_at` cutover; trades older than 30 days excluded). Status tiers: `INSUFFICIENT_DATA` (MC sample < 50) → `BLOCKED` → `WATCH` → `READY`. Flipping to live remains manual.

The weekly learning-cycle Telegram summary (`_send_multi_strategy_telegram` in `src/run_learning_cycle.py`) includes a per-strategy status icon and a readiness footer. The Telegram bot's `/readiness` and `/readiness <strategy>` commands expose the same data on demand.

### Telegram Bot

`src/execution/telegram_bot.py` is a long-polling Telegram bot that listens for commands from the configured `chat_id`. Supported commands: `/positions`, `/positions paper`, `/positions live`, `/positions <strategy>`, `/equity`, `/readiness`, `/readiness <strategy>`, `/help`. Run as a systemd service on the VPS.

### Feature Extraction

`src/indicators/features.py` extracts 40+ features per trade signal (volume ratios, ATR, RSI, moving averages, etc.) stored as JSON in `backtest_trades.features_json`. These features drive the winner/loser analysis in `src/learning/analyzer.py`.

## VPS Deployment

The system runs on a VPS with systemd services:

```bash
# Paper traders (one per strategy)
sudo systemctl start cts-paper-trader@darvas
sudo systemctl start cts-paper-trader@rsi_vwap

# Weekly auto-learning cycle
sudo systemctl start cts-learning.timer

# Telegram bot (for /positions, /equity queries)
sudo systemctl start cts-telegram-bot

# Check status
sudo systemctl status cts-paper-trader@rsi_vwap
sudo systemctl status cts-telegram-bot
journalctl -u cts-learning.service --since today --no-pager
```

### Systemd service files

- `/etc/systemd/system/cts-paper-trader@.service` — template for paper traders
- `/etc/systemd/system/cts-learning.service` — auto-learning cycle
- `/etc/systemd/system/cts-learning.timer` — weekly trigger (Sundays at midnight)
- `/etc/systemd/system/cts-telegram-bot.service` — Telegram bot
