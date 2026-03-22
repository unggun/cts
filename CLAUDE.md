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
```

## Architecture

All modules run as `python -m src.<module>` from the project root. Config is loaded from `config.yaml` (falls back to `config.example.yaml`). Secrets can be overridden via env vars: `TOKOCRYPTO_API_KEY`, `TOKOCRYPTO_SECRET`, `CLAUDE_API_KEY`, `TELEGRAM_BOT_TOKEN`.

**Data flow**: CCXT download → SQLite (`data/trading.db`) → backtest engine → Monte Carlo sim → Claude API review → parameter update → repeat.

### Strategy Registry

`src/backtest/engine.py` defines `STRATEGY_REGISTRY` mapping strategy names to `(detect_fn, signal_col, levels_fn)` tuples. Available strategies: `darvas`, `dbw`, `candlestick`, `support_resistance`, `ema_crossover`. Each strategy has a detector in `src/indicators/` that adds signal columns to the DataFrame, and a levels function that returns `entry`, `stop_loss`, `target`, `risk_reward`.

### Parameter Versioning

Strategy parameters are versioned in `strategy_parameters` table. The auto-learning loop (`src/learning/claude_review.py`) saves Claude-suggested parameters as new versions via `save_parameters()`. `load_latest_parameters()` retrieves the most recent version for a strategy.

### Auto-Learning Loop

`src/run_learning_cycle.py` orchestrates the full cycle: download → backtest all pairs → analyze winners/losers → Monte Carlo → Claude review → save updated parameters → Telegram notification. The `--dry-run` flag skips the Claude API call and shows the prompt instead.

### Database Schema

Six tables in SQLite: `ohlcv` (candle data), `backtest_trades`, `strategy_parameters` (versioned), `learning_sessions` (Claude review history), `simulation_results`, `trade_journal` (paper/live trades). Schema is in `src/data/database.py:init_db()`.

### Feature Extraction

`src/indicators/features.py` extracts 40+ features per trade signal (volume ratios, ATR, RSI, moving averages, etc.) stored as JSON in `backtest_trades.features_json`. These features drive the winner/loser analysis in `src/learning/analyzer.py`.
