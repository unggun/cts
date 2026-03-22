# Crypto Auto-Learning Trading System

A self-improving crypto trading system that learns from its own backtest results and refines its strategies over time.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   DATA LAYER                         │
│  CCXT (Tokocrypto) → OHLCV → SQLite                │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               ANALYSIS LAYER                         │
│  pandas-ta → Pattern Detection (Darvas/DBW)          │
│  Feature Extraction (40+ features per trade)         │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               BACKTEST LAYER                         │
│  Strategy Runner → Trade Logger → Performance Calc   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│             SIMULATION LAYER                         │
│  Monte Carlo (10k paths) → Risk Metrics              │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│             AUTO-LEARNING LAYER                      │
│  Claude API → Pattern Review → Parameter Updates     │
│  Winner/Loser Analysis → Filter Rule Generation      │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              EXECUTION LAYER                         │
│  Paper Trading → Live Trading (Tokocrypto)           │
│  Telegram Alerts → Trade Journal                     │
└─────────────────────────────────────────────────────┘
```

## Setup

```bash
# 1. Clone to your VPS
scp -r crypto-trading-system/ user@your-vps:/opt/

# 2. Install dependencies
cd /opt/crypto-trading-system
pip install -r requirements.txt

# 3. Configure
cp config.example.yaml config.yaml
# Edit config.yaml with your Tokocrypto API keys

# 4. Download historical data
python -m src.data.downloader --pairs BTC/IDR ETH/IDR --timeframe 1h --days 365

# 5. Run backtest
python -m src.backtest.runner --strategy darvas --pair BTC/IDR

# 6. Run Monte Carlo simulation
python -m src.simulation.monte_carlo --trades-from-backtest

# 7. Run auto-learning analysis
python -m src.learning.analyzer

# 8. Start paper trading
python -m src.execution.paper_trader
```

## Project Structure

```
crypto-trading-system/
├── config.example.yaml      # Configuration template
├── requirements.txt          # Python dependencies
├── src/
│   ├── data/
│   │   ├── downloader.py     # OHLCV data fetcher via CCXT
│   │   └── database.py       # SQLite schema & operations
│   ├── indicators/
│   │   ├── darvas.py         # Darvas box pattern detector
│   │   ├── dbw.py            # DBW pattern detector
│   │   └── features.py       # 40+ feature extraction
│   ├── backtest/
│   │   ├── engine.py         # Core backtest engine
│   │   └── runner.py         # CLI backtest runner
│   ├── simulation/
│   │   └── monte_carlo.py    # Monte Carlo simulator
│   ├── learning/
│   │   ├── analyzer.py       # Winner/loser pattern analysis
│   │   ├── claude_review.py  # Claude API integration
│   │   └── parameter_store.py # Track parameter evolution
│   └── execution/
│       └── paper_trader.py   # Paper trading engine
└── data/                     # SQLite DB & exports (gitignored)
```

## Auto-Learning Loop

The system improves itself through this cycle:

1. **Collect** — Download OHLCV data, run pattern detection
2. **Backtest** — Execute strategy against historical data
3. **Analyze** — Extract features, split winners vs losers
4. **Simulate** — Monte Carlo on backtest results
5. **Learn** — Claude API reviews findings, suggests parameter changes
6. **Update** — Apply new parameters, store evolution history
7. **Repeat** — Next cycle uses updated parameters

Schedule via cron on your VPS.
