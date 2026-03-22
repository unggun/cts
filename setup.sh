#!/bin/bash
# Quick start script for the Crypto Auto-Learning Trading System
# Run this on your VPS after cloning the project.

set -e

echo "============================================"
echo "Crypto Auto-Learning Trading System Setup"
echo "============================================"
echo ""

# Check Python version
python3 --version || { echo "Python 3 is required. Install with: apt install python3"; exit 1; }

# Install dependencies
echo "[1/4] Installing Python dependencies..."
pip3 install -r requirements.txt --break-system-packages 2>/dev/null || pip3 install -r requirements.txt

# Create config from template
if [ ! -f config.yaml ]; then
    echo "[2/4] Creating config.yaml from template..."
    cp config.example.yaml config.yaml
    echo "  ⚠️  Edit config.yaml with your API keys before proceeding!"
    echo "  Required:"
    echo "    - Tokocrypto API key & secret"
    echo "    - Claude API key (for auto-learning)"
    echo "    - Telegram bot token (optional, for notifications)"
else
    echo "[2/4] config.yaml already exists, skipping."
fi

# Initialize database
echo "[3/4] Initializing database..."
python3 -c "from src.data.database import init_db; init_db()"

# Create data directory
mkdir -p data

echo "[4/4] Setup complete!"
echo ""
echo "============================================"
echo "Next steps:"
echo "============================================"
echo ""
echo "1. Edit config.yaml with your Tokocrypto API keys"
echo ""
echo "2. Download historical data:"
echo "   python3 -m src.data.downloader --pairs BTC/IDR ETH/IDR --timeframe 1h --days 365"
echo ""
echo "3. Run your first backtest:"
echo "   python3 -m src.backtest.runner --strategy darvas --pair BTC/IDR"
echo ""
echo "4. Run Monte Carlo simulation:"
echo "   python3 -m src.simulation.monte_carlo --strategy darvas"
echo ""
echo "5. Analyze winner/loser patterns:"
echo "   python3 -m src.learning.analyzer --strategy darvas"
echo ""
echo "6. Run Claude auto-review (dry run first):"
echo "   python3 -m src.learning.claude_review --strategy darvas --dry-run"
echo ""
echo "7. Run the full auto-learning cycle:"
echo "   python3 -m src.run_learning_cycle --strategy darvas --dry-run"
echo ""
echo "8. Start paper trading:"
echo "   python3 -m src.execution.paper_trader --strategy darvas"
echo ""
echo "============================================"
echo "Cron setup for auto-learning (add to crontab):"
echo "============================================"
echo ""
echo "# Weekly auto-learning cycle (Sunday midnight)"
echo "0 0 * * 0 cd $(pwd) && python3 -m src.run_learning_cycle --strategy darvas >> data/learning.log 2>&1"
echo ""
echo "# Daily data download (every 6 hours)"
echo "0 */6 * * * cd $(pwd) && python3 -m src.data.downloader --timeframe 1h --days 7 >> data/download.log 2>&1"
