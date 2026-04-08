"""Core backtest engine that runs strategies against historical data."""
import uuid
import json
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

from src.indicators.darvas import detect_darvas_boxes, calculate_trade_levels
from src.indicators.dbw import detect_dbw, calculate_dbw_levels
from src.indicators.candlestick import detect_candlestick_patterns, calculate_candle_levels
from src.indicators.support_resistance import detect_sr_breakouts, calculate_sr_levels
from src.indicators.ema_crossover import detect_ema_crossovers, calculate_ema_levels
from src.indicators.rsi_vwap import detect_rsi_vwap, calculate_rsi_vwap_levels
from src.indicators.cvd import detect_cvd_divergence, calculate_cvd_levels
from src.indicators.macd import detect_macd, calculate_macd_levels
from src.indicators.features import extract_features
from src.data.database import get_connection, save_backtest_trades, load_ohlcv
from src.learning.filter_rules import load_active_filters, apply_filters

# Strategy registry — maps strategy name to (detect_fn, signal_col, levels_fn)
STRATEGY_REGISTRY = {
    "darvas": (detect_darvas_boxes, "signal", calculate_trade_levels),
    "dbw": (detect_dbw, "dbw_signal", calculate_dbw_levels),
    "candlestick": (detect_candlestick_patterns, "candle_signal", calculate_candle_levels),
    "support_resistance": (detect_sr_breakouts, "sr_signal", calculate_sr_levels),
    "sr_breakout": (detect_sr_breakouts, "sr_signal", calculate_sr_levels),
    "ema_crossover": (detect_ema_crossovers, "ema_signal", calculate_ema_levels),
    "ma_crossover": (detect_ema_crossovers, "ema_signal", calculate_ema_levels),
    "rsi_vwap": (detect_rsi_vwap, "rsi_vwap_signal", calculate_rsi_vwap_levels),
    "cvd": (detect_cvd_divergence, "cvd_signal", calculate_cvd_levels),
    "macd": (detect_macd, "macd_signal", calculate_macd_levels),
}


class BacktestEngine:
    """Run a trading strategy against historical OHLCV data."""

    def __init__(self, strategy: str, params: dict, initial_capital: float = 10_000_000):
        self.strategy = strategy
        self.params = params
        self.initial_capital = initial_capital
        self.run_id = f"{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Load active filter rules
        self.filters = load_active_filters(strategy)
        self.filtered_count = 0
        self.signal_count = 0
        self.rr_rejected_count = 0

        # State
        self.capital = initial_capital
        self.position = None  # Current open position
        self.trades = []      # Completed trades
        self.equity_curve = []

    def run(self, df: pd.DataFrame, pair: str, timeframe: str) -> dict:
        """Execute the backtest.

        Args:
            df: OHLCV DataFrame
            pair: Trading pair name
            timeframe: Candle timeframe

        Returns:
            dict with performance metrics
        """
        # Apply pattern detection using strategy registry
        if self.strategy not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy: {self.strategy}. "
                           f"Available: {list(STRATEGY_REGISTRY.keys())}")

        detect_fn, signal_col, levels_fn = STRATEGY_REGISTRY[self.strategy]
        df = detect_fn(df, self.params)

        risk_cfg = self.params.get("risk", {
            "max_position_pct": 0.05,
            "max_daily_loss_pct": 0.03,
        })
        max_pos_pct = risk_cfg.get("max_position_pct", 0.05)
        strategy_params = self.params.get(self.strategy, self.params)

        for i in range(50, len(df)):
            price = df.iloc[i]["close"]

            # ── Check exit conditions for open position ──
            if self.position is not None:
                self._check_exit(df, i)

            # ── Check entry conditions ──
            if self.position is None and df.iloc[i].get(signal_col) == "buy":
                self.signal_count += 1

                # Calculate trade levels using the registered function
                levels = levels_fn(df, i, strategy_params)

                # Check risk-reward
                min_rr = strategy_params.get("min_risk_reward", 2.0)
                if levels["risk_reward"] < min_rr and levels.get("target") is not None:
                    self.rr_rejected_count += 1
                    continue

                # Position sizing
                risk_amount = self.capital * max_pos_pct
                if levels["risk"] > 0:
                    position_size = risk_amount / levels["risk"]
                else:
                    continue

                cost = position_size * levels["entry"]
                if cost > self.capital:
                    position_size = self.capital * 0.95 / levels["entry"]

                # Extract features
                features = extract_features(df, i)

                # Apply filter rules
                if self.filters:
                    should_skip, skip_reason = apply_filters(features, self.filters)
                    if should_skip:
                        self.filtered_count += 1
                        continue

                # Open position
                self.position = {
                    "pair": pair,
                    "timeframe": timeframe,
                    "entry_idx": i,
                    "entry_time": str(df.index[i]),
                    "entry_price": levels["entry"],
                    "stop_loss": levels["stop_loss"],
                    "target": levels.get("target"),
                    "position_size": position_size,
                    "is_ath": levels.get("is_ath", False),
                    "features": features,
                    "levels": levels,
                }

            # Track equity
            equity = self.capital
            if self.position:
                unrealized = (price - self.position["entry_price"]) * self.position["position_size"]
                equity += unrealized
            self.equity_curve.append({"timestamp": str(df.index[i]), "equity": equity})

        # Force close any remaining position at end
        if self.position is not None:
            self._close_position(df, len(df) - 1, "end_of_data")

        # Calculate performance
        return self._calculate_performance(pair, timeframe)

    def _check_exit(self, df: pd.DataFrame, i: int):
        """Check if the current position should be closed."""
        pos = self.position
        price = df.iloc[i]["close"]
        low = df.iloc[i]["low"]
        high = df.iloc[i]["high"]

        # Stop loss hit
        if low <= pos["stop_loss"]:
            self._close_position(df, i, "stop_loss", exit_price=pos["stop_loss"])
            return

        # Target hit (if not ATH breakout)
        if pos["target"] is not None and high >= pos["target"]:
            self._close_position(df, i, "target", exit_price=pos["target"])
            return

        # ATH trailing stop: use 2x ATR as trailing stop
        if pos.get("is_ath", False):
            from src.indicators.features import _atr
            atr = _atr(
                df["high"].values[:i+1],
                df["low"].values[:i+1],
                df["close"].values[:i+1],
                14
            )
            trailing_stop = high - (2 * atr)
            if trailing_stop > pos["stop_loss"]:
                pos["stop_loss"] = trailing_stop

    def _close_position(self, df: pd.DataFrame, i: int, reason: str,
                        exit_price: float = None):
        """Close the current position and record the trade."""
        pos = self.position
        if exit_price is None:
            exit_price = df.iloc[i]["close"]

        pnl_per_unit = exit_price - pos["entry_price"]
        pnl_absolute = pnl_per_unit * pos["position_size"]
        pnl_pct = pnl_per_unit / pos["entry_price"]

        self.capital += pnl_absolute

        trade = {
            "run_id": self.run_id,
            "strategy": self.strategy,
            "pair": pos["pair"],
            "timeframe": pos["timeframe"],
            "direction": "long",
            "entry_time": pos["entry_time"],
            "entry_price": pos["entry_price"],
            "exit_time": str(df.index[i]),
            "exit_price": exit_price,
            "stop_loss": pos["stop_loss"],
            "take_profit": pos.get("target"),
            "position_size": pos["position_size"],
            "pnl_pct": pnl_pct,
            "pnl_absolute": pnl_absolute,
            "exit_reason": reason,
            "features": pos.get("features", {}),
        }
        self.trades.append(trade)
        self.position = None

    def _calculate_performance(self, pair: str, timeframe: str) -> dict:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {
                "run_id": self.run_id,
                "strategy": self.strategy,
                "pair": pair,
                "timeframe": timeframe,
                "total_trades": 0,
                "signal_count": self.signal_count,
                "rr_rejected": self.rr_rejected_count,
                "filtered_trades": self.filtered_count,
                "message": "No trades generated",
            }

        pnls = [t["pnl_pct"] for t in self.trades]
        pnl_abs = [t["pnl_absolute"] for t in self.trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        # Equity curve analysis
        equities = [e["equity"] for e in self.equity_curve]
        peak = equities[0]
        max_dd = 0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (annualized, assuming hourly if timeframe is 1h)
        pnl_arr = np.array(pnls)
        sharpe = 0
        if len(pnl_arr) > 1 and np.std(pnl_arr) > 0:
            sharpe = np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(252)

        # Profit factor
        gross_profit = sum(p for p in pnl_abs if p > 0)
        gross_loss = abs(sum(p for p in pnl_abs if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Win rate by exit reason
        exit_reasons = {}
        for t in self.trades:
            r = t["exit_reason"]
            if r not in exit_reasons:
                exit_reasons[r] = {"count": 0, "wins": 0}
            exit_reasons[r]["count"] += 1
            if t["pnl_pct"] > 0:
                exit_reasons[r]["wins"] += 1

        return {
            "run_id": self.run_id,
            "strategy": self.strategy,
            "pair": pair,
            "timeframe": timeframe,
            "total_trades": len(self.trades),
            "signal_count": self.signal_count,
            "rr_rejected": self.rr_rejected_count,
            "filtered_trades": self.filtered_count,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(self.trades),
            "avg_win": np.mean(winners) if winners else 0,
            "avg_loss": np.mean(losers) if losers else 0,
            "largest_win": max(pnls),
            "largest_loss": min(pnls),
            "total_return": (self.capital - self.initial_capital) / self.initial_capital,
            "final_capital": self.capital,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "profit_factor": profit_factor,
            "exit_reasons": exit_reasons,
        }

    def save_to_db(self):
        """Save all trades to the database."""
        if self.trades:
            conn = get_connection()
            save_backtest_trades(conn, self.trades)
            conn.close()
            print(f"Saved {len(self.trades)} trades to database (run: {self.run_id})")
