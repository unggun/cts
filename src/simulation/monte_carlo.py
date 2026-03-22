"""Monte Carlo simulator for trade strategy robustness testing.

Simulates thousands of random equity paths by sampling from actual
backtest trade results to estimate:
- Probability of ruin
- Expected drawdown distribution
- Confidence intervals on final equity
"""
import argparse
import json
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.database import get_connection, init_db, load_backtest_trades


class MonteCarloSimulator:
    """Run Monte Carlo simulations on backtest trade results."""

    def __init__(self, num_paths: int = 10_000, initial_capital: float = 10_000_000,
                 ruin_threshold: float = 0.5):
        """
        Args:
            num_paths: Number of simulated equity paths
            initial_capital: Starting capital
            ruin_threshold: Capital fraction below which = "ruin" (0.5 = 50% loss)
        """
        self.num_paths = num_paths
        self.initial_capital = initial_capital
        self.ruin_threshold = ruin_threshold

    def simulate(self, trade_returns: np.ndarray, trades_per_path: int = None) -> dict:
        """Run the Monte Carlo simulation.

        Args:
            trade_returns: Array of percentage returns per trade (from backtest)
            trades_per_path: Number of trades per simulated path
                            (default: same as input length)

        Returns:
            dict with simulation results
        """
        if len(trade_returns) < 5:
            return {"error": "Need at least 5 trades for simulation", "verdict": "INSUFFICIENT_DATA"}

        if trades_per_path is None:
            trades_per_path = len(trade_returns)

        ruin_level = self.initial_capital * self.ruin_threshold

        # Storage
        final_equities = np.zeros(self.num_paths)
        max_drawdowns = np.zeros(self.num_paths)
        ruin_count = 0

        for path in range(self.num_paths):
            # Random sample of trades (with replacement)
            sampled = np.random.choice(trade_returns, size=trades_per_path, replace=True)

            equity = self.initial_capital
            peak = equity
            max_dd = 0.0

            for ret in sampled:
                equity *= (1 + ret)

                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

                if equity < ruin_level:
                    ruin_count += 1
                    break

            final_equities[path] = equity
            max_drawdowns[path] = max_dd

        # Calculate statistics
        prob_ruin = ruin_count / self.num_paths
        median_equity = np.median(final_equities)
        mean_equity = np.mean(final_equities)

        # Percentiles
        p5 = np.percentile(final_equities, 5)
        p25 = np.percentile(final_equities, 25)
        p75 = np.percentile(final_equities, 75)
        p95 = np.percentile(final_equities, 95)

        # Drawdown stats
        median_dd = np.median(max_drawdowns)
        worst_dd = np.max(max_drawdowns)
        p95_dd = np.percentile(max_drawdowns, 95)

        # Determine verdict
        if prob_ruin > 0.10:
            verdict = "FRAGILE"
            verdict_emoji = "🔴"
            verdict_note = "High probability of ruin. Fix position sizing and filters before trading."
        elif prob_ruin > 0.03:
            verdict = "RISKY"
            verdict_emoji = "🟡"
            verdict_note = "Moderate risk. Consider reducing position size or adding filters."
        elif median_equity < self.initial_capital:
            verdict = "UNPROFITABLE"
            verdict_emoji = "🟠"
            verdict_note = "Strategy doesn't generate positive returns on average."
        else:
            verdict = "ROBUST"
            verdict_emoji = "🟢"
            verdict_note = "Strategy shows positive expectancy with acceptable risk."

        return {
            "num_paths": self.num_paths,
            "num_source_trades": len(trade_returns),
            "trades_per_path": trades_per_path,
            "initial_capital": self.initial_capital,
            "median_final_equity": median_equity,
            "mean_final_equity": mean_equity,
            "worst_final_equity": np.min(final_equities),
            "best_final_equity": np.max(final_equities),
            "p5_equity": p5,
            "p25_equity": p25,
            "p75_equity": p75,
            "p95_equity": p95,
            "probability_of_ruin": prob_ruin,
            "max_drawdown_median": median_dd,
            "max_drawdown_worst": worst_dd,
            "max_drawdown_p95": p95_dd,
            "median_return": (median_equity - self.initial_capital) / self.initial_capital,
            "verdict": verdict,
            "verdict_emoji": verdict_emoji,
            "verdict_note": verdict_note,
        }


def run_simulation(run_id: str = None, strategy: str = None) -> dict:
    """Run Monte Carlo simulation on backtest results from DB."""
    config = load_config()
    sim_cfg = config.get("simulation", {})

    init_db()
    conn = get_connection()
    trades_df = load_backtest_trades(conn, run_id=run_id, strategy=strategy)
    conn.close()

    if trades_df.empty:
        print("No backtest trades found. Run a backtest first.")
        return {}

    # Filter to completed trades with PnL
    trades_df = trades_df[trades_df["pnl_pct"].notna()]
    returns = trades_df["pnl_pct"].values

    print(f"Running Monte Carlo with {len(returns)} trades...")
    print(f"Source trade stats: mean={np.mean(returns):.3%}, "
          f"std={np.std(returns):.3%}, win_rate={np.mean(returns > 0):.1%}")

    sim = MonteCarloSimulator(
        num_paths=sim_cfg.get("num_paths", 10_000),
        initial_capital=sim_cfg.get("initial_capital", 10_000_000),
    )
    results = sim.simulate(returns)

    # Print results
    print("\n" + "=" * 60)
    print(f"MONTE CARLO SIMULATION RESULTS")
    print("=" * 60)
    print(f"Paths simulated:       {results['num_paths']:,}")
    print(f"Trades per path:       {results['trades_per_path']}")
    print(f"Initial capital:       {results['initial_capital']:,.0f}")
    print(f"")
    print(f"Median final equity:   {results['median_final_equity']:,.0f}")
    print(f"Mean final equity:     {results['mean_final_equity']:,.0f}")
    print(f"Best case:             {results['best_final_equity']:,.0f}")
    print(f"Worst case:            {results['worst_final_equity']:,.0f}")
    print(f"")
    print(f"Probability of ruin:   {results['probability_of_ruin']:.2%}")
    print(f"Max drawdown (median): {results['max_drawdown_median']:.2%}")
    print(f"Max drawdown (worst):  {results['max_drawdown_worst']:.2%}")
    print(f"Max drawdown (95th):   {results['max_drawdown_p95']:.2%}")
    print(f"")
    print(f"Verdict: {results['verdict_emoji']} {results['verdict']}")
    print(f"  {results['verdict_note']}")

    # Save to DB
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO simulation_results
        (run_id, strategy, num_paths, initial_capital, median_final_equity,
         mean_final_equity, worst_final_equity, best_final_equity,
         probability_of_ruin, max_drawdown_median, max_drawdown_worst,
         sharpe_ratio, profit_factor, verdict, details_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id or "all", strategy or "all", results["num_paths"],
        results["initial_capital"], results["median_final_equity"],
        results["mean_final_equity"], results["worst_final_equity"],
        results["best_final_equity"], results["probability_of_ruin"],
        results["max_drawdown_median"], results["max_drawdown_worst"],
        None, None, results["verdict"], json.dumps(results)
    ))
    conn.commit()
    conn.close()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo simulation")
    parser.add_argument("--run-id", default=None, help="Specific backtest run ID")
    parser.add_argument("--strategy", default=None, help="Strategy name filter")
    args = parser.parse_args()

    run_simulation(run_id=args.run_id, strategy=args.strategy)
