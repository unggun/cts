"""CLI runner for backtests."""
import argparse
import json

from src.config import load_config
from src.data.database import get_connection, init_db, load_ohlcv, load_latest_parameters
from src.backtest.engine import BacktestEngine


def run_backtest(strategy: str, pair: str, timeframe: str = "1h",
                 config: dict = None) -> dict:
    """Run a single backtest and return results.

    Args:
        strategy: Strategy name ('darvas' or 'dbw')
        pair: Trading pair
        timeframe: Candle timeframe
        config: Optional config override

    Returns:
        Performance metrics dict
    """
    if config is None:
        config = load_config()

    init_db()
    conn = get_connection()

    # Load data
    df = load_ohlcv(conn, pair, timeframe)
    if df.empty:
        print(f"No data found for {pair} {timeframe}. Run the downloader first.")
        conn.close()
        return {}

    print(f"Loaded {len(df)} candles for {pair} {timeframe}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Load parameters — prefer DB version (auto-learned), fall back to config
    params = load_latest_parameters(conn, strategy)
    if params is None:
        params = config.get("strategy", {}).get(strategy, {})
        params["risk"] = config.get("risk", {})
        print(f"Using config parameters for {strategy}")
    else:
        params["risk"] = config.get("risk", {})
        print(f"Using learned parameters for {strategy}")

    # Run backtest
    initial_capital = config.get("simulation", {}).get("initial_capital", 10_000_000)
    engine = BacktestEngine(strategy, params, initial_capital)
    results = engine.run(df, pair, timeframe)

    # Save trades
    engine.save_to_db()
    conn.close()

    # Print results
    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS: {strategy.upper()} on {pair} ({timeframe})")
    print("=" * 60)

    if results.get("total_trades", 0) == 0:
        signals = results.get("signal_count", 0)
        rr_rej = results.get("rr_rejected", 0)
        filtered = results.get("filtered_trades", 0)
        print("No trades generated.")
        if signals > 0:
            print(f"  Diagnostics: {signals} signals detected, "
                  f"{rr_rej} rejected by risk-reward, "
                  f"{filtered} rejected by filter rules")
        else:
            print(f"  Diagnostics: 0 signals — strategy pattern not found in data")
        return results

    print(f"Total trades:    {results['total_trades']}")
    print(f"Win rate:        {results['win_rate']:.1%}")
    print(f"Winners/Losers:  {results['winners']}/{results['losers']}")
    print(f"Avg win:         {results['avg_win']:.2%}")
    print(f"Avg loss:        {results['avg_loss']:.2%}")
    print(f"Largest win:     {results['largest_win']:.2%}")
    print(f"Largest loss:    {results['largest_loss']:.2%}")
    print(f"Total return:    {results['total_return']:.2%}")
    print(f"Max drawdown:    {results['max_drawdown']:.2%}")
    print(f"Sharpe ratio:    {results['sharpe_ratio']:.2f}")
    print(f"Profit factor:   {results['profit_factor']:.2f}")
    print(f"\nExit reasons:")
    for reason, stats in results.get("exit_reasons", {}).items():
        wr = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {reason}: {stats['count']} trades ({wr:.0%} win rate)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--strategy", required=True,
                        choices=["darvas", "dbw", "candlestick", "sr_breakout", "support_resistance", "ma_crossover", "ema_crossover", "rsi_vwap", "cvd", "macd"],
                        help="Strategy to backtest")
    parser.add_argument("--pair", required=True, help="Trading pair (e.g. BTC/IDR)")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    args = parser.parse_args()

    run_backtest(args.strategy, args.pair, args.timeframe)
