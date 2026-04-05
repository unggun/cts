"""Query active positions for paper and live trading."""
import json
from src.data.database import get_connection, init_db


def get_open_positions(mode: str = None, strategy: str = None) -> list[dict]:
    """Get all open positions from trade_journal."""
    init_db()
    conn = get_connection()
    cur = conn.cursor()

    query = "SELECT * FROM trade_journal WHERE exit_time IS NULL"
    params = []
    if mode:
        query += " AND mode=?"
        params.append(mode)
    if strategy:
        query += " AND strategy=?"
        params.append(strategy)
    query += " ORDER BY mode, strategy, pair"

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def print_positions(positions: list[dict]):
    """Pretty-print positions to terminal."""
    if not positions:
        print("No open positions.")
        return

    current_mode = None
    current_strategy = None

    for pos in positions:
        if pos["mode"] != current_mode:
            current_mode = pos["mode"]
            print(f"\n{'=' * 60}")
            print(f"  {'🔴 LIVE' if current_mode == 'live' else '📄 PAPER'} TRADING")
            print(f"{'=' * 60}")

        if pos["strategy"] != current_strategy:
            current_strategy = pos["strategy"]
            print(f"\n  Strategy: {current_strategy.upper()}")
            print(f"  {'─' * 50}")

        target_str = f"{pos['take_profit']:,.0f}" if pos.get("take_profit") else "ATH trail"
        print(f"  {pos['pair']:12s} | Entry: {pos['entry_price']:>14,.0f} | "
              f"SL: {pos['stop_loss']:>14,.0f} | Target: {target_str:>14s} | "
              f"Size: {pos['position_size']:.6f}")
        print(f"  {'':12s} | Opened: {pos['entry_time'][:19]}")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Query active trading positions")
    parser.add_argument("--mode", choices=["paper", "live"],
                        help="Filter by trading mode")
    parser.add_argument("--strategy",
                        help="Filter by strategy name")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    positions = get_open_positions(mode=args.mode, strategy=args.strategy)

    if args.json:
        print(json.dumps(positions, indent=2, default=str))
    else:
        print_positions(positions)
        print(f"Total: {len(positions)} open position(s)")
