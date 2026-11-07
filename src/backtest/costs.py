"""Trading costs shared by the backtest engine and the paper trader.

Without fees and slippage a backtest overstates every trade by the
round-trip cost (~0.3% at the defaults), which is a large share of the
average edge on 1h strategies.
"""

DEFAULT_FEE_PCT = 0.001        # per side; check your Tokocrypto fee tier
DEFAULT_SLIPPAGE_PCT = 0.0005  # per side; adverse price move on each fill


def trading_costs(config: dict = None) -> dict:
    """Read ``costs.fee_pct`` / ``costs.slippage_pct`` from config (per side)."""
    if config is None:
        from src.config import load_config
        config = load_config()
    costs = config.get("costs", {}) or {}
    return {
        "fee_pct": float(costs.get("fee_pct", DEFAULT_FEE_PCT)),
        "slippage_pct": float(costs.get("slippage_pct", DEFAULT_SLIPPAGE_PCT)),
    }


def net_pnl(entry_price: float, exit_price: float, size: float,
            fee_pct: float, slippage_pct: float) -> tuple[float, float]:
    """Net P&L of a long trade after slippage and fees on both sides.

    Returns ``(pnl_absolute, pnl_pct)`` where ``pnl_pct`` is relative to the
    total cash spent on entry (fill price plus entry fee).
    """
    buy_px = entry_price * (1 + slippage_pct)
    sell_px = exit_price * (1 - slippage_pct)
    cost = size * buy_px * (1 + fee_pct)
    proceeds = size * sell_px * (1 - fee_pct)
    pnl_abs = proceeds - cost
    pnl_pct = pnl_abs / cost if cost > 0 else 0.0
    return pnl_abs, pnl_pct
