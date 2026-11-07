"""Exit rules shared by the backtest engine and the paper trader, so paper
results validate the same logic the backtest measured."""


def trail_stop(entry_price: float, stop_loss: float, highest_high: float,
               atr: float, trail_mult: float) -> float:
    """Return the ratcheted stop for a long position.

    The trail activates once price has moved one ATR in our favor, then sits
    ``trail_mult`` ATRs below the highest high. It never lowers the stop.
    """
    if highest_high > entry_price + atr:
        trailing = highest_high - trail_mult * atr
        if trailing > stop_loss:
            return trailing
    return stop_loss
