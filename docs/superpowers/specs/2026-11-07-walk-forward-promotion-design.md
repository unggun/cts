# Walk-forward promotion: propose → verify → promote

## Why

Until now the learning cycle either saved Claude's suggested parameters
untested, or blocked every update when the *current* version failed the
quality gates or Monte Carlo. After fees were added, most strategies fail
those gates, so no strategy could improve: Claude's proposals were thrown
away. Filters were saved untested regardless.

The fix: judge each proposal by whether it beats the current version on
recent data that neither the analyzer nor Claude saw.

## Design (defaults chosen without user input; all configurable)

**Holdout split.** The newest `learning.promotion.holdout_days` (90) of
candles are held out. The cycle's backtests, winner/loser analysis, Monte
Carlo and Claude's prompt use only the older "train" window.

**Candidate bundle.** Claude's `updated_parameters` (or the current params
when it proposes none) plus the proposed filter set: Claude's and the
analyzer's rules, capped by `select_filters` (3 strongest, one per feature).
The incumbent is the current params plus the currently active filters.

**Evaluation.** Both bundles are backtested on the holdout window, after
fees, with indicators warmed up on earlier data (entries only from the
holdout start). These runs are not saved to `backtest_trades`.

**Promotion rule.** Promote only if all hold:
- candidate has ≥ `min_holdout_trades` (20) holdout trades
- candidate keeps ≥ `min_trade_ratio` (50%) of the incumbent's holdout trades
- if the incumbent has enough trades: candidate expectancy (mean net
  pnl_pct per trade) ≥ incumbent + `min_expectancy_gain` (0.05%), and
  candidate max drawdown ≤ incumbent + `max_drawdown_worsening` (5 pp)
- if the incumbent has too few trades: candidate expectancy > 0
- the bundle actually differs from the incumbent

Claude confidence "low" still skips evaluation. Quality gates and the Monte
Carlo verdict are still computed and reported, but no longer block
promotion. Live readiness (separate module) stays the gate for live trading.

**On promotion:** save a new `strategy_parameters` version (holdout metrics
in `performance_json`) and make the candidate filter set the only active
filters. **On rejection:** save nothing but the evaluation.

**Memory.** Each evaluation is stored on its `learning_sessions` row
(`evaluation_json`, `applied`). The next prompt includes the last 3
proposals with their holdout outcome, so Claude stops repeating rejected
ideas, and the prompt tells Claude how proposals are judged.

**Paper traders** reload params when a new version appears and refresh
filters every loop, so promotions reach paper trading without restarts.

## Non-goals (later)

- Rolling multi-fold walk-forward; shadow paper-trading of challengers.
- Analyzing the unfiltered signal population.
- Changing the live trader.
