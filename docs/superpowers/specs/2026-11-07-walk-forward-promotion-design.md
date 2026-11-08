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

Claude confidence does not skip evaluation (changed 2026-11-08: in the first
run, "low" blocked 8 of 9 proposals because the prompt tied confidence to
effect sizes above 0.5, which this data never reaches). Quality gates and the Monte
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

## No-filters fallback (added 2026-11-07)

Filters learned before this design (April–May) were often inverted and
starved strategies (candlestick: 1 holdout trade). Each proposal was compared
with that starved set and could still lose, so the old filters stayed.

Now, when the proposal isn't promoted and the strategy has active filters,
the current parameters with no filters are evaluated under the same rule
(`decide`). If they pass, all filters are removed and a parameter version is
saved with source `no_filter_fallback`. It's skipped when the proposal itself
was "current params, no filters". Claude's history shows its proposal as
rejected, with a note that the filters were removed instead.

Dry run on production data (rule-search picks as the proposal, no Claude):
candlestick 1 trade / −2.76% → no filters 97 / +0.44% (removed); the other
eight strategies were decided by their proposal or kept their filters.
