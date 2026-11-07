# Learn filters from every signal

## Why

The analyzer only saw trades the active filters let through, so a rule's
blind spots stayed hidden. Rules were picked by Cohen's d between winners
and losers with a midpoint cutoff, which never asks whether skipping those
trades raises expectancy: a rule can drop more losers than winners and still
cut the few big winners. The cycle's "filters starved" fallback was a patch
for the first problem.

## Design (defaults chosen without user input; all configurable)

**Two train backtests per cycle.** With the active filters: the deployed
version, for reports, quality gates and Monte Carlo. Without filters: the
trades the analyzer and Claude learn from. When no signal was filtered the
second run is skipped. The starvation fallback is removed.

**Rule search** (`src/learning/rule_search.py`). Candidates are `>=`/`<=`
at each feature's deciles and `==` for categorical features (≤ 12 values).
A rule qualifies only if:
- it removes ≥ `min_removed` (30) trades; day/hour/session features need
  `time_evidence_mult` (2×) that,
- the whole set keeps ≥ `min_kept_ratio` (50%) of all trades,
- kept expectancy (mean net pnl per trade) rises by ≥ `min_gain` (0.05%),
- kept expectancy is not below the current one in either chronological
  half of the train window.

The best qualifying rule (largest gain) is taken, its trades removed, and
the rest re-scored, up to `learning.max_filter_rules` (3), one per feature.

**Claude** sees the unfiltered analysis, these rules with "removes N of M
trades; expectancy a → b", and the currently active filters.

**Promotion** on the 90-day holdout is unchanged and stays the final check.

## Evidence (2026-11-07, production data, analyzer filters only)

Holdout trades / expectancy per trade, old midpoint method vs rule search,
both on the same unfiltered train trades:

| strategy | no filters | old | new |
|---|---|---|---|
| dbw | 74 / −0.60% | 74 / −0.60% | 66 / −0.44% |
| candlestick | 97 / +0.43% | 23 / +0.04% | 52 / −0.02% |
| sr_breakout | 149 / +0.51% | 149 / +0.51% | 115 / +0.75% |
| ma_crossover | 194 / +0.39% | 194 / +0.39% | 136 / +0.39% |
| rsi_vwap | 15 / +0.39% | 15 / +0.39% | 13 / +0.56% |
| cvd | 211 / +0.00% | 211 / +0.00% | 145 / +0.04% |
| macd | 151 / −0.06% | 118 / −0.50% | 95 / −0.39% |
| mean_reversion | 86 / +0.05% | 20 / +0.20% | 39 / +0.20% |
| bb_squeeze | 28 / +0.97% | 5 / +1.93% | 28 / +0.97% |

(darvas had 11 train trades and none on the holdout.) Rule search is better
in 5, tied in 2, worse in 2 (candlestick, and bb_squeeze on 5 trades). Some
in-sample rules still fail out of sample (macd, candlestick: `return_1`),
which is why promotion remains the gate.

## Non-goals (later)

Learning from paper/live trades; rule expiry; multi-fold walk-forward.
