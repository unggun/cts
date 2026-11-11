# Research Journal — Design

## Why

Every Claude proposal is now tested on the 90-day holdout. The promoted versions
then trade on paper. Nothing connects those two parts. We can't tell whether the
holdout predicts paper results, and we can't tell why a proposal was made, beyond
whether the bundle as a whole won. Before any strategy goes live, we need
evidence that a version which wins on the holdout also holds up on paper.

## Scope (agreed 2026-11-08)

- **A. Holdout vs paper, per promoted version.** Compare paper results with what
  the holdout predicted, and flag versions that are clearly behind.
- **B. Claude's hypothesis and prediction, per proposal.** Claude states one main
  idea and the numbers it expects on the holdout, so the journal can show
  predicted, holdout and paper side by side.
- **D. A way to read it.** `python -m src.learning.journal` and the Telegram
  `/journal` command.

Not in scope: C (a long-term list of tried ideas shown to Claude). It needs a few
weeks of A and B entries first, and P3 memory may take it over.

**Decision (my call, the user didn't choose):** a version that falls behind is
reported only, in the learning-cycle Telegram summary and in `/journal`. It is
never rolled back automatically.

## Design

### Linking paper trades to versions

- New column `trade_journal.param_id`, holding the `strategy_parameters.id` the
  paper trader was trading when it opened the position.
- The paper trader already tracks `param_row_id`, and now writes it on every
  insert.
- Older rows have no `param_id`. They are assigned by time: the version whose
  `created_at` is at or before the trade's `entry_time`, and before the next
  version's `created_at`.
  - Watch the time zones: `created_at` is SQLite `datetime('now')` (UTC), while
    `entry_time` is the trader's local `datetime.now()`. UTC times are converted
    to local before comparing.
- The live trader is not changed. The journal reads paper trades only.
- A promotion also stores the new row's `param_id` in `evaluation_json`, and the
  journal links each review to its version by that id. Version numbers restart
  at 1 after `--reset-params`, so a number alone could point an old review at a
  new version. Evaluations saved before this change have only the number. They
  are linked by number only if that version was saved within an hour after the
  review.

### Paper stats per version

For each version, from the closed paper trades assigned to it:
- trades
- expectancy (mean `pnl_pct`, already net of costs)
- win rate
- trades per week
- max drawdown of the cumulative `pnl_absolute` curve, measured against
  `simulation.initial_capital`

The holdout side comes from `strategy_parameters.performance_json["holdout"]`.

### Verdict (A)

| Verdict | Condition |
|---|---|
| `NO_PREDICTION` | the version has no holdout metrics (legacy versions) |
| `TOO_FEW` | fewer than 20 closed paper trades |
| `BEHIND` | paper expectancy + 2 standard errors < holdout expectancy |
| `ON_TRACK` | none of the above |

`BEHIND` means the paper shortfall is too large to be noise at roughly 95%
confidence.

### Hypothesis (B)

Claude's JSON gains:

```json
"hypothesis": {"idea": "...", "expected_expectancy_pct": 0.25, "expected_trades": 80}
```

It is stored as part of `recommendations_json`, so no schema change is needed.
The journal parses it defensively: missing or non-numeric values show as "—".
The per-change reasons (`parameter_changes[].reason`,
`filter_rules[].expected_impact`) already exist and are shown in the
single-strategy view.

### Journal entries (D)

- **`python -m src.learning.journal`:** one line per strategy, showing the
  current version, its holdout result, its paper result and the verdict.
- **`--strategy s [--limit 10] [--json]`:**
  - The current version block.
  - Then the strategy's Claude reviews, newest first. For each review: date,
    confidence, hypothesis and prediction, and the outcome (promoted vN / all
    filters removed vN / rejected: reason / not evaluated: reason). For evaluated
    reviews, the holdout result of the candidate and the incumbent. For promoted
    ones, the paper result and the verdict.
- **Telegram:** `/journal` gives the overview and `/journal <strategy>` gives the
  detail (last 5 reviews).
- **Learning-cycle summary:** a footer lists any `BEHIND` versions.

## Left to the builder

The output formatting, and the module layout (`src/learning/journal.py`).
