# Live-Readiness Check — Design

Date: 2026-04-19
Author: brainstorming session with Claude

## Purpose

Provide a read-only CLI that evaluates, per strategy, whether the system is ready to flip from paper to live trading. Flipping itself remains manual. The tool aggregates signals that already exist in the database (Monte Carlo verdicts, Claude confidence, paper-trade outcomes) and checks them against a fixed set of gates.

Today (2026-04-19) we had to manually query four tables and cross-reference JSON blobs to answer "is anything ready?". This script replaces that manual work.

## CLI

```
python -m src.execution.live_readiness               # report on all strategies
python -m src.execution.live_readiness --strategy dbw
python -m src.execution.live_readiness --json
```

Style and entry-point conventions follow `src/execution/positions.py`.

## Data sources (all read-only)

| Source | What we read |
|---|---|
| `simulation_results` | Latest row per strategy → `verdict`, `probability_of_ruin`, `details_json.num_source_trades` |
| `learning_sessions` | Latest `session_type='claude_review'` row per strategy → `recommendations_json.confidence` |
| `strategy_parameters` | Latest (max version) row per strategy → `created_at` used as the param-cutover timestamp |
| `trade_journal` (mode='paper') | Trades in the last 30 days, partitioned by entry_time relative to cutover |

No schema changes. No writes.

## Paper-trade partitioning (Option C — Hybrid)

For each strategy, fetch paper trades from the last 30 days, then split:

- `current` — `entry_time >= strategy_parameters.created_at`
- `pre_update` — `entry_time <  strategy_parameters.created_at`

Gates are evaluated against the `current` bucket only. The `pre_update` bucket is displayed for context but does not gate the READY status.

## Gates

All must pass for `READY`:

| Gate | Threshold | Source |
|---|---|---|
| MC verdict | `ROBUST` | `simulation_results.verdict` |
| Probability of ruin | < 3% | `simulation_results.probability_of_ruin` |
| MC source trades | ≥ 50 | `details_json.num_source_trades` |
| Claude confidence | `high` | latest `learning_sessions.recommendations_json.confidence` |
| Current paper trades | ≥ 30 closed | count where `exit_time IS NOT NULL` |
| Current paper win rate | ≥ 50% | `pnl_pct > 0` / closed |
| Current paper total PnL | > 0 | sum of `pnl_pct` on closed |

## Status tiers

Evaluated in order; first match wins.

1. `INSUFFICIENT_DATA` — MC source trades < 50 (sample too small for verdict to be trusted)
2. `BLOCKED` — any hard failure:
   - MC verdict is `FRAGILE` or `RISKY`, OR
   - Probability of ruin ≥ 3%, OR
   - Claude confidence is `low` (including the case where no learning session exists), OR
   - Current-param paper sample ≥ 30 closed trades AND total PnL ≤ 0
3. `WATCH` — not BLOCKED, but at least one of:
   - Current-param paper sample < 30 closed trades (sample still growing), OR
   - Current-param paper WR < 50% with positive PnL, OR
   - Claude confidence is `medium`
4. `READY` — all gates pass (MC verdict ROBUST, ruin < 3%, confidence `high`, ≥ 30 current-param paper trades closed, WR ≥ 50%, total PnL > 0)

## Output

Per strategy:

```
Strategy: ma_crossover                              Status: WATCH
  MC verdict ............ ROBUST         ✓
  Probability of ruin ... 1.8%           ✓
  MC source trades ...... 321            ✓
  Claude confidence ..... high           ✓
  Paper (current params)  2 closed, -%, +0.00% PnL  ✗  need 30
  Paper (pre-update)      12 closed, 50% WR, +3.5% PnL  (informational)
  Param version since ... 2026-04-18 22:01
```

Footer: `X READY | Y WATCH | Z BLOCKED | W INSUFFICIENT_DATA` across all strategies.

`--json` emits a list of per-strategy dicts with the same fields plus a top-level `summary` count.

## Scope boundaries (YAGNI)

- Read-only. No config writes, no param edits, no auto-flip.
- No `trade_journal` schema change; param-version is inferred from `entry_time` vs `strategy_parameters.created_at`.
- No multi-version history in the report — only the current snapshot.
- No Telegram/notification integration. The CLI prints to stdout; wrapping it in a Telegram command is a separate concern.
- No backtest-trade gates; backtest already feeds into MC, which is already a gate.

## Testing approach

- Smoke test on the live DB: run the CLI, verify it doesn't error and classifies today's known strategies correctly (e.g., `candlestick` → BLOCKED, `ma_crossover` → WATCH).
- Single-strategy filter: `--strategy dbw` returns only dbw.
- `--json` flag produces parseable JSON.
