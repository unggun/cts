# Readiness–Telegram Integration — Design

Date: 2026-04-19

## Purpose

Expose the existing live-readiness check (built earlier today in `src/execution/live_readiness.py`) through two Telegram surfaces so the operator never has to SSH to the VPS to see readiness state:

1. Append a compact readiness column to the weekly multi-strategy learning-cycle Telegram summary.
2. Add a `/readiness` command to the Telegram bot for on-demand queries.

Both reuse the same underlying `build_report()` function. Flipping paper → live remains manual.

## Scope

| In scope | Out of scope |
|---|---|
| Extend `_send_multi_strategy_telegram` in `src/run_learning_cycle.py` | New systemd timer |
| Add `/readiness` and `/readiness <strategy>` handler to `src/execution/telegram_bot.py` | Alerts / notifications when status flips |
| New `short_summary(report) -> str` helper in `src/execution/live_readiness.py` | Persisting readiness history |
| Update `/help` text | Changing single-strategy weekly message |

## Changes

### 1. `src/execution/live_readiness.py`

Add one helper function — a single source of truth for the one-line format used by both surfaces.

```python
_STATUS_ICON = {"READY": "🟢", "WATCH": "🟡", "BLOCKED": "🔴", "INSUFFICIENT_DATA": "⚪"}
# (already exists)

def short_summary(report: dict) -> str:
    """One-line summary: just the status icon and label.

    Example: '🟡 WATCH'
    """
    return f"{_STATUS_ICON.get(report['status'], '')} {report['status']}"
```

### 2. `src/run_learning_cycle.py` — weekly summary extension

Modify `_send_multi_strategy_telegram` (currently at line 482). For each strategy, call `build_report(strategy)` and append the status icon to the existing WR line:

Before:
```
*ma_crossover*: WR 55% (v4)
*candlestick*: WR 48% (v3)
```

After:
```
*ma_crossover*: WR 55% (v4) 🟡
*candlestick*: WR 48% (v3) 🔴
```

And add a one-line footer *before* the existing `Run parameter_store --trend` footer:
```
Readiness: 0 READY | 2 WATCH | 1 BLOCKED | 0 INSUFFICIENT_DATA
```

If `build_report(strategy)` raises for a given strategy, catch the exception and simply omit the status icon for that row — never block the whole summary. The readiness footer counts only strategies where `build_report` succeeded.

Edge case: if `get_performance_trend(strategy)` returns empty (no parameter history yet) but `build_report` succeeds, still emit a line in the form `*<strategy>*: 🟡 WATCH` (no WR segment). A strategy with neither trend nor readiness data is simply skipped, same as today.

### 3. `src/execution/telegram_bot.py` — `/readiness` command

Add a handler method `_cmd_readiness(args: list) -> str`:

- **Zero args:** one message, one line per strategy in `KNOWN_STRATEGIES`, plus a summary footer.
  ```
  📊 *Live-Readiness*

  *ma_crossover*: 🟡 WATCH
  *candlestick*: 🔴 BLOCKED
  *dbw*: 🟡 WATCH
  ...

  _0 READY | 2 WATCH | 8 BLOCKED | 0 INSUFFICIENT_DATA_
  ```
  Telegram message length limit is 4096 chars; 10 strategies × ~35 chars = well within limits.

- **One arg (strategy name):**
  - If in `KNOWN_STRATEGIES`: return the full `format_text(build_report(strategy))` output wrapped in ```` ``` ```` so it renders in fixed-width font with all gate rows.
  - If not in `KNOWN_STRATEGIES`: return `Unknown strategy: <name>\nKnown: darvas, dbw, candlestick, ...` listing all 10.

Wire into existing `handlers` dict at line 112:
```python
"/readiness": self._cmd_readiness,
"/ready": self._cmd_readiness,
```

Update `/help` text at line 154 to include:
```
`/readiness` — per-strategy paper→live readiness
`/readiness <strategy>` — full gate detail for one strategy
```

## Error handling / graceful degradation

- `build_report` raising for any single strategy in the weekly summary → omit the icon for that row; the rest of the message still sends.
- `build_report` raising in `/readiness` (multi) → show `*<strategy>*: ⚠️ error` for that line; summary footer counts only successful strategies.
- `build_report` raising in `/readiness <strategy>` → return a plain error message (`Error evaluating <strategy>: <exception class>`).
- Telegram bot already has plain-text fallback in `_send` if Markdown fails.
- Unknown strategy arg → bot returns known-strategy list.

## Testing

- Unit test `short_summary()` with sample report dicts for each of the four statuses.
- No unit tests for `_send_multi_strategy_telegram` — the existing one has none and this is an incremental extension. Manual verification.
- No unit tests for `_cmd_readiness` — existing bot handlers have none. Manual verification by running the bot locally.
- Manual: run `python -m src.run_learning_cycle --all-strategies --dry-run` and inspect the generated Telegram message.
- Manual: run `python -m src.execution.telegram_bot` locally and send `/readiness`, `/readiness dbw`, `/readiness bogus`, `/help`.

## YAGNI boundaries

- No readiness alerts / flip-detection.
- No readiness history.
- No changes to single-strategy weekly message.
- No `--readiness-only` flag on `run_learning_cycle`.
- No JSON output for `/readiness` (the JSON CLI already exists).
